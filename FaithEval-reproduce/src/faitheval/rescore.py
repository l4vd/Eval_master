"""Re-score stored FaithEval predictions offline: no model, no torch, no ``datasets``.

A generation run writes ``<task>_predictions.jsonl`` (``index``, ``prediction``,
``correct``). Every rule below is a pure function of the prediction text (plus, for
``contains``, the gold answer joined from the local dataset JSONL by ``index``), so any
of them can be applied to a finished run for free.

These re-scorings feed the optional all-variants overview (Eval_master
``analysis/variants.py``). They are **descriptive**: none replaces the original number.

Rules (the names equal the overview's variant directory suffixes):

  strict     ``phrase_match`` against the task's ``strict_valid_phrases`` (``unknown``,
             ``conflict``) — what ``--strict-match`` scores, without a re-run.
  wordmatch  the task's ``valid_phrases``, matched at word boundaries in the normalized
             prediction, so ``not`` no longer matches inside ``cannot`` or ``nothing``.
  contains   counterfactual: any normalized reference occurs at word boundaries in the
             normalized prediction. Reported overall and per prediction-length stratum,
             with exact match for the same rows, because a verbose answer can only ever
             pass under containment.
  lenient    the original rule, recomputed. Verification only: it must reproduce every
             stored ``correct`` and writes nothing.

Imports only :mod:`faitheval.metrics` (stdlib) and :mod:`faitheval.config` (PyYAML), so it
runs in any interpreter with PyYAML.

Usage:
    python src/rescore.py <run>/faitheval/unanswerable_predictions.jsonl --rule strict \\
        --output-dir <run>/faitheval.strict
    python src/rescore.py <run>/faitheval/unanswerable_predictions.jsonl --rule lenient
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from faitheval.config import ANSWER_MATCH, PHRASE_MATCH, TaskConfig, load_task_config
from faitheval.metrics import answer_match, normalize_answer, phrase_match

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
RULES = ("strict", "wordmatch", "contains", "lenient")
PHRASE_RULES = ("strict", "wordmatch")
# Prediction length in whitespace words (as mean_prediction_words counts them).
# An empty prediction falls in the first stratum.
LENGTH_STRATA = (("1_5", 0, 5), ("6_20", 6, 20), ("21_60", 21, 60), ("61_plus", 61, None))
# Copied from the run's own summary so a re-scored summary is comparable on its own.
PROVENANCE_KEYS = ("model_id", "batch_size", "sort_by_length", "max_new_tokens",
                   "num_samples", "split", "strict_match", "dtype")


def word_match(prediction: str, phrases) -> bool:
    """True if any phrase occurs in the normalized prediction at word boundaries."""
    normalized = normalize_answer(prediction)
    return any(re.search(rf"\b{re.escape(normalize_answer(p))}\b", normalized) for p in phrases)


def contains_match(prediction: str, references) -> bool:
    """True if any non-empty normalized reference occurs at word boundaries in the prediction."""
    normalized = normalize_answer(prediction)
    for ref in references:
        ref = normalize_answer(str(ref))
        if ref and re.search(rf"\b{re.escape(ref)}\b", normalized):
            return True
    return False


def length_stratum(prediction: str) -> str:
    words = len(prediction.split())
    for name, low, high in LENGTH_STRATA:
        if words >= low and (high is None or words <= high):
            return name
    return LENGTH_STRATA[-1][0]  # pragma: no cover


def _references(row: dict[str, Any], task_config: TaskConfig) -> list[str]:
    reference = row[task_config.answer_column]
    return reference if isinstance(reference, list) else [reference]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def data_dir_default() -> Path:
    env = os.environ.get("FAITHEVAL_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data" / "faitheval"


def dataset_path(task_config: TaskConfig, split: str, data_dir: Path) -> Path:
    slug = task_config.dataset_name.rstrip("/").split("/")[-1]
    return Path(data_dir) / slug / f"{split}.jsonl"


def task_from_path(path: Path) -> str:
    name = Path(path).name
    if not name.endswith("_predictions.jsonl"):
        raise ValueError(f"{name} is not a <task>_predictions.jsonl file")
    return name[: -len("_predictions.jsonl")]


def rescore(
    predictions_path: Path,
    rule: str,
    *,
    task: str | None = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    data_dir: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Score one predictions file under ``rule``; returns the summary payload.

    For ``lenient`` the payload also carries ``num_mismatched`` against the stored
    ``correct`` (0 when the re-implementation reproduces the run).
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
    predictions_path = Path(predictions_path)
    task = task or task_from_path(predictions_path)
    task_config = load_task_config(Path(config_dir) / f"{task}.yaml")
    if rule in PHRASE_RULES and task_config.scoring != PHRASE_MATCH:
        raise ValueError(f"rule {rule!r} needs a phrase_match task; {task!r} is {task_config.scoring}")
    if rule == "contains" and task_config.scoring != ANSWER_MATCH:
        raise ValueError(f"rule 'contains' needs an answer_match task; {task!r} is {task_config.scoring}")
    if rule == "strict" and not task_config.strict_valid_phrases:
        raise ValueError(f"task {task!r} defines no strict_valid_phrases")

    rows = read_jsonl(predictions_path)
    source_summary = predictions_path.with_name(f"{task}_summary.json")
    provenance: dict[str, Any] = {}
    if source_summary.is_file():
        data = json.loads(source_summary.read_text(encoding="utf-8"))
        provenance = {k: data[k] for k in PROVENANCE_KEYS if k in data}
    split = split or provenance.get("split") or "test"

    gold: dict[int, dict[str, Any]] = {}
    needs_gold = task_config.scoring == ANSWER_MATCH
    if needs_gold:
        data_path = dataset_path(task_config, split, data_dir or data_dir_default())
        if not data_path.is_file():
            raise FileNotFoundError(f"rule {rule!r} on {task!r} needs the gold answers at {data_path}")
        gold = dict(enumerate(read_jsonl(data_path)))

    num_correct = num_mismatched = total_words = 0
    strata = {name: {"n": 0, "num_correct": 0, "num_exact": 0} for name, _, _ in LENGTH_STRATA}
    for row in rows:
        prediction = row.get("prediction") or ""
        total_words += len(prediction.split())
        if needs_gold:
            references = _references(gold[int(row["index"])], task_config)
        if rule == "strict":
            correct = phrase_match(prediction, task_config.strict_valid_phrases)
        elif rule == "wordmatch":
            correct = word_match(prediction, task_config.valid_phrases)
        elif rule == "contains":
            correct = contains_match(prediction, references)
            block = strata[length_stratum(prediction)]
            block["n"] += 1
            block["num_correct"] += int(correct)
            block["num_exact"] += int(answer_match(prediction, references))
        elif task_config.scoring == PHRASE_MATCH:
            correct = phrase_match(prediction, task_config.valid_phrases)
        else:
            correct = answer_match(prediction, references)
        num_correct += int(correct)
        if rule == "lenient" and "correct" in row and bool(row["correct"]) != correct:
            num_mismatched += 1

    n = len(rows)
    summary: dict[str, Any] = {
        "task": task,
        "variant": rule,
        "scoring": "generate",
        "source_predictions": predictions_path.name,
        "source_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        **provenance,
        "num_examples": n,
        "num_correct": num_correct,
        "accuracy": num_correct / n if n else 0.0,
        "mean_prediction_words": total_words / n if n else 0.0,
    }
    if rule in PHRASE_RULES:
        summary["phrases"] = list(task_config.strict_valid_phrases if rule == "strict"
                                  else task_config.valid_phrases)
    if rule == "contains":
        summary["length_strata"] = {
            name: {
                "n": b["n"],
                "accuracy": b["num_correct"] / b["n"] if b["n"] else None,
                "exact_match": b["num_exact"] / b["n"] if b["n"] else None,
            }
            for name, b in strata.items()
        }
        summary["exact_match"] = sum(b["num_exact"] for b in strata.values()) / n if n else 0.0
    if rule == "lenient":
        summary["num_mismatched"] = num_mismatched
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, help="a <task>_predictions.jsonl file")
    ap.add_argument("--rule", required=True, choices=RULES)
    ap.add_argument("--task", default=None, help="default: read from the file name")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="write <task>_summary.json here (the variant's own dir; not with lenient)")
    ap.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="FaithEval JSONL root (default: $FAITHEVAL_DATA_DIR or data/faitheval)")
    ap.add_argument("--split", default=None, help="default: the source summary's split, else test")
    args = ap.parse_args(argv)

    if args.rule == "lenient" and args.output_dir is not None:
        ap.error("--rule lenient is verification only and writes nothing")
    if args.rule != "lenient" and args.output_dir is None:
        ap.error(f"--rule {args.rule} needs --output-dir")
    if args.output_dir is not None and args.output_dir.resolve() == args.predictions.parent.resolve():
        raise SystemExit(f"refusing to write a re-scored summary next to {args.predictions.name}: that "
                         "directory holds the original run. Pass the variant's own directory.")

    summary = rescore(args.predictions, args.rule, task=args.task, config_dir=args.config_dir,
                      data_dir=args.data_dir, split=args.split)
    print(f"{summary['task']} [{args.rule}] accuracy {summary['accuracy']:.4f} "
          f"({summary['num_correct']}/{summary['num_examples']})")
    if args.rule == "lenient":
        if summary["num_mismatched"]:
            print(f"!! {summary['num_mismatched']} row(s) disagree with the stored `correct`", file=sys.stderr)
            return 1
        print("   reproduces every stored `correct`")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{summary['task']}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
