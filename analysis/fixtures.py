"""Synthetic summary generators that reproduce each benchmark's real on-disk shape.

Used by the offline test suite (no model downloads): given a target run dir and a set
of per-benchmark values, write ``<task>_summary.json`` (faitheval/halueval), nested
``summary.json`` (ragtruth), ``summary.csv`` pivot (truthfulqa), and the ``results``-list
``summary.json`` (harness), plus a sibling ``run_metadata.json`` carrying the seed.

Kept in the package (not just under tests/) so the CLI's ``--selftest`` / demo path and
docs can materialise a realistic fixture tree without a real eval run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


def write_run_metadata(run_dir: Path, seed: int | None) -> None:
    (run_dir).mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"seed": seed}, indent=2), encoding="utf-8"
    )


def write_faitheval(bench_dir: Path, per_task_accuracy: Mapping[str, float], n: int = 100,
                    **extra) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    for task, acc in per_task_accuracy.items():
        summary = {
            "task": task,
            "model_id": "synthetic",
            "num_examples": n,
            "num_correct": round(acc * n),
            "accuracy": acc,
            **extra,
        }
        (bench_dir / f"{task}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def write_halueval(
    bench_dir: Path, per_task_accuracy: Mapping[str, float], label: str = "synthetic", n: int = 100
) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    for task, acc in per_task_accuracy.items():
        correct = round(acc * n)
        summary = {
            "task": task,
            "model": label,
            "backend": "hf",
            "num_examples": n,
            "num_correct": correct,
            "num_incorrect": n - correct,
            "accuracy": acc,
        }
        (bench_dir / f"{task}_{label}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def write_halueval_constrained(
    bench_dir: Path, per_task_auroc: Mapping[str, float], label: str = "synthetic",
    n: int = 100, auroc_se: float = 0.02,
) -> None:
    """``<task>_<label>_constrained_summary.json`` as ``evaluate.py --scoring constrained`` writes it."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    for task, auc in per_task_auroc.items():
        summary = {
            "task": task, "model": label, "backend": "hf", "scoring": "constrained",
            "num_examples": n, "auroc": auc, "auroc_se": auroc_se, "accuracy_argmax": auc,
            "tpr": auc, "tnr": auc, "judged_yes_rate": 0.5, "mean_verdict_mass": 0.9,
            "median_verdict_mass": 0.95, "frac_mass_below_half": 0.05,
        }
        (bench_dir / f"{task}_{label}_constrained_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def write_halueval_decontam(
    bench_dir: Path, per_task: Mapping[str, tuple[float, float]], label: str = "synthetic",
    n: int = 100, n_seen: int = 10, constrained: bool = False,
) -> None:
    """``*_decontam_summary.json`` as ``score_results.decontam_summary`` writes it.

    ``per_task`` maps a task to ``(decontaminated value, seen value)`` — accuracies, or
    AUROCs with ``constrained=True``.
    """
    bench_dir.mkdir(parents=True, exist_ok=True)
    scoring, metric = ("constrained", "auroc") if constrained else ("strict", "accuracy")
    for task, (kept, seen) in per_task.items():
        def block(value: float, count: int) -> dict:
            return {"num_examples": count, scoring: {metric: value, f"{metric}_se": 0.01}}

        overall = (kept * (n - n_seen) + seen * n_seen) / n
        summary = {
            "task": task, "model": label, "variant": "decontam",
            "scoring": "constrained" if constrained else "generate",
            "n_excluded": n_seen, "num_examples": n - n_seen, metric: kept, f"{metric}_se": 0.01,
            "row_sets": {
                "original": block(overall, n), "decontaminated": block(kept, n - n_seen),
                "seen": block(seen, n_seen), "clean": block(kept, n - n_seen),
            },
            "contrasts": {"seen_minus_clean": {scoring: {"metric": metric, "diff": seen - kept, "se": 0.02}}},
        }
        suffix = "_constrained_decontam_summary.json" if constrained else "_decontam_summary.json"
        (bench_dir / f"{task}_{label}{suffix}").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_faitheval_mc(bench_dir: Path, accuracy: float, accuracy_norm: float | None = None,
                       n: int = 100) -> None:
    """``counterfactual_mc_summary.json`` (``scoring: choice_loglik``)."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    norm = accuracy if accuracy_norm is None else accuracy_norm
    summary = {
        "task": "counterfactual_mc", "model_id": "synthetic", "scoring": "choice_loglik",
        "num_examples": n, "num_correct": round(accuracy * n), "accuracy": accuracy,
        "num_correct_norm": round(norm * n), "accuracy_norm": norm,
    }
    (bench_dir / "counterfactual_mc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def write_halueval_variant(
    bench_dir: Path, per_task_accuracy: Mapping[str, float], variant: str,
    label: str = "synthetic", n: int = 100,
) -> None:
    """``<task>_<label>_<variant>_summary.json`` as ``score_results.py --emit-variant`` writes it."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    for task, acc in per_task_accuracy.items():
        summary = {
            "task": task, "model": label, "variant": variant, "scoring": "generate",
            "source_results": f"{task}_{label}_results.json", "source_sha256": "0" * 64,
            "num_examples": n, "num_scored": n, "accuracy": acc, "format_compliance": 0.9,
            "tpr": acc, "tnr": acc, "judged_yes_rate": 0.5,
        }
        (bench_dir / f"{task}_{label}_{variant}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")


def write_faitheval_rescored(
    bench_dir: Path, per_task_accuracy: Mapping[str, float], rule: str,
    n: int = 100, words: float = 50.0, strata: Mapping[str, float] | None = None,
) -> None:
    """``<task>_summary.json`` as ``FaithEval-reproduce/src/rescore.py --rule <rule>`` writes it."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    for task, acc in per_task_accuracy.items():
        summary = {
            "task": task, "variant": rule, "scoring": "generate",
            "source_predictions": f"{task}_predictions.jsonl", "source_sha256": "0" * 64,
            "num_examples": n, "num_correct": round(acc * n), "accuracy": acc,
            "mean_prediction_words": words,
        }
        if strata is not None:
            summary["length_strata"] = {k: {"n": 10, "accuracy": v, "exact_match": 0.0}
                                        for k, v in strata.items()}
            summary["exact_match"] = 0.01
        (bench_dir / f"{task}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_ragtruth(
    bench_dir: Path,
    overall_rate: float,
    per_task_rate: Mapping[str, float] | None = None,
    gold_f1: Mapping[str, float] | None = None,
    n: int = 100,
) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    per_task = {
        t: {"total": n, "flagged": round(r * n), "hallucination_rate": r}
        for t, r in (per_task_rate or {}).items()
    }
    summary: dict = {
        "detector_model_id": "synthetic-detector",
        "gold_f1_mode": gold_f1 is not None,
        "rate": {
            "total": n,
            "flagged": round(overall_rate * n),
            "hallucination_rate": overall_rate,
            "parse_failures": 0,
            "per_task": per_task,
        },
    }
    if gold_f1 is not None:
        summary["gold_f1"] = {
            "overall": {"total": n, **dict(gold_f1)},
            "per_task": {},
        }
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_truthfulqa(bench_dir: Path, metrics: Mapping[str, float], model: str = "synthetic") -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    cols = list(metrics)
    with (bench_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *cols])
        writer.writerow([model, *[metrics[c] for c in cols]])


def write_harness(
    bench_dir: Path,
    rows: list[Mapping[str, object]],
    provenance: Mapping[str, object] | None = None,
) -> None:
    """rows: partial dicts; missing keys are filled with schema defaults.

    Values are serialised as strings to mirror harness's ``json.dumps(default=str)``.
    """
    bench_dir.mkdir(parents=True, exist_ok=True)
    full_rows = []
    for r in rows:
        row = {
            "task": None, "kind": "task", "alias": None, "metric": "acc",
            "filter": "none", "value": None, "stderr": None,
            "higher_is_better": True, "num_fewshot": 0, "version": 2.0, "n_samples": 100,
        } #NOTE: why is harder is better hardcoded???
        row.update(r)
        # harness writes numbers via default=str -> value/stderr land as strings.
        if row["value"] is not None:
            row["value"] = str(row["value"])
        if row["stderr"] is not None:
            row["stderr"] = str(row["stderr"])
        full_rows.append(row)
    summary = dict(provenance or {"model_args_string": "pretrained=synthetic"})
    summary["results"] = full_rows
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_full_run(
    run_dir: Path,
    *,
    seed: int | None,
    faitheval: Mapping[str, float] | None = None,
    halueval: Mapping[str, float] | None = None,
    ragtruth_rate: float | None = None,
    truthfulqa: Mapping[str, float] | None = None,
    harness_rows: list[Mapping[str, object]] | None = None,
    halueval_constrained: Mapping[str, float] | None = None,
    halueval_decontam: Mapping[str, tuple[float, float]] | None = None,
    faitheval_mc: float | None = None,
) -> Path:
    """Materialise a complete run dir with whichever benchmarks are requested."""
    run_dir = Path(run_dir)
    write_run_metadata(run_dir, seed)
    if faitheval is not None:
        write_faitheval(run_dir / "faitheval", faitheval)
    if faitheval_mc is not None:
        write_faitheval_mc(run_dir / "faitheval", faitheval_mc)
    if halueval is not None:
        write_halueval(run_dir / "halueval", halueval)
    if halueval_constrained is not None:
        write_halueval_constrained(run_dir / "halueval", halueval_constrained)
    if halueval_decontam is not None:
        write_halueval_decontam(run_dir / "halueval", halueval_decontam)
    if ragtruth_rate is not None:
        write_ragtruth(run_dir / "ragtruth", ragtruth_rate,
                       per_task_rate={"QA": ragtruth_rate, "Summary": ragtruth_rate})
    if truthfulqa is not None:
        write_truthfulqa(run_dir / "truthfulqa", truthfulqa)
    if harness_rows is not None:
        write_harness(run_dir / "harness", harness_rows)
    return run_dir
