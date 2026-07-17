"""Normalise lm_eval's nested results into the repo's output convention.

lm_eval returns a deeply nested dict keyed by `"metric,filter"` pairs. We flatten
it into one row per (task, metric, filter) with its stderr attached, and write the
repo-standard `summary.json` (flat rows + provenance) alongside `samples.jsonl` and
the raw lm_eval dump.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Keys inside a task's metric block that are metadata, not metric values.
_META_KEYS = frozenset({"alias", "name", "sample_len", "sample_count"})
# Provenance lm_eval records itself — authoritative, so copy it through verbatim.
_LM_EVAL_PROVENANCE = (
    "lm_eval_version",
    "transformers_version",
    "git_hash",
    "date",
    "chat_template_sha",
)


def flatten_results(res: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per (task, metric, filter), with stderr, version, and n-shot attached.

    Handles both individual tasks (`results`) and any aggregate rows (`groups`),
    though the TruthfulQA tags don't aggregate, so `groups` is typically absent.
    """
    higher = res.get("higher_is_better") or {}
    n_shot = res.get("n-shot") or {}
    versions = res.get("versions") or {}
    n_samples = res.get("n-samples") or {}

    rows: list[dict[str, Any]] = []
    for kind, section in (("task", res.get("results")), ("group", res.get("groups"))):
        for task_name, metrics in (section or {}).items():
            for key, value in metrics.items():
                if key in _META_KEYS:
                    continue
                metric, _, filt = key.partition(",")  # "acc,none" -> ("acc", "none")
                if metric.endswith("_stderr"):
                    continue  # consumed as its base metric's stderr, below
                stderr = metrics.get(f"{metric}_stderr,{filt}")
                rows.append(
                    {
                        "task": task_name,
                        "kind": kind,
                        # Group-member aliases arrive indented (" - truthfulqa_mc1"),
                        # so key on task_name, never alias.
                        "alias": metrics.get("alias"),
                        "metric": metric,
                        "filter": filt,
                        "value": value,
                        # lm_eval writes the literal string "N/A" for an absent stderr.
                        "stderr": None if stderr in (None, "N/A") else stderr,
                        "higher_is_better": (higher.get(task_name) or {}).get(metric),
                        "num_fewshot": n_shot.get(task_name),
                        "version": versions.get(task_name),
                        "n_samples": n_samples.get(task_name),
                    }
                )
    return rows


def build_summary(res: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble `summary.json`: caller provenance + lm_eval provenance + flat rows."""
    summary: dict[str, Any] = dict(provenance)
    for key in _LM_EVAL_PROVENANCE:
        if key in res:
            summary[key] = res[key]
    if res.get("group_subtasks"):
        summary["group_subtasks"] = res["group_subtasks"]
    summary["results"] = flatten_results(res)
    return summary


def write_samples(samples: Mapping[str, Any], path: Path) -> None:
    """Write one JSONL record per evaluated document, tagged with its task name."""
    with path.open("w", encoding="utf-8") as f:
        for task_name, docs in (samples or {}).items():
            for doc in docs:
                if isinstance(doc, dict):
                    record = {"task": task_name, **doc}
                else:
                    record = {"task": task_name, "sample": doc}
                # ensure_ascii=False keeps non-Latin text (31 okapi languages)
                # literal; default=str tolerates lm_eval's numpy scalars.
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_outputs(
    res: Mapping[str, Any],
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    log_samples: bool,
) -> None:
    """Write summary.json, lm_eval_results.json, and (optionally) samples.jsonl."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # default=str: lm_eval results carry numpy scalars, which json can't serialise.
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    raw = {k: v for k, v in res.items() if k != "samples"}
    (output_dir / "lm_eval_results.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8"
    )

    if log_samples:
        write_samples(res.get("samples") or {}, output_dir / "samples.jsonl")
