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


def write_faitheval(bench_dir: Path, per_task_accuracy: Mapping[str, float], n: int = 100) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    for task, acc in per_task_accuracy.items():
        summary = {
            "task": task,
            "model_id": "synthetic",
            "num_examples": n,
            "num_correct": round(acc * n),
            "accuracy": acc,
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
        }
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
) -> Path:
    """Materialise a complete run dir with whichever benchmarks are requested."""
    run_dir = Path(run_dir)
    write_run_metadata(run_dir, seed)
    if faitheval is not None:
        write_faitheval(run_dir / "faitheval", faitheval)
    if halueval is not None:
        write_halueval(run_dir / "halueval", halueval)
    if ragtruth_rate is not None:
        write_ragtruth(run_dir / "ragtruth", ragtruth_rate,
                       per_task_rate={"QA": ragtruth_rate, "Summary": ragtruth_rate})
    if truthfulqa is not None:
        write_truthfulqa(run_dir / "truthfulqa", truthfulqa)
    if harness_rows is not None:
        write_harness(run_dir / "harness", harness_rows)
    return run_dir
