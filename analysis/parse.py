"""Per-benchmark summary readers -> long-form :class:`MetricRecord` rows.

Each of the five benchmarks writes a *different* on-disk shape (verified against the
writer code): FaithEval/HaluEval write one flat ``*_summary.json`` per task, RAGTruth
writes a nested ``summary.json``, TruthfulQA writes a ``summary.csv`` pivot (no JSON),
and harness writes a ``summary.json`` whose ``results`` is a flat list of rows carrying
their own ``higher_is_better`` flag.

The :data:`PARSERS` registry maps a benchmark name to its reader, mirroring the
launcher's ``FOLDERS``/``BUILDERS`` maps. Adding/removing a benchmark is a one-line
registry edit — nothing here hardcodes "five" benchmarks or a fixed task set, and
harness contributes whatever lm_eval task rows happen to be present.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Callable

from analysis.model import MetricRecord

# --- Direction of metrics that don't self-report higher/lower-is-better ----------
# (harness rows carry their own ``higher_is_better`` and bypass this table.)
_HIGHER_IS_BETTER: dict[tuple[str, str], bool] = {
    ("faitheval", "accuracy"): True,
    ("halueval", "accuracy"): True,
    ("ragtruth", "hallucination_rate"): False,   # the one inverted metric
    ("ragtruth", "gold_precision"): True,
    ("ragtruth", "gold_recall"): True,
    ("ragtruth", "gold_f1"): True,
}


def _direction(benchmark: str, metric: str, default: bool = True) -> bool:
    return _HIGHER_IS_BETTER.get((benchmark, metric), default)


# --- Primary-metric tagging (the headline metric per benchmark) ------------------
# Overridable at runtime via a YAML map (see AnalysisConfig / --primary-map).
# A record is primary if is_primary(benchmark, task, metric) is True.
def _default_is_primary(benchmark: str, task: str, metric: str) -> bool:
    if benchmark in ("faitheval", "halueval"):
        # Per-task accuracy is the headline; the synthesized task="mean" is a
        # convenience record, not primary (avoids double-counting in ranked deltas).
        return metric == "accuracy" and task != "mean"
    if benchmark == "ragtruth":
        return task == "overall" and metric == "hallucination_rate"
    if benchmark == "truthfulqa":
        return metric in ("MC1", "MC2")
    if benchmark == "harness":
        return task in ("truthfulqa_mc1", "truthfulqa_mc2") and metric == "acc"
    return False


# The active predicate; cli/spec may replace it with an override-backed one.
PrimaryPredicate = Callable[[str, str, str], bool]


def _coerce_float(value: Any) -> float | None:
    """harness serialises with ``default=str`` so numeric values arrive as strings."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =================================================================================
# Individual parsers. Signature: (bench_dir, arm, seed, is_primary) -> list[record]
# =================================================================================

def parse_faitheval(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """One flat ``<task>_summary.json`` per task; key ``accuracy`` (higher better)."""
    records: list[MetricRecord] = []
    accs: list[float] = []
    for path in sorted(bench_dir.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        task = data.get("task") or path.stem.replace("_summary", "")
        acc = _coerce_float(data.get("accuracy"))
        if acc is None:
            continue
        accs.append(acc)
        records.append(
            _mk(arm, seed, "faitheval", task, "accuracy", acc,
                None, data.get("num_examples"), is_primary)
        )
    _append_task_mean(records, arm, seed, "faitheval", "accuracy", accs, is_primary)
    return records


def parse_halueval(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """One flat ``<task>_<label>_summary.json`` per task; key ``accuracy``."""
    records: list[MetricRecord] = []
    accs: list[float] = []
    for path in sorted(bench_dir.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        task = data.get("task") or path.stem.replace("_summary", "")
        acc = _coerce_float(data.get("accuracy"))
        if acc is None:
            continue
        accs.append(acc)
        records.append(
            _mk(arm, seed, "halueval", task, "accuracy", acc,
                None, data.get("num_examples"), is_primary)
        )
    _append_task_mean(records, arm, seed, "halueval", "accuracy", accs, is_primary)
    return records


def parse_ragtruth(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """Nested ``summary.json``: hallucination_rate (lower better) + optional gold F1."""
    path = bench_dir / "summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[MetricRecord] = []

    rate = data.get("rate") or {}
    overall = _coerce_float(rate.get("hallucination_rate"))
    if overall is not None:
        records.append(
            _mk(arm, seed, "ragtruth", "overall", "hallucination_rate", overall,
                None, rate.get("total"), is_primary)
        )
    for task, block in (rate.get("per_task") or {}).items():
        val = _coerce_float(block.get("hallucination_rate"))
        if val is not None:
            records.append(
                _mk(arm, seed, "ragtruth", task, "hallucination_rate", val,
                    None, block.get("total"), is_primary)
            )

    gold = data.get("gold_f1") or {}
    gold_overall = gold.get("overall") or {}
    for m in ("precision", "recall", "f1"):
        val = _coerce_float(gold_overall.get(m))
        if val is not None:
            records.append(
                _mk(arm, seed, "ragtruth", "overall", f"gold_{m}", val,
                    None, gold_overall.get("total"), is_primary)
            )
    for task, block in (gold.get("per_task") or {}).items():
        for m in ("precision", "recall", "f1"):
            val = _coerce_float(block.get(m))
            if val is not None:
                records.append(
                    _mk(arm, seed, "ragtruth", task, f"gold_{m}", val,
                        None, None, is_primary)
                )
    return records


def parse_truthfulqa(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """``summary.csv`` pivot (rows=Model, cols=metrics). All metrics higher-better."""
    path = bench_dir / "summary.csv"
    if not path.exists():
        return []
    records: list[MetricRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        model_col = reader.fieldnames[0] if reader.fieldnames else "Model"
        for row in reader:
            for col, cell in row.items():
                if col == model_col or cell in (None, ""):
                    continue
                val = _coerce_float(cell)
                if val is None:
                    continue
                # TruthfulQA metrics have no native per-task split -> task="overall".
                records.append(
                    _mk(arm, seed, "truthfulqa", "overall", col, val,
                        None, None, is_primary, direction_default=True)
                )
    return records


def parse_harness(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """``summary.json`` with a flat ``results`` list; each row carries its direction."""
    path = bench_dir / "summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[MetricRecord] = []
    for row in data.get("results") or []:
        value = _coerce_float(row.get("value"))
        if value is None:
            continue  # skip non-numeric rows (e.g. alias strings)
        task = row.get("task")
        metric = row.get("metric")
        if task is None or metric is None:
            continue
        filt = row.get("filter")
        # Disambiguate multiple filters on the same metric without losing the base name.
        if filt not in (None, "", "none"):
            metric = f"{metric}::{filt}"
        hib = row.get("higher_is_better")
        records.append(
            MetricRecord(
                arm=arm, seed=seed, benchmark="harness", task=task, metric=metric,
                value=value, stderr=_coerce_float(row.get("stderr")),
                higher_is_better=bool(hib) if hib is not None else True,
                n_samples=row.get("n_samples"),
                is_primary=is_primary("harness", task, row.get("metric")),
            )
        )
    return records


PARSERS: dict[str, Callable[..., list[MetricRecord]]] = {
    "faitheval": parse_faitheval,
    "truthfulqa": parse_truthfulqa,
    "halueval": parse_halueval,
    "ragtruth": parse_ragtruth,
    "harness": parse_harness,
}


def parse_run_dir(
    run_dir: Path,
    arm: str,
    seed: int | None,
    *,
    is_primary: PrimaryPredicate = _default_is_primary,
    benchmarks: list[str] | None = None,
) -> list[MetricRecord]:
    """Parse every benchmark subfolder present under one run dir.

    Iterates whatever benchmarks are on disk (optionally restricted to
    ``benchmarks``); a missing/unreadable benchmark folder warns and is skipped so a
    partial run still aggregates.
    """
    run_dir = Path(run_dir)
    names = benchmarks if benchmarks is not None else list(PARSERS)
    records: list[MetricRecord] = []
    for name in names:
        parser = PARSERS.get(name)
        if parser is None:
            warnings.warn(f"No parser registered for benchmark '{name}'; skipping.")
            continue
        bench_dir = run_dir / name
        if not bench_dir.is_dir():
            continue
        try:
            records.extend(parser(bench_dir, arm, seed, is_primary))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            warnings.warn(f"Failed to parse {name} in {run_dir}: {exc}")
    return records


# --- helpers ---------------------------------------------------------------------

def _mk(
    arm, seed, benchmark, task, metric, value, stderr, n_samples, is_primary,
    *, direction_default: bool = True,
) -> MetricRecord:
    return MetricRecord(
        arm=arm, seed=seed, benchmark=benchmark, task=task, metric=metric,
        value=float(value), stderr=(None if stderr is None else float(stderr)),
        higher_is_better=_direction(benchmark, metric, direction_default),
        n_samples=n_samples, is_primary=is_primary(benchmark, task, metric),
    )


def _append_task_mean(records, arm, seed, benchmark, metric, values, is_primary) -> None:
    """Emit a convenience task-averaged record (not primary) when >1 task present."""
    if len(values) > 1:
        mean = sum(values) / len(values)
        records.append(
            _mk(arm, seed, benchmark, "mean", metric, mean, None, None, is_primary)
        )
