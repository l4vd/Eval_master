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

**Modified protocols** are reported under their own benchmark names —
``halueval.constrained``, ``halueval.decontam``, ``halueval.constrained_decontam``,
``faitheval.mc`` — so they can never be averaged into, compared with, or plotted as the
original, upstream-comparable numbers (:func:`analysis.model.protocol_of`). The two
readers below route each summary by the ``scoring`` / ``variant`` field its writer
records; a summary without either is an original one and parses exactly as before.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Callable

from analysis.model import MetricRecord, base_benchmark, benchmark_selected

# --- Direction of metrics that don't self-report higher/lower-is-better ----------
# (harness rows carry their own ``higher_is_better`` and bypass this table.)
_HIGHER_IS_BETTER: dict[tuple[str, str], bool] = {
    ("faitheval", "accuracy"): True,
    ("halueval", "accuracy"): True,
    ("halueval", "format_compliance"): True,
    ("halueval", "tpr"): True,
    ("halueval", "tnr"): True,
    # judged_yes_rate is diagnostic, not directional: 0.0 and 1.0 are both degenerate
    # and ~0.5 is healthy. It gets the default direction only because the table needs
    # one; never rank on it.
    ("halueval", "judged_yes_rate"): True,
    ("ragtruth", "hallucination_rate"): False,   # the one inverted metric
    ("ragtruth", "gold_precision"): True,
    ("ragtruth", "gold_recall"): True,
    ("ragtruth", "gold_f1"): True,
    # Modified protocols. verdict_mass / frac_mass_below_half are diagnostics: how much
    # probability the judge puts on a verdict token at all, not how well it judges.
    ("halueval.constrained", "auroc"): True,
    ("halueval.constrained", "frac_mass_below_half"): False,
    ("halueval.constrained_decontam", "auroc"): True,
    ("halueval.constrained_decontam", "frac_mass_below_half"): False,
    ("halueval.decontam", "accuracy"): True,
    ("faitheval.mc", "accuracy"): True,
    ("faitheval.mc", "accuracy_norm"): True,
}


def _direction(benchmark: str, metric: str, default: bool = True) -> bool:
    return _HIGHER_IS_BETTER.get((benchmark, metric), default)


# --- Primary-metric tagging (the headline metric per benchmark) ------------------
# Overridable at runtime via a YAML map (see AnalysisConfig / --primary-map).
# A record is primary if is_primary(benchmark, task, metric) is True.
_PER_TASK_PRIMARY = {
    "faitheval": "accuracy",
    "halueval": "accuracy",
    # Threshold-free, so a constant or badly calibrated judge cannot score by base rate.
    "halueval.constrained": "auroc",
    "halueval.constrained_decontam": "auroc",
    "halueval.decontam": "accuracy",
    "faitheval.mc": "accuracy",
}


def _default_is_primary(benchmark: str, task: str, metric: str) -> bool:
    if benchmark in _PER_TASK_PRIMARY:
        # Per-task headline metric; the synthesized task="mean" is a convenience record,
        # not primary (avoids double-counting in ranked deltas).
        return metric == _PER_TASK_PRIMARY[benchmark] and task != "mean"
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

def _faitheval_benchmark(data: dict) -> str:
    """``faitheval.mc`` for a multiple-choice log-likelihood summary, else the original."""
    return "faitheval.mc" if data.get("scoring") == "choice_loglik" else "faitheval"


def parse_faitheval(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """One flat ``<task>_summary.json`` per task; key ``accuracy`` (higher better).

    ``counterfactual_mc`` (``scoring: choice_loglik``) goes to ``faitheval.mc`` with its
    ``accuracy_norm`` alongside, and never into the original task mean.
    """
    records: list[MetricRecord] = []
    accs: dict[str, list[float]] = {}
    for path in sorted(bench_dir.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        benchmark = _faitheval_benchmark(data)
        task = data.get("task") or path.stem.replace("_summary", "")
        acc = _coerce_float(data.get("accuracy"))
        if acc is None:
            continue
        accs.setdefault(benchmark, []).append(acc)
        records.append(
            _mk(arm, seed, benchmark, task, "accuracy", acc,
                None, data.get("num_examples"), is_primary)
        )
        if benchmark == "faitheval.mc":
            norm = _coerce_float(data.get("accuracy_norm"))
            if norm is not None:
                records.append(
                    _mk(arm, seed, benchmark, task, "accuracy_norm", norm,
                        None, data.get("num_examples"), is_primary)
                )
    for benchmark, values in accs.items():
        _append_task_mean(records, arm, seed, benchmark, "accuracy", values, is_primary)
    return records


# Diagnostics HaluEval writes next to its accuracy. None is primary — they explain a
# headline number rather than competing with it — but they must reach the analysis
# layer, because `accuracy` alone cannot tell "judged wrongly" from "never emitted a
# verdict" from "answered one constant label". Absent on runs predating them.
_HALUEVAL_DIAGNOSTICS = ("format_compliance", "tpr", "tnr", "judged_yes_rate")
# The constrained (prefill-only) scorer's diagnostics, next to its AUROC.
_CONSTRAINED_DIAGNOSTICS = ("accuracy_argmax", "tpr", "tnr", "judged_yes_rate",
                            "mean_verdict_mass", "frac_mass_below_half")
# Row sets a decontaminated summary scores side by side (score_results.decontam_summary).
_DECONTAM_ROW_SETS = ("original", "seen", "clean", "unseen_exposed")


def _halueval_benchmark(data: dict) -> str:
    constrained = data.get("scoring") == "constrained"
    if data.get("variant") == "decontam":
        return "halueval.constrained_decontam" if constrained else "halueval.decontam"
    return "halueval.constrained" if constrained else "halueval"


def parse_halueval(
    bench_dir: Path, arm: str, seed: int | None, is_primary: PrimaryPredicate
) -> list[MetricRecord]:
    """One flat ``<task>_<label>_summary.json`` per task; key ``accuracy`` (+ diagnostics).

    Variant summaries (``*_constrained_summary.json``, ``*_decontam_summary.json``) carry
    ``scoring`` / ``variant`` and are routed to their own benchmark names.
    """
    records: list[MetricRecord] = []
    headline: dict[str, tuple[str, list[float]]] = {}
    for path in sorted(bench_dir.glob("*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        benchmark = _halueval_benchmark(data)
        constrained = data.get("scoring") == "constrained"
        task = data.get("task") or path.stem.replace("_summary", "")
        metric = "auroc" if constrained else "accuracy"
        value = _coerce_float(data.get(metric))
        if value is None:
            continue
        n = data.get("num_examples")
        headline.setdefault(benchmark, (metric, []))[1].append(value)
        stderr = _coerce_float(data.get("auroc_se")) if constrained else None
        records.append(_mk(arm, seed, benchmark, task, metric, value, stderr, n, is_primary))
        for diag in (_CONSTRAINED_DIAGNOSTICS if constrained else _HALUEVAL_DIAGNOSTICS):
            value = _coerce_float(data.get(diag))
            if value is not None:
                records.append(_mk(arm, seed, benchmark, task, diag, value, None, n, is_primary))
        if data.get("variant") == "decontam":
            records.extend(_decontam_records(data, arm, seed, benchmark, task, is_primary))
    for benchmark, (metric, values) in headline.items():
        _append_task_mean(records, arm, seed, benchmark, metric, values, is_primary)
    return records


def _decontam_records(data, arm, seed, benchmark, task, is_primary) -> list[MetricRecord]:
    """Row-set scores (``accuracy_seen`` ...) and contrasts (``seen_minus_clean``).

    A contrast carries its own SE in ``stderr``; row-set accuracies leave it None so a
    reader derives the binomial SE from ``n_samples``, as for any accuracy.
    """
    constrained = data.get("scoring") == "constrained"
    scoring, metric = ("constrained", "auroc") if constrained else ("strict", "accuracy")
    out: list[MetricRecord] = []
    lenient = _coerce_float(data.get("accuracy_lenient"))
    if lenient is not None:
        out.append(_mk(arm, seed, benchmark, task, "accuracy_lenient", lenient,
                       None, data.get("num_examples"), is_primary))
    row_sets = data.get("row_sets") or {}
    for name in _DECONTAM_ROW_SETS:
        block = row_sets.get(name) or {}
        scores = block.get(scoring) or {}
        value = _coerce_float(scores.get(metric))
        if value is None:
            continue
        stderr = _coerce_float(scores.get("auroc_se")) if constrained else None
        out.append(_mk(arm, seed, benchmark, task, f"{metric}_{name}", value,
                       stderr, block.get("num_examples"), is_primary))
    for name, block in (data.get("contrasts") or {}).items():
        contrast = (block or {}).get(scoring) or {}
        value = _coerce_float(contrast.get("diff"))
        if value is not None:
            out.append(_mk(arm, seed, benchmark, task, name, value,
                           _coerce_float(contrast.get("se")), None, is_primary))
    return out


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
    partial run still aggregates. A base name in ``benchmarks`` (``halueval``) selects
    the benchmark and all its variants; a dotted one (``halueval.constrained``) selects
    only that variant — both read the same ``halueval/`` folder.
    """
    run_dir = Path(run_dir)
    names = benchmarks if benchmarks is not None else list(PARSERS)
    records: list[MetricRecord] = []
    for name in dict.fromkeys(base_benchmark(n) for n in names):
        parser = PARSERS.get(name)
        if parser is None:
            warnings.warn(f"No parser registered for benchmark '{name}'; skipping.", stacklevel=2)
            continue
        bench_dir = run_dir / name
        if not bench_dir.is_dir():
            continue
        try:
            parsed = parser(bench_dir, arm, seed, is_primary)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            warnings.warn(f"Failed to parse {name} in {run_dir}: {exc}", stacklevel=2)
            continue
        if benchmarks is not None:
            parsed = [r for r in parsed if benchmark_selected(r.benchmark, benchmarks)]
        records.extend(parsed)
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
