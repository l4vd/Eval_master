"""JSON / JSONL / LaTeX writers for the analysis outputs.

Mirrors the training-side conventions: JSON is ``indent=2, ensure_ascii=False`` (as in
``hallu_mitigate.utils.io``) and the LaTeX table follows ``ensemble.aggregation.to_latex_table``
(booktabs ``tabular``) so the appendix tables look consistent across the two sides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from analysis.aggregate import ArmAggregate, MetricAggregate
from analysis.model import RecordSet


# --- low-level I/O (parent dirs auto-created) ------------------------------------

def write_json(obj, path: str | Path, indent: int = 2) -> Path:
    p = _as_path(path)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding="utf-8")
    return p


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    p = _as_path(path)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- records (the long-form dump; reload point for standalone plotting) ----------

def write_records(records: RecordSet, path: str | Path) -> Path:
    return write_jsonl(records.to_jsonl_records(), path)


def load_records(path: str | Path) -> RecordSet:
    return RecordSet.from_jsonl_records(load_jsonl(path))


# --- aggregate outputs -----------------------------------------------------------

def write_aggregate(
    arms: dict[str, ArmAggregate], outdir: str | Path, *, primary_only: bool = True
) -> dict[str, Path]:
    """Write ``aggregate.json`` (all arms, all metrics) + ``aggregate.tex`` (appendix)."""
    outdir = Path(outdir)
    payload = {arm: agg.to_dict() for arm, agg in arms.items()}
    json_path = write_json(payload, outdir / "aggregate.json")
    tex_path = _as_path(outdir / "aggregate.tex")
    tex_path.write_text(aggregate_to_latex(arms, primary_only=primary_only), encoding="utf-8")
    return {"json": json_path, "tex": tex_path}


def aggregate_to_latex(arms: dict[str, ArmAggregate], *, primary_only: bool = True) -> str:
    """One row per (arm, benchmark, task, metric): mean +/- std and 95% CI."""
    lines = [
        r"\begin{tabular}{llllrl}",
        r"\toprule",
        r"Arm & Benchmark & Task & Metric & Seeds & Mean $\pm$ Std (95\% CI) \\",
        r"\midrule",
    ]
    rows: list[tuple] = []
    for arm, agg in arms.items():
        metrics = agg.primary() if primary_only else list(agg.metrics.values())
        for m in metrics:
            rows.append((arm, m))
    rows.sort(key=lambda t: (t[1].benchmark, t[1].task, t[1].metric, t[0]))
    for arm, m in rows:
        arrow = "" if m.higher_is_better else r"$\downarrow$"
        lines.append(
            f"{_esc(arm)} & {_esc(m.benchmark)} & {_esc(m.task)} & "
            f"{_esc(m.metric)}{arrow} & {m.n_seeds} & "
            f"{m.mean:.3f} $\\pm$ {m.std:.3f} "
            f"[{m.ci_95_lower:.3f}, {m.ci_95_upper:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# --- comparison outputs ----------------------------------------------------------

def write_comparisons(comparisons: list, outdir: str | Path) -> dict[str, Path]:
    """Write ``comparisons.json`` + ``comparisons.tex`` for the arm-vs-arm results."""
    outdir = Path(outdir)
    rows = [c.to_dict() for c in comparisons]
    json_path = write_json(rows, outdir / "comparisons.json")
    tex_path = _as_path(outdir / "comparisons.tex")
    tex_path.write_text(comparisons_to_latex(comparisons), encoding="utf-8")
    return {"json": json_path, "tex": tex_path}


def comparisons_to_latex(comparisons: list) -> str:
    """One row per comparison: signed delta vs. reference + test/CI per regime."""
    from analysis.stats import significance_stars

    lines = [
        r"\begin{tabular}{lllrll}",
        r"\toprule",
        r"Arm (vs.\ ref) & Benchmark & Task/Metric & $\Delta$ & Regime & Test/CI \\",
        r"\midrule",
    ]
    for c in sorted(comparisons, key=lambda c: (c.benchmark, c.task, c.metric, c.arm)):
        tm = f"{c.task}/{c.metric}"
        if c.regime == "paired" and c.paired is not None:
            p = c.paired.get("p_value")
            stars = significance_stars(p) if p is not None else ""
            detail = f"Wilcoxon $p={_fmt(p)}$ {stars}".strip()
        elif c.one_sample is not None:
            lo = c.one_sample.get("ci_95_lower")
            hi = c.one_sample.get("ci_95_upper")
            detail = f"CI $[{_fmt(lo)}, {_fmt(hi)}]$"
        else:
            detail = ""
        lines.append(
            f"{_esc(c.arm)} vs {_esc(c.reference)} & {_esc(c.benchmark)} & "
            f"{_esc(tm)} & {c.signed_delta:+.3f} & {_esc(c.regime)} & {detail} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "--"


def _esc(text: str) -> str:
    """Escape the LaTeX specials that appear in benchmark/task/metric labels."""
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _as_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
