"""External, decoupled plotting module (matplotlib).

Consumes only the intermediate data model — a :class:`~analysis.model.RecordSet`, the
:class:`~analysis.aggregate.ArmAggregate` map, and :class:`~analysis.compare.Comparison`
list — never raw summaries. ``matplotlib`` is imported lazily so the rest of the package
stays importable without the ``plot`` extra.

Runs fully standalone: :func:`main` reloads persisted ``records.jsonl`` /
``comparisons.json`` from an analysis dir and re-plots with no eval attached, so figures
can be restyled long after the evals finished. Per-benchmark disable (``--benchmarks`` /
``--exclude``) drops any framework from every figure.

Plot altitudes
--------------
1. :func:`plot_benchmark`        one benchmark, arms side by side (primary metrics), CI error bars + seed points
2. :func:`plot_tasks`            per-task grouped bars within a benchmark
3. :func:`plot_cross_benchmark_panels`  small multiples, one panel per benchmark (native scales)
4. :func:`plot_ranked_deltas`    direction-normalized delta of each arm vs. a reference, sorted
5. :func:`plot_paired_deltas`    per-seed deltas + mean + CI + Wilcoxon significance (paired regime only)

Colour: the Okabe-Ito colourblind-safe qualitative palette, assigned to arms in a fixed
order (colour follows the arm identity, never its rank, never cycled).
"""

from __future__ import annotations

import warnings
from pathlib import Path

from analysis.aggregate import ArmAggregate, aggregate_all
from analysis.model import RecordSet, signed_value

# Okabe-Ito: fixed categorical order. Black kept last (reads as text otherwise).
_OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
# Diverging pair for signed deltas: improvement (green) vs. regression (vermillion).
_POS_COLOR = "#009E73"
_NEG_COLOR = "#D55E00"
_ZERO_COLOR = "#888888"


def arm_colors(arms: list[str]) -> dict[str, str]:
    """Map arms to fixed palette slots. >8 arms folds extras to grey (fold, don't cycle)."""
    colors: dict[str, str] = {}
    for i, arm in enumerate(arms):
        if i < len(_OKABE_ITO):
            colors[arm] = _OKABE_ITO[i]
        else:
            warnings.warn(f"More than {len(_OKABE_ITO)} arms; '{arm}' folded to grey.")
            colors[arm] = "#BBBBBB"
    return colors


def _mpl():
    """Lazy matplotlib import with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _direction_note(higher_is_better: bool) -> str:
    return "higher is better" if higher_is_better else "lower is better"


# =================================================================================
# 1 + 2: within-benchmark
# =================================================================================

def plot_benchmark(
    records: RecordSet, benchmark: str, outdir: Path, *,
    arm_order: list[str] | None = None, primary_only: bool = True,
) -> Path | None:
    """One benchmark: for each primary (task, metric), arms side by side with CI + seeds."""
    plt = _mpl()
    rs = records.filter(benchmark=benchmark)
    if primary_only:
        rs = rs.filter(primary=True)
    if not rs:
        return None
    arms = arm_order or rs.arms()
    colors = arm_colors(arms)
    aggs = {a: aggregate_all(rs).get(a) for a in arms}

    keys = rs.keys()  # (benchmark, task, metric)
    n = len(keys)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * n + 1.5), 4.2))
    group_w = 0.8
    bar_w = group_w / max(len(arms), 1)

    for gi, key in enumerate(keys):
        _, task, metric = key
        for ai, arm in enumerate(arms):
            agg = aggs.get(arm)
            m = agg.get(key) if agg else None
            if m is None:
                continue
            x = gi + (ai - (len(arms) - 1) / 2) * bar_w
            yerr = _ci_err(m)
            ax.bar(x, m.mean, width=bar_w * 0.92, color=colors[arm], zorder=2,
                   label=arm if gi == 0 else None,
                   yerr=yerr, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#333333"})
            # individual seed points (spread), skip the fixed-point None seed
            ys = [v for s, v in m.per_seed.items() if s is not None]
            if len(ys) > 1:
                ax.scatter([x] * len(ys), ys, s=14, color="#222222",
                           alpha=0.55, zorder=3, linewidths=0)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{t}\n{me}" for _, t, me in keys], fontsize=8)
    ax.set_ylabel("value")
    hib = next(iter(rs)).higher_is_better
    ax.set_title(f"{benchmark} — arms by primary metric ({_direction_note(hib)})", fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=min(len(arms), 4))
    return _save(fig, outdir / f"{benchmark}.png")


def plot_tasks(
    records: RecordSet, benchmark: str, outdir: Path, *,
    metric: str | None = None, arm_order: list[str] | None = None,
) -> Path | None:
    """Per-task grouped bars within a benchmark for a single metric (default: its primary)."""
    plt = _mpl()
    rs = records.filter(benchmark=benchmark)
    if metric is None:
        prim = rs.filter(primary=True)
        if not prim:
            return None
        metric = next(iter(prim)).metric
    rs = rs.filter(metric=metric)
    if not rs:
        return None
    arms = arm_order or rs.arms()
    colors = arm_colors(arms)
    aggs = aggregate_all(rs)
    tasks = sorted({r.task for r in rs})

    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(tasks) + 1.5), 4.2))
    bar_w = 0.8 / max(len(arms), 1)
    for ti, task in enumerate(tasks):
        for ai, arm in enumerate(arms):
            m = aggs[arm].get((benchmark, task, metric)) if arm in aggs else None
            if m is None:
                continue
            x = ti + (ai - (len(arms) - 1) / 2) * bar_w
            ax.bar(x, m.mean, width=bar_w * 0.92, color=colors[arm], zorder=2,
                   yerr=_ci_err(m), capsize=3,
                   error_kw={"elinewidth": 1, "ecolor": "#333333"},
                   label=arm if ti == 0 else None)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylabel(metric)
    hib = next(iter(rs)).higher_is_better
    ax.set_title(f"{benchmark} — {metric} by task ({_direction_note(hib)})", fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=min(len(arms), 4))
    return _save(fig, outdir / f"{benchmark}_tasks.png")


# =================================================================================
# 3: cross-benchmark small multiples (no cross-scale mixing)
# =================================================================================

def plot_cross_benchmark_panels(
    records: RecordSet, outdir: Path, *, arm_order: list[str] | None = None,
) -> Path | None:
    """One panel per benchmark (native scale), arms side by side on the primary metric mean."""
    plt = _mpl()
    rs = records.filter(primary=True)
    if not rs:
        return None
    arms = arm_order or rs.arms()
    colors = arm_colors(arms)
    benches = rs.benchmarks()
    aggs = aggregate_all(rs)

    ncol = min(3, len(benches))
    nrow = (len(benches) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    for idx, bench in enumerate(benches):
        ax = axes[idx // ncol][idx % ncol]
        # collapse each arm to the mean over this benchmark's primary metrics
        for ai, arm in enumerate(arms):
            ms = [m for k, m in aggs[arm].metrics.items()
                  if k[0] == bench and m.is_primary] if arm in aggs else []
            if not ms:
                continue
            mean = sum(m.mean for m in ms) / len(ms)
            ax.bar(ai, mean, color=colors[arm], zorder=2, width=0.7)
        hib = next((r.higher_is_better for r in rs if r.benchmark == bench), True)
        ax.set_title(f"{bench}\n({_direction_note(hib)})", fontsize=9)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, fontsize=7, rotation=30, ha="right")
        _style_axes(ax)
    for j in range(len(benches), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Cross-benchmark summary — primary metric mean per arm", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, outdir / "cross_benchmark_panels.png", tight=False)


# =================================================================================
# 4: ranked deltas vs. reference (direction-normalized)
# =================================================================================

def plot_ranked_deltas(comparisons: list, outdir: Path) -> Path | None:
    """Signed (improvement-oriented) delta of each arm vs. reference, sorted, per metric.

    Positive = the arm improves on the reference regardless of the metric's native
    direction (hallucination-rate reductions show as positive too).
    """
    plt = _mpl()
    if not comparisons:
        return None
    reference = comparisons[0].reference
    rows = sorted(comparisons, key=lambda c: c.signed_delta)
    labels = [f"{c.arm}: {c.benchmark}/{c.task}/{c.metric}" for c in rows]
    vals = [c.signed_delta for c in rows]
    colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in vals]

    fig, ax = plt.subplots(figsize=(8.0, max(2.5, 0.4 * len(rows) + 1.0)))
    y = range(len(rows))
    ax.barh(list(y), vals, color=colors, zorder=2, height=0.7)
    ax.axvline(0, color=_ZERO_COLOR, linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"signed improvement vs. '{reference}' (higher = better)")
    ax.set_title("Ranked deltas vs. reference (direction-normalized)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return _save(fig, outdir / "ranked_deltas.png")


# =================================================================================
# 5: paired deltas with significance (paired regime only)
# =================================================================================

def plot_paired_deltas(
    comparison, aggregates: dict[str, ArmAggregate], outdir: Path,
) -> Path | None:
    """Per-seed deltas (arm - reference) + mean + CI + Wilcoxon stars. Paired regime only.

    No-ops (returns None with a warning) when the comparison resolved to one-sample —
    a paired-delta plot would misrepresent a comparison against a zero-variance point.
    """
    from analysis.stats import significance_stars

    if comparison.regime != "paired" or comparison.paired is None:
        warnings.warn(
            f"plot_paired_deltas skipped for {comparison.arm} vs {comparison.reference} "
            f"({comparison.benchmark}/{comparison.task}/{comparison.metric}): "
            "not a paired comparison."
        )
        return None
    plt = _mpl()
    key = (comparison.benchmark, comparison.task, comparison.metric)
    arm_m = aggregates[comparison.arm].get(key)
    ref_m = aggregates[comparison.reference].get(key)
    seeds = comparison.paired["seeds"]
    diffs = [arm_m.per_seed[s] - ref_m.per_seed[s] for s in seeds]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(range(len(seeds)), diffs, s=40, color="#0072B2", zorder=3)
    mean_diff = comparison.paired["mean_diff"]
    ax.axhline(mean_diff, color=_POS_COLOR if mean_diff >= 0 else _NEG_COLOR,
               linewidth=1.5, label=f"mean $\\Delta$ = {mean_diff:+.3f}")
    lo, hi = comparison.paired.get("ci_95_lower"), comparison.paired.get("ci_95_upper")
    if lo is not None and hi == hi:  # not NaN
        ax.axhspan(lo, hi, color=_POS_COLOR, alpha=0.12, zorder=1)
    ax.axhline(0, color=_ZERO_COLOR, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=8)
    ax.set_xlabel("seed")
    ax.set_ylabel(f"$\\Delta$ ({comparison.arm} - {comparison.reference})")
    p = comparison.paired.get("p_value")
    stars = significance_stars(p) if p is not None else ""
    ax.set_title(
        f"{comparison.arm} vs {comparison.reference} — "
        f"{comparison.benchmark}/{comparison.task}/{comparison.metric}\n"
        f"Wilcoxon p = {_fmt(p)} {stars}", fontsize=9,
    )
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    safe = f"{comparison.arm}_vs_{comparison.reference}_{comparison.benchmark}_{comparison.task}_{comparison.metric}"
    safe = safe.replace("/", "_").replace(":", "_").replace("::", "_")
    return _save(fig, outdir / f"paired_{safe}.png")


# =================================================================================
# orchestration + standalone entry
# =================================================================================

def plot_all(
    records: RecordSet, outdir: Path, *,
    reference: str | None = None, comparisons: list | None = None,
    arm_order: list[str] | None = None,
) -> list[Path]:
    """Produce every figure the available data supports; returns the saved paths."""
    outdir = Path(outdir)
    arms = arm_order or records.arms()
    paths: list[Path] = []
    for bench in records.benchmarks():
        for fn in (plot_benchmark, plot_tasks):
            p = fn(records, bench, outdir, arm_order=arms)
            if p:
                paths.append(p)
    p = plot_cross_benchmark_panels(records, outdir, arm_order=arms)
    if p:
        paths.append(p)
    if comparisons:
        p = plot_ranked_deltas(comparisons, outdir)
        if p:
            paths.append(p)
        aggregates = aggregate_all(records)
        for c in comparisons:
            if c.regime == "paired":
                pp = plot_paired_deltas(c, aggregates, outdir)
                if pp:
                    paths.append(pp)
    return paths


def _load_comparisons(path: Path) -> list:
    """Rehydrate Comparison objects from a persisted comparisons.json."""
    import json

    from analysis.compare import Comparison

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Comparison(**r) for r in rows]


def main(argv: list[str] | None = None) -> int:
    """Standalone entry: reload persisted artifacts from --from and re-plot."""
    import argparse

    from analysis.report import load_records

    ap = argparse.ArgumentParser(description="Standalone plotting from persisted analysis artifacts.")
    ap.add_argument("--from", dest="src", required=True,
                    help="analysis dir containing records.jsonl (+ optional comparisons.json)")
    ap.add_argument("--out", default=None, help="figure output dir (default: <from>/figures)")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--benchmarks", default=None, help="comma-separated include list")
    ap.add_argument("--exclude", default=None, help="comma-separated exclude list")
    args = ap.parse_args(argv)

    src = Path(args.src)
    outdir = Path(args.out) if args.out else src / "figures"
    records = load_records(src / "records.jsonl")
    include = args.benchmarks.split(",") if args.benchmarks else None
    exclude = args.exclude.split(",") if args.exclude else None
    records = records.include_benchmarks(include, exclude)

    comparisons = None
    comp_path = src / "comparisons.json"
    if comp_path.exists():
        comparisons = _load_comparisons(comp_path)

    paths = plot_all(records, outdir, reference=args.reference, comparisons=comparisons)
    print(f"Wrote {len(paths)} figures to {outdir}")
    return 0


# --- helpers ---------------------------------------------------------------------

def _ci_err(m):
    """Asymmetric yerr [[lo],[hi]] from a MetricAggregate CI; None if degenerate."""
    lo, hi = m.ci_95_lower, m.ci_95_upper
    if lo != lo or hi != hi or (lo == m.mean and hi == m.mean):  # NaN or no spread
        return None
    return [[max(0.0, m.mean - lo)], [max(0.0, hi - m.mean)]]


def _fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "--"


def _save(fig, path: Path, *, tight: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    # also emit a vector PDF for the thesis appendix
    fig.savefig(path.with_suffix(".pdf"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
