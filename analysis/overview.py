"""Arm-comparison overview across every variant — descriptive, not pre-registered.

Question answered: for each benchmark task and each variant (original protocol, the
modified protocols, the offline re-scorings), how do the arms compare with each other?
No p-values, no multiplicity correction, no cross-variant statistics. The deciding metric
stays harness MC2 (EXPERIMENT_PROCEDURE §5.8); this is for looking at the data.

Outputs, under ``<out>/overview/`` (never a ``comparisons.json``):

* ``halueval_<task>.png``, ``faitheval_<task>.png``, ``harness.png`` — one panel per
  variant, each with its own y-axis: every arm as mean ± seed-bootstrap 95 % CI, its seeds
  as faint dots, the base (reference) arm as a dashed line, chance where meaningful. Arms
  keep one order (by harness MC2) and one colour in every panel of every figure.
* ``rank_grid.png`` — arms × variants, each cell the arm's rank under that variant.
* ``delta_vs_base.png`` — the same grid, cell = (arm − base) / pooled seed SD, oriented so
  positive is better.
* ``faitheval_length.png`` — arm accuracy against arm answer length per variant, plus the
  ``contains`` length strata.
* ``overview.tsv`` / ``overview.md`` — arm × variant × task: mean, CI, seeds, rank, Δ.

Called from ``analysis.cli --variants-overview``; re-plot without re-parsing with
``python -m analysis.overview --from <analysis_out>``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from analysis.aggregate import ArmAggregate, MetricAggregate, aggregate_all
from analysis.model import RecordSet
from analysis.plot import _mpl, _save, _stem, _style_axes, arm_colors

TITLE = "overview — descriptive, not pre-registered"
MC2 = ("harness", "truthfulqa_mc2", "acc")
MC1 = ("harness", "truthfulqa_mc1", "acc")
LENGTH_STRATA = ("1_5", "6_20", "21_60", "61_plus")
FAITHEVAL_TASKS = ("unanswerable", "inconsistent", "counterfactual")


@dataclass(frozen=True)
class Panel:
    benchmark: str
    task: str
    metric: str
    chance: float | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.benchmark, self.task, self.metric)

    @property
    def label(self) -> str:
        return f"{self.benchmark}\n{self.task} · {self.metric}"


def figure_specs(present: set[tuple[str, str, str]]) -> dict[str, list[Panel]]:
    """Figure name -> panels, keeping only panels with data and figures with any panel."""
    halueval_tasks = sorted({t for b, t, _ in present if b.startswith("halueval") and t != "mean"})
    specs: dict[str, list[Panel]] = {}
    for task in halueval_tasks:
        specs[f"halueval_{task}"] = [
            Panel("halueval", task, "accuracy", 0.5),
            Panel("halueval.parsed", task, "accuracy", 0.5),
            Panel("halueval.lenient", task, "accuracy", 0.5),
            Panel("halueval.constrained", task, "auroc", 0.5),
            Panel("halueval.decontam", task, "accuracy", 0.5),
            Panel("halueval.constrained_decontam", task, "auroc", 0.5),
        ]
    for task in ("unanswerable", "inconsistent"):
        specs[f"faitheval_{task}"] = [Panel(b, task, "accuracy") for b in
                                      ("faitheval", "faitheval.strict", "faitheval.wordmatch")]
    specs["faitheval_counterfactual"] = [
        Panel("faitheval", "counterfactual", "accuracy"),
        Panel("faitheval.contains", "counterfactual", "accuracy"),
        Panel("faitheval.mc", "counterfactual_mc", "accuracy"),
        Panel("faitheval.mc", "counterfactual_mc", "accuracy_norm"),
    ]
    specs["harness"] = [Panel(*MC1), Panel(*MC2)]
    out = {}
    for name, panels in specs.items():
        kept = [p for p in panels if p.key in present]
        if kept:
            out[name] = kept
    return out


# =====================================================================================
# Numbers (no matplotlib)
# =====================================================================================

def _agg(aggs: dict[str, ArmAggregate], arm: str, key) -> MetricAggregate | None:
    a = aggs.get(arm)
    return a.get(key) if a is not None else None


def arm_order(aggs: dict[str, ArmAggregate], base: str | None) -> list[str]:
    """Non-base arms, best harness MC2 first; arms without MC2 last, by name."""
    def sort_key(arm):
        m = _agg(aggs, arm, MC2)
        return (m is None, -(m.signed_mean if m is not None else 0.0), arm)
    return sorted((a for a in aggs if a != base), key=sort_key)


def ranks(aggs, arms: list[str], key) -> dict[str, int]:
    """Rank 1 = best under ``key`` (direction-aware); ties share the better rank."""
    values = {a: m.signed_mean for a in arms if (m := _agg(aggs, a, key)) is not None
              and not math.isnan(m.mean)}
    return {a: 1 + sum(1 for w in values.values() if w > v) for a, v in values.items()}


def pooled_sd(aggs, arms: list[str], key) -> float | None:
    variances = [m.std ** 2 for a in arms if (m := _agg(aggs, a, key)) is not None
                 and m.n_seeds >= 2 and not math.isnan(m.std)]
    if not variances:
        return None
    sd = math.sqrt(sum(variances) / len(variances))
    return sd if sd > 0 else None


def deltas_vs_base(aggs, arms: list[str], base: str | None, key) -> dict[str, float]:
    """(arm − base) / pooled seed SD, signed so that positive means better."""
    b = _agg(aggs, base, key) if base else None
    sd = pooled_sd(aggs, arms, key)
    if b is None or sd is None:
        return {}
    return {a: (m.signed_mean - b.signed_mean) / sd for a in arms if (m := _agg(aggs, a, key)) is not None}


def grid_columns(specs: dict[str, list[Panel]]) -> list[Panel]:
    """MC2 first, then every panel once, in figure order."""
    columns: list[Panel] = []
    seen = set()
    for panels in [[Panel(*MC2)]] + list(specs.values()):
        for p in panels:
            if p.key not in seen:
                seen.add(p.key)
                columns.append(p)
    return columns


def table_rows(aggs, arms: list[str], base: str | None, columns: list[Panel]) -> list[dict]:
    rows = []
    for p in columns:
        rank = ranks(aggs, arms, p.key)
        delta = deltas_vs_base(aggs, arms, base, p.key)
        for arm in ([base] if base in aggs else []) + arms:
            m = _agg(aggs, arm, p.key)
            if m is None:
                continue
            rows.append({
                "benchmark": p.benchmark, "task": p.task, "metric": p.metric, "arm": arm,
                "is_base": arm == base, "mean": m.mean, "ci_95_lower": m.ci_95_lower,
                "ci_95_upper": m.ci_95_upper, "n_seeds": m.n_seeds,
                "rank": rank.get(arm), "delta_vs_base_sd": delta.get(arm),
            })
    return rows


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


_COLUMNS = ("benchmark", "task", "metric", "arm", "is_base", "mean", "ci_95_lower",
            "ci_95_upper", "n_seeds", "rank", "delta_vs_base_sd")


def write_tables(rows: list[dict], outdir: Path, base: str | None) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    tsv = outdir / "overview.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# {TITLE}\n")
        f.write("\t".join(_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(_cell(r[c]) for c in _COLUMNS) + "\n")
    md = outdir / "overview.md"
    lines = [f"# Variant overview ({TITLE})", "",
             "Every arm under every variant. Rank 1 = best among the non-base arms; "
             f"Δ = (arm − base) / pooled seed SD, positive = better; base (reference, the dashed "
             f"line in the figures): `{base or 'none'}`. "
             "The deciding metric stays harness MC2 (EXPERIMENT_PROCEDURE §5.8).", ""]
    current = None
    for r in rows:
        key = (r["benchmark"], r["task"], r["metric"])
        if key != current:
            current = key
            lines += ["", f"## {r['benchmark']} · {r['task']} · {r['metric']}", "",
                      "| arm | mean | 95 % CI | seeds | rank | Δ/SD |", "|---|---|---|---|---|---|"]
        ci = f"[{_cell(r['ci_95_lower'])}, {_cell(r['ci_95_upper'])}]"
        arm = f"{r['arm']} (base)" if r["is_base"] else r["arm"]
        delta = "" if r["delta_vs_base_sd"] is None else f"{r['delta_vs_base_sd']:+.2f}"
        lines.append(f"| {arm} | {_cell(r['mean'])} | {ci} | {r['n_seeds']} | {_cell(r['rank'])} | {delta} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [tsv, md]


# =====================================================================================
# Figures
# =====================================================================================

def _panel(ax, aggs, arms, colors, base, panel: Panel) -> None:
    for i, arm in enumerate(arms):
        m = _agg(aggs, arm, panel.key)
        if m is None:
            continue
        seeds = [v for v in m.per_seed.values() if v == v]
        ax.scatter([i] * len(seeds), seeds, color=colors[arm], alpha=0.25, s=10, zorder=2)
        lo, hi = m.ci_95_lower, m.ci_95_upper
        yerr = None if (lo != lo or hi != hi) else [[max(0.0, m.mean - lo)], [max(0.0, hi - m.mean)]]
        ax.errorbar([i], [m.mean], yerr=yerr, fmt="o", color=colors[arm], capsize=3, zorder=3)
    b = _agg(aggs, base, panel.key) if base else None
    if b is not None:
        ax.axhline(b.mean, color="#444444", linestyle="--", linewidth=1, label=f"{base} (base)")
    if panel.chance is not None:
        ax.axhline(panel.chance, color="#AAAAAA", linestyle=":", linewidth=1, label="chance")
    ax.set_title(panel.label, fontsize=8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=60, ha="right", fontsize=6)
    _style_axes(ax)


def plot_figure(name: str, panels: list[Panel], aggs, arms, colors, base, outdir: Path) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 1, 4.2), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        _panel(ax, aggs, arms, colors, base, panel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(f"{name}: arms under each variant ({TITLE})", fontsize=9)
    return _save(fig, outdir / f"{_stem(name)}.png")


def _grid(values: dict[tuple[str, int], float], arms, columns, *, title, cmap, center, fmt, outdir, name,
          vmax=None) -> Path:
    import numpy as np

    plt = _mpl()
    data = np.full((len(arms), len(columns)), np.nan)
    for (arm, j), v in values.items():
        data[arms.index(arm), j] = v
    fig, ax = plt.subplots(figsize=(0.75 * len(columns) + 3, 0.4 * len(arms) + 2.5))
    kwargs = {}
    if center:
        bound = vmax or (np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0) or 1.0
        kwargs = {"vmin": -bound, "vmax": bound}
    image = ax.imshow(data, cmap=cmap, aspect="auto", **kwargs)
    for i in range(len(arms)):
        for j in range(len(columns)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt(data[i, j]), ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=7)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c.label.replace("\n", " ") for c in columns], rotation=70, ha="right", fontsize=6)
    fig.colorbar(image, ax=ax, shrink=0.7)
    ax.set_title(f"{title}\n({TITLE})", fontsize=9)
    return _save(fig, outdir / name)


def plot_length(aggs, arms, colors, outdir: Path) -> Path | None:
    """Arm accuracy vs arm answer length per FaithEval variant, plus the contains strata."""
    variants = ("faitheval", "faitheval.strict", "faitheval.wordmatch", "faitheval.contains")
    markers = dict(zip(variants, ("o", "s", "^", "D")))
    tasks = [t for t in FAITHEVAL_TASKS
             if any(_agg(aggs, a, (v, t, "mean_prediction_words")) for a in arms for v in variants)]
    strata = [s for s in LENGTH_STRATA
              if any(_agg(aggs, a, ("faitheval.contains", "counterfactual", f"accuracy_len_{s}")) for a in arms)]
    if not tasks and not strata:
        return None
    plt = _mpl()
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 12)
    for k, task in enumerate(tasks):
        ax = fig.add_subplot(grid[0, 4 * k:4 * k + 4])
        for arm in arms:
            words = next((m.mean for v in variants[1:]
                          if (m := _agg(aggs, arm, (v, task, "mean_prediction_words")))), None)
            if words is None:
                continue
            for v in variants:
                m = _agg(aggs, arm, (v, task, "accuracy"))
                if m is not None:
                    ax.scatter(words, m.mean, color=colors[arm], marker=markers[v], s=28)
        for v in variants:
            ax.scatter([], [], color="#666666", marker=markers[v], label=v)
        ax.set_title(f"{task}: accuracy vs mean answer words", fontsize=8)
        ax.set_xlabel("arm mean prediction words", fontsize=7)
        ax.legend(fontsize=6, frameon=False)
        _style_axes(ax)
    for k, stratum in enumerate(strata):
        ax = fig.add_subplot(grid[1, 3 * k:3 * k + 3])
        _panel(ax, aggs, arms, colors, None,
               Panel("faitheval.contains", "counterfactual", f"accuracy_len_{stratum}"))
        ax.set_title(f"contains, answers of {stratum.replace('_', '-')} words", fontsize=8)
    arm_handles = [plt.Line2D([], [], color=colors[a], marker="o", linestyle="", label=a) for a in arms]
    fig.legend(handles=arm_handles, loc="center left", bbox_to_anchor=(1.0, 0.75), fontsize=7, frameon=False)
    fig.suptitle(f"FaithEval: does a variant's arm order follow answer length? ({TITLE})", fontsize=9)
    return _save(fig, outdir / "faitheval_length.png")


def write_overview(records: RecordSet, outdir: Path, *, reference: str | None, rng_seed: int = 0,
                   plot: bool = True) -> list[Path]:
    """Every overview artifact the records support; returns the written paths."""
    outdir = Path(outdir)
    aggs = aggregate_all(records, seed=rng_seed)
    base = reference if reference in aggs else None
    arms = arm_order(aggs, base)
    present = {k for a in aggs.values() for k in a.metrics}
    specs = figure_specs(present)
    columns = grid_columns(specs)
    columns = [c for c in columns if c.key in present]
    paths = write_tables(table_rows(aggs, arms, base, columns), outdir, base)
    if not plot or not arms:
        return paths
    try:
        _mpl()
    except ImportError:
        print("!! overview: matplotlib not installed; tables written, figures skipped")
        return paths
    colors = arm_colors(arms)
    for name, panels in specs.items():
        paths.append(plot_figure(name, panels, aggs, arms, colors, base, outdir))
    rank_values = {(a, j): r for j, c in enumerate(columns) for a, r in ranks(aggs, arms, c.key).items()}
    paths.append(_grid(rank_values, arms, columns, title="Rank of each arm under each variant (1 = best)",
                       cmap="viridis_r", center=False, fmt=lambda v: f"{int(v)}", outdir=outdir,
                       name="rank_grid.png"))
    if base is not None:
        delta_values = {(a, j): d for j, c in enumerate(columns)
                        for a, d in deltas_vs_base(aggs, arms, base, c.key).items()}
        paths.append(_grid(delta_values, arms, columns,
                           title=f"(arm − {base}) / pooled seed SD, positive = better",
                           cmap="RdBu", center=True, fmt=lambda v: f"{v:+.1f}", outdir=outdir,
                           name="delta_vs_base.png"))
    length = plot_length(aggs, arms, colors, outdir)
    if length is not None:
        paths.append(length)
    return paths


def main(argv: list[str] | None = None) -> int:
    from analysis.report import load_records

    ap = argparse.ArgumentParser(description=f"Re-plot the variant overview ({TITLE}).")
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="analysis dir holding records.jsonl and/or modified/records.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default: <from>/overview")
    ap.add_argument("--reference", default=None, help="base arm (default: 'base' if present)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true", help="tables only")
    args = ap.parse_args(argv)

    records = []
    for path in (args.src / "records.jsonl", args.src / "modified" / "records.jsonl"):
        if path.is_file():
            records.extend(load_records(path))
    if not records:
        raise SystemExit(f"no records.jsonl under {args.src}")
    rs = RecordSet(records)
    reference = args.reference or ("base" if "base" in rs.arms() else None)
    paths = write_overview(rs, args.out or args.src / "overview", reference=reference,
                           rng_seed=args.rng_seed, plot=not args.no_plot)
    print(f"Wrote {len(paths)} overview file(s) to {args.out or args.src / 'overview'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
