"""Plotting: figures created, standalone reload, paired-only guard, benchmark filter."""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("matplotlib")

from analysis.aggregate import aggregate_all
from analysis.compare import compare_all
from analysis.model import RecordSet
from analysis.parse import parse_run_dir
from analysis.report import write_comparisons, write_records
from analysis.spec import AnalysisConfig, ArmSpec, build_records
from analysis import fixtures, plot


def _multi_arm(tmp_path):
    """base (fixed) + two DPO ensembles sharing seeds {42,7}."""
    fixtures.write_full_run(tmp_path / "base" / "point", seed=None,
                            faitheval={"counterfactual": 0.3, "inconsistent": 0.35},
                            ragtruth_rate=0.4, truthfulqa={"MC1": 0.2, "MC2": 0.3})
    for arm, base_acc in (("dpo_base", 0.5), ("dpo_curr", 0.6)):
        for seed in (42, 7):
            fixtures.write_full_run(
                tmp_path / arm / f"seed_{seed}", seed=seed,
                faitheval={"counterfactual": base_acc, "inconsistent": base_acc + 0.02},
                ragtruth_rate=0.3, truthfulqa={"MC1": 0.25, "MC2": 0.35})
    config = AnalysisConfig(
        arms=[ArmSpec("base", str(tmp_path / "base")),
              ArmSpec("dpo_base", str(tmp_path / "dpo_base")),
              ArmSpec("dpo_curr", str(tmp_path / "dpo_curr"))],
        out=str(tmp_path / "out"), reference="base")
    res = build_records(config)
    aggs = aggregate_all(res.records)
    return res, aggs


def test_arm_colors_fixed_order():
    c = plot.arm_colors(["base", "sft", "dpo"])
    assert c["base"] != c["sft"] != c["dpo"]
    # fixed order: same arm list -> same colors regardless of call
    assert plot.arm_colors(["base", "sft", "dpo"]) == c


def test_plot_benchmark_and_tasks(tmp_path):
    res, _ = _multi_arm(tmp_path)
    out = tmp_path / "figs"
    p1 = plot.plot_benchmark(res.records, "faitheval", out)
    p2 = plot.plot_tasks(res.records, "faitheval", out)
    assert p1.is_file() and p2.is_file()
    assert p1.with_suffix(".pdf").is_file()  # vector copy for the appendix


def test_plot_all_and_ranked_deltas(tmp_path):
    res, aggs = _multi_arm(tmp_path)
    comps = compare_all(aggs, res.arm_meta, reference="base")
    paths = plot.plot_all(res.records, tmp_path / "figs", reference="base", comparisons=comps)
    names = {p.name for p in paths}
    assert "cross_benchmark_panels.png" in names
    assert "ranked_deltas.png" in names


def test_paired_deltas_only_for_paired(tmp_path):
    res, aggs = _multi_arm(tmp_path)
    comps = compare_all(aggs, res.arm_meta, reference="base")
    # base is fixed -> every comparison is one-sample -> paired plot must no-op
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = plot.plot_paired_deltas(comps[0], aggs, tmp_path / "figs")
    assert result is None
    assert any("not a paired" in str(w.message) for w in caught)


def test_paired_deltas_renders_for_paired_regime(tmp_path):
    res, aggs = _multi_arm(tmp_path)
    # dpo_base vs dpo_curr share seeds -> paired
    comps = compare_all(aggs, res.arm_meta, reference="dpo_base")
    paired = [c for c in comps if c.arm == "dpo_curr" and c.regime == "paired"]
    assert paired
    p = plot.plot_paired_deltas(paired[0], aggs, tmp_path / "figs")
    assert p.is_file()


def test_standalone_reload(tmp_path):
    res, aggs = _multi_arm(tmp_path)
    comps = compare_all(aggs, res.arm_meta, reference="base")
    analysis_dir = tmp_path / "analysis"
    write_records(res.records, analysis_dir / "records.jsonl")
    write_comparisons(comps, analysis_dir)
    rc = plot.main(["--from", str(analysis_dir), "--out", str(analysis_dir / "figures")])
    assert rc == 0
    assert (analysis_dir / "figures" / "ranked_deltas.png").is_file()


def test_standalone_benchmark_filter(tmp_path):
    res, aggs = _multi_arm(tmp_path)
    analysis_dir = tmp_path / "analysis"
    write_records(res.records, analysis_dir / "records.jsonl")
    plot.main(["--from", str(analysis_dir), "--out", str(analysis_dir / "figs"),
               "--benchmarks", "faitheval"])
    figs = {p.name for p in (analysis_dir / "figs").glob("*.png")}
    assert "faitheval.png" in figs
    assert "ragtruth.png" not in figs      # dropped by filter
