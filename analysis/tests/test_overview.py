"""The descriptive all-variants overview: arm order, ranks, Δ vs base, and its outputs."""

from __future__ import annotations

import pytest

from analysis import fixtures, overview
from analysis.aggregate import aggregate_all
from analysis.model import MetricRecord, RecordSet


def _rec(arm, seed, benchmark, task, metric, value, hib=True):
    return MetricRecord(arm, seed, benchmark, task, metric, value, None, hib, 100, True)


def _records():
    """base + three arms; MC2 orders them b > c > a, halueval.parsed reverses a and b."""
    mc2 = {"base": 0.30, "a": 0.40, "b": 0.50, "c": 0.45}
    parsed = {"base": 0.50, "a": 0.70, "b": 0.55, "c": 0.60}
    rate = {"base": 0.50, "a": 0.30, "b": 0.60, "c": 0.40}  # lower is better
    out = []
    for arm in mc2:
        for k, seed in enumerate((1, 2, 3)):
            jitter = 0.01 * (k - 1)
            out.append(_rec(arm, seed, "harness", "truthfulqa_mc2", "acc", mc2[arm] + jitter))
            out.append(_rec(arm, seed, "halueval.parsed", "qa", "accuracy", parsed[arm] + jitter))
            out.append(_rec(arm, seed, "halueval", "qa", "accuracy", parsed[arm] - 0.1 + jitter))
            out.append(_rec(arm, seed, "ragtruth", "overall", "hallucination_rate", rate[arm] + jitter, hib=False))
    return RecordSet(out)


def test_arm_order_follows_mc2_and_leaves_out_the_base():
    aggs = aggregate_all(_records())
    assert overview.arm_order(aggs, "base") == ["b", "c", "a"]


def test_arm_order_is_identical_across_every_panel(tmp_path):
    pytest.importorskip("matplotlib")
    seen = []
    original = overview._panel

    def spy(ax, aggs, arms, colors, base, panel):
        seen.append((tuple(arms), tuple(colors[a] for a in arms)))
        return original(ax, aggs, arms, colors, base, panel)

    overview._panel = spy
    try:
        overview.write_overview(_records(), tmp_path, reference="base")
    finally:
        overview._panel = original
    assert len(seen) >= 3 and len(set(seen)) == 1


def test_rank_grid_on_constructed_data():
    aggs = aggregate_all(_records())
    arms = overview.arm_order(aggs, "base")
    assert overview.ranks(aggs, arms, ("harness", "truthfulqa_mc2", "acc")) == {"b": 1, "c": 2, "a": 3}
    assert overview.ranks(aggs, arms, ("halueval.parsed", "qa", "accuracy")) == {"a": 1, "c": 2, "b": 3}
    # lower-is-better metrics rank the smallest value first
    assert overview.ranks(aggs, arms, ("ragtruth", "overall", "hallucination_rate")) == {"a": 1, "c": 2, "b": 3}


def test_ties_share_the_better_rank():
    rs = RecordSet([_rec(a, 1, "x", "t", "m", v) for a, v in (("p", 0.5), ("q", 0.5), ("r", 0.1))])
    assert overview.ranks(aggregate_all(rs), ["p", "q", "r"], ("x", "t", "m")) == {"p": 1, "q": 1, "r": 3}


def test_delta_vs_base_sign_follows_the_metric_direction():
    aggs = aggregate_all(_records())
    arms = overview.arm_order(aggs, "base")
    parsed = overview.deltas_vs_base(aggs, arms, "base", ("halueval.parsed", "qa", "accuracy"))
    assert all(d > 0 for d in parsed.values())  # every arm above base
    rate = overview.deltas_vs_base(aggs, arms, "base", ("ragtruth", "overall", "hallucination_rate"))
    assert rate["a"] > 0 and rate["b"] < 0  # a lowered the rate (better), b raised it
    sd = overview.pooled_sd(aggs, arms, ("halueval.parsed", "qa", "accuracy"))
    assert parsed["a"] == pytest.approx((0.70 - 0.50) / sd)
    assert overview.deltas_vs_base(aggs, arms, None, ("halueval.parsed", "qa", "accuracy")) == {}


def test_tables_are_written_without_matplotlib_and_never_a_comparisons_json(tmp_path, monkeypatch):
    def no_mpl():
        raise ImportError("no matplotlib")

    monkeypatch.setattr(overview, "_mpl", no_mpl)
    paths = overview.write_overview(_records(), tmp_path, reference="base")
    assert sorted(p.name for p in paths) == ["overview.md", "overview.tsv"]
    tsv = (tmp_path / "overview.tsv").read_text(encoding="utf-8")
    assert tsv.startswith("# overview") and "descriptive, not pre-registered" in tsv.splitlines()[0]
    assert "halueval.parsed\tqa\taccuracy\ta\t\t0.7000" in tsv
    assert not list(tmp_path.rglob("comparisons.json"))


def test_figures_with_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    records = list(_records())
    for arm, words in (("base", 10.0), ("a", 80.0), ("b", 40.0), ("c", 60.0)):
        for seed in (1, 2, 3):
            records.append(_rec(arm, seed, "faitheval", "unanswerable", "accuracy", 0.3))
            records.append(_rec(arm, seed, "faitheval.strict", "unanswerable", "accuracy", 0.1))
            records.append(_rec(arm, seed, "faitheval.strict", "unanswerable", "mean_prediction_words", words))
            records.append(_rec(arm, seed, "faitheval.contains", "counterfactual", "accuracy_len_1_5", 0.5))
            records.append(_rec(arm, seed, "faitheval.contains", "counterfactual", "accuracy", 0.25))
    names = {p.name for p in overview.write_overview(RecordSet(records), tmp_path, reference="base")}
    assert {"halueval_qa.png", "faitheval_unanswerable.png", "faitheval_counterfactual.png", "harness.png",
            "rank_grid.png", "delta_vs_base.png", "faitheval_length.png"} <= names
    assert not list(tmp_path.rglob("comparisons.json"))


def test_replot_from_an_analysis_dir(tmp_path):
    from analysis.cli import main

    for arm, acc in (("base", 0.3), ("cur", 0.5)):
        for seed in (1, 2):
            run = fixtures.write_full_run(tmp_path / arm / f"seed_{seed}", seed=seed, halueval={"qa": acc},
                                          harness_rows=[{"task": "truthfulqa_mc2", "value": acc}])
            fixtures.write_halueval_variant(run / "halueval.parsed", {"qa": acc + 0.1}, "parsed")
    out = tmp_path / "out"
    assert main(["--arm", f"base={tmp_path / 'base'}", "--arm", f"cur={tmp_path / 'cur'}", "--no-plot",
                 "--variants-overview", "--out", str(out)]) == 0
    first = (out / "overview" / "overview.tsv").read_text(encoding="utf-8")
    assert "halueval.parsed" in first
    assert overview.main(["--from", str(out), "--out", str(tmp_path / "again"), "--no-plot"]) == 0
    assert (tmp_path / "again" / "overview.tsv").read_text(encoding="utf-8") == first
