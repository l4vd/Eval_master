"""Regime auto-selection, matched-seed enforcement, and multi-arm comparison."""

from __future__ import annotations

import pytest

from analysis.aggregate import aggregate_all
from analysis.compare import ONE_SAMPLE, PAIRED, compare_all
from analysis.spec import AnalysisConfig, ArmSpec, build_records
from analysis.stats import SeedMismatchError
from analysis import fixtures


def _build(tmp_path, arm_to_seedacc, **cfg):
    """arm -> {seed_or_None: accuracy}; writes runs and builds records + meta."""
    arms = []
    for arm, seed_acc in arm_to_seedacc.items():
        for seed, acc in seed_acc.items():
            name = f"seed_{seed}" if seed is not None else "point"
            fixtures.write_full_run(tmp_path / arm / name, seed=seed,
                                    faitheval={"counterfactual": acc})
        arms.append(ArmSpec(name=arm, spec=str(tmp_path / arm)))
    config = AnalysisConfig(arms=arms, out=str(tmp_path / "out"), **cfg)
    res = build_records(config)
    return res, aggregate_all(res.records)


def test_one_sample_vs_fixed_base(tmp_path):
    res, aggs = _build(tmp_path, {
        "base": {None: 0.30},                 # single point -> fixed
        "dpo": {42: 0.40, 7: 0.50, 99: 0.45},
    })
    assert res.arm_meta["base"].is_fixed_point is True
    comps = compare_all(aggs, res.arm_meta, reference="base")
    dpo = [c for c in comps if c.arm == "dpo"][0]
    assert dpo.regime == ONE_SAMPLE
    assert dpo.paired is None and dpo.one_sample is not None
    assert abs(dpo.reference_point - 0.30) < 1e-9
    assert abs(dpo.delta - (0.45 - 0.30)) < 1e-9


def test_paired_over_matched_seeds(tmp_path):
    res, aggs = _build(tmp_path, {
        "dpo_base": {42: 0.40, 7: 0.50, 99: 0.45},
        "dpo_curr": {42: 0.50, 7: 0.55, 99: 0.60},
    })
    comps = compare_all(aggs, res.arm_meta, reference="dpo_base")
    curr = [c for c in comps if c.arm == "dpo_curr"][0]
    assert curr.regime == PAIRED
    assert curr.paired is not None
    assert curr.paired["n_pairs"] == 3
    assert curr.paired["seeds"] == [7, 42, 99]
    assert curr.signed_delta > 0            # curriculum improves accuracy


def test_paired_seed_mismatch_fails_loudly(tmp_path):
    res, aggs = _build(tmp_path, {
        "a": {42: 0.4, 7: 0.5},
        "b": {42: 0.5, 1337: 0.6},          # 7 vs 1337 mismatch
    })
    with pytest.raises(SeedMismatchError):
        compare_all(aggs, res.arm_meta, reference="a")


def test_paired_mismatch_intersect_when_relaxed(tmp_path):
    res, aggs = _build(tmp_path, {
        "a": {42: 0.4, 7: 0.5},
        "b": {42: 0.5, 1337: 0.6},
    })
    comps = compare_all(aggs, res.arm_meta, reference="a", require_matched=False)
    b = [c for c in comps if c.arm == "b"][0]
    assert b.paired["n_pairs"] == 1 and b.paired["seeds"] == [42]


def test_multi_arm_against_reference(tmp_path):
    res, aggs = _build(tmp_path, {
        "base": {None: 0.30},
        "sft": {42: 0.40, 7: 0.42},
        "dpo": {42: 0.50, 7: 0.55},
    })
    comps = compare_all(aggs, res.arm_meta, reference="base")
    arms = {c.arm for c in comps}
    assert arms == {"sft", "dpo"}
    assert all(c.regime == ONE_SAMPLE for c in comps)  # base fixed -> all one-sample
