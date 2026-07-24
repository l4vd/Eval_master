"""Across-seed aggregation: multi-seed stats, single-instance path, round-trip."""

from __future__ import annotations

from analysis.aggregate import aggregate_all, aggregate_arm
from analysis.model import RecordSet
from analysis.parse import parse_run_dir
from analysis import fixtures


def _arm_records(tmp_path, arm, seed_to_acc):
    recs = []
    for seed, acc in seed_to_acc.items():
        run = fixtures.write_full_run(tmp_path / f"{arm}_{seed}", seed=seed,
                                      faitheval={"counterfactual": acc})
        recs += parse_run_dir(run, arm, seed)
    return RecordSet(recs)


def test_aggregate_multi_seed(tmp_path):
    rs = _arm_records(tmp_path, "dpo", {42: 0.3, 7: 0.5, 99: 0.4})
    agg = aggregate_arm(rs, "dpo")
    m = agg.get(("faitheval", "counterfactual", "accuracy"))
    assert m.n_seeds == 3
    assert abs(m.mean - 0.4) < 1e-9
    assert m.std > 0
    assert m.ci_95_lower <= m.mean <= m.ci_95_upper
    assert set(m.per_seed) == {42, 7, 99}
    assert m.is_primary is True


def test_single_instance_no_variance(tmp_path):
    rs = _arm_records(tmp_path, "base", {None: 0.42})
    agg = aggregate_arm(rs, "base")
    m = agg.get(("faitheval", "counterfactual", "accuracy"))
    assert m.n_seeds == 1
    assert m.mean == 0.42
    assert m.std == 0.0            # single instance -> no spread
    assert None in m.per_seed


def test_aggregate_all_arms(tmp_path):
    rs = RecordSet(
        list(_arm_records(tmp_path, "sft", {1: 0.5, 2: 0.6}))
        + list(_arm_records(tmp_path, "dpo", {1: 0.7, 2: 0.8}))
    )
    allagg = aggregate_all(rs)
    assert set(allagg) == {"sft", "dpo"}
    assert allagg["dpo"].get(("faitheval", "counterfactual", "accuracy")).mean == 0.75


def test_to_dict_stringifies_seed_keys(tmp_path):
    rs = _arm_records(tmp_path, "dpo", {42: 0.3})
    d = aggregate_arm(rs, "dpo").to_dict()
    per_seed = d["metrics"][0]["per_seed"]
    assert all(isinstance(k, str) for k in per_seed)
