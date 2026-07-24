"""End-to-end CLI: single-ensemble, multi-arm, toggles, primary-map override."""

from __future__ import annotations

import json

from analysis.cli import main
from analysis import fixtures


def _write_arm(tmp_path, arm, seed_to_acc, **extra):
    for seed, acc in seed_to_acc.items():
        name = f"seed_{seed}" if seed is not None else "point"
        fixtures.write_full_run(tmp_path / arm / name, seed=seed,
                                faitheval={"counterfactual": acc}, **extra)


def test_single_ensemble_no_compare(tmp_path):
    _write_arm(tmp_path, "dpo", {42: 0.4, 7: 0.5, 99: 0.45})
    out = tmp_path / "analysis"
    rc = main(["--arm", f"dpo={tmp_path / 'dpo'}", "--no-compare", "--no-plot",
               "--out", str(out)])
    assert rc == 0
    assert (out / "records.jsonl").is_file()
    assert (out / "aggregate.json").is_file()
    assert (out / "aggregate.tex").is_file()
    assert not (out / "comparisons.json").exists()   # --no-compare


def test_multi_arm_compare(tmp_path):
    _write_arm(tmp_path, "base", {None: 0.3})
    _write_arm(tmp_path, "dpo", {42: 0.5, 7: 0.6})
    out = tmp_path / "analysis"
    rc = main(["--arm", f"base={tmp_path / 'base'}",
               "--arm", f"dpo={tmp_path / 'dpo'}",
               "--reference", "base", "--no-plot", "--out", str(out)])
    assert rc == 0
    comps = json.loads((out / "comparisons.json").read_text(encoding="utf-8"))
    assert comps and comps[0]["regime"] == "one_sample"   # base is a fixed point


def test_single_instance_n1(tmp_path):
    # one model, one seed -> flows through the same path, no CI/tests
    _write_arm(tmp_path, "solo", {None: 0.42})
    out = tmp_path / "analysis"
    rc = main(["--arm", f"solo={tmp_path / 'solo'}", "--no-compare", "--no-plot",
               "--out", str(out)])
    assert rc == 0
    agg = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    m = agg["solo"]["metrics"][0]
    assert m["n_seeds"] == 1 and m["std"] == 0.0


def test_primary_map_override(tmp_path):
    _write_arm(tmp_path, "dpo", {42: 0.4}, truthfulqa={"MC1": 0.2, "MC2": 0.3})
    pmap = tmp_path / "primary.json"
    pmap.write_text(json.dumps({"truthfulqa": ["MC1"]}), encoding="utf-8")  # only MC1 primary
    out = tmp_path / "analysis"
    main(["--arm", f"dpo={tmp_path / 'dpo'}", "--no-compare", "--no-plot",
          "--out", str(out), "--primary-map", str(pmap)])
    agg = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    tqa = {(m["task"], m["metric"]): m for m in agg["dpo"]["metrics"]
           if m["benchmark"] == "truthfulqa"}
    assert tqa[("overall", "MC1")]["is_primary"] is True
    assert tqa[("overall", "MC2")]["is_primary"] is False   # overridden out


def test_benchmark_exclude(tmp_path):
    _write_arm(tmp_path, "dpo", {42: 0.4}, ragtruth_rate=0.3)
    out = tmp_path / "analysis"
    main(["--arm", f"dpo={tmp_path / 'dpo'}", "--no-compare", "--no-plot",
          "--out", str(out), "--exclude", "ragtruth"])
    agg = json.loads((out / "aggregate.json").read_text(encoding="utf-8"))
    benches = {m["benchmark"] for m in agg["dpo"]["metrics"]}
    assert "ragtruth" not in benches and "faitheval" in benches
