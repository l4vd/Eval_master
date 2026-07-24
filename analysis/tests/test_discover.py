"""Seed inference + run-dir discovery across the layouts we must accept."""

from __future__ import annotations

from analysis import fixtures
from analysis.discover import discover_arm, expand_spec, infer_seed


def test_infer_seed_from_metadata(tmp_path):
    fixtures.write_full_run(tmp_path / "run", seed=1337, faitheval={"t": 0.5})
    assert infer_seed(tmp_path / "run") == 1337


def test_infer_seed_from_dir_name(tmp_path):
    d = tmp_path / "seed_99"
    fixtures.write_faitheval(d / "faitheval", {"t": 0.5})  # no run_metadata.json
    assert infer_seed(d) == 99


def test_infer_seed_none_when_unknown(tmp_path):
    d = tmp_path / "base_run"
    fixtures.write_faitheval(d / "faitheval", {"t": 0.5})
    assert infer_seed(d) is None  # no hash fallback -> fixed point


def test_expand_training_group_dir(tmp_path):
    group = tmp_path / "2026-07-24" / "12-00-00_dpo_ensemble"
    for s in (42, 7, 99):
        fixtures.write_full_run(group / f"seed_{s}", seed=s, faitheval={"t": 0.5})
    (group / "ensemble").mkdir(parents=True)  # sibling, not a run dir
    (group / "ensemble" / "ensemble_results.json").write_text("{}", encoding="utf-8")

    run_dirs = expand_spec(str(group))
    names = sorted(p.name for p in run_dirs)
    assert names == ["seed_42", "seed_7", "seed_99"]  # ensemble/ excluded


def test_expand_multirun_numeric_subdirs(tmp_path):
    root = tmp_path / "multirun" / "2026-07-24" / "10-00-00"
    for i in range(3):
        fixtures.write_full_run(root / str(i), seed=None, faitheval={"t": 0.5})
    run_dirs = expand_spec(str(root))
    assert sorted(p.name for p in run_dirs) == ["0", "1", "2"]


def test_expand_glob(tmp_path):
    for s in (1, 2):
        fixtures.write_full_run(tmp_path / f"seed_{s}", seed=s, faitheval={"t": 0.5})
    run_dirs = expand_spec(str(tmp_path / "seed_*"))
    assert len(run_dirs) == 2


def test_discover_arm_pairs_dirs_with_seeds(tmp_path):
    group = tmp_path / "grp"
    for s in (42, 7):
        fixtures.write_full_run(group / f"seed_{s}", seed=s, faitheval={"t": 0.5})
    pairs = dict((p.name, seed) for p, seed in discover_arm(str(group)))
    assert pairs == {"seed_42": 42, "seed_7": 7}
