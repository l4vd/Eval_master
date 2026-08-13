"""Checkpoint discovery + the eval-production driver (no real models loaded).

The launcher subprocess is faked to write a faitheval fixture into each output dir, so
these tests exercise the *layout* contract end to end: a training ``seed_<SEED>`` ensemble
in, an analysis-ready ``seed_<SEED>`` ensemble out that ``discover_arm`` pairs by seed.
"""

from __future__ import annotations

import subprocess

from analysis import eval_checkpoints, fixtures
from analysis.discover import discover_arm
from analysis.eval_checkpoints import (
    completed_benchmarks,
    discover_checkpoints,
    plan_evaluations,
    run_evaluations,
)


def _write_ckpt_ensemble(group, seeds, *, subdir="final_checkpoint"):
    """Fabricate a training group dir: <group>/seed_<S>/<subdir>/ with a weight file."""
    for s in seeds:
        ck = group / f"seed_{s}" / subdir
        ck.mkdir(parents=True)
        (ck / "config.json").write_text("{}", encoding="utf-8")
    return group


def _fake_launcher(monkeypatch, *, fail_seeds=()):
    """Replace the launcher subprocess with one that writes a faitheval fixture.

    ``fail_seeds`` (matched on the ``seed_<S>`` output dir name) return exit 1 without
    writing, to exercise the continue-on-error path.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        out = next(t.split("=", 1)[1] for t in cmd if t.startswith("output_dir="))
        from pathlib import Path

        out = Path(out)
        if out.name in {f"seed_{s}" for s in fail_seeds}:
            return subprocess.CompletedProcess(cmd, 1)
        fixtures.write_faitheval(out / "faitheval", {"counterfactual": 0.5})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(eval_checkpoints.subprocess, "run", fake_run)
    return calls


# --- discovery ------------------------------------------------------------------

def test_discover_group_dir_finds_final_checkpoints(tmp_path):
    group = _write_ckpt_ensemble(tmp_path / "12-00-00_dpo_ensemble", [42, 7, 99])
    (group / "ensemble").mkdir()  # sibling: no seed, no checkpoint -> excluded
    pairs = discover_checkpoints(str(group))
    assert [s for s, _ in pairs] == [7, 42, 99]  # sorted by seed, ensemble/ dropped
    for seed, ckpt in pairs:
        assert ckpt.name == "final_checkpoint" and ckpt.parent.name == f"seed_{seed}"


def test_discover_glob_of_seed_dirs(tmp_path):
    group = _write_ckpt_ensemble(tmp_path / "grp", [1, 2])
    pairs = discover_checkpoints(str(group / "seed_*"))
    assert {s for s, _ in pairs} == {1, 2}


def test_discover_falls_back_to_seed_dir_when_no_subdir(tmp_path):
    # A layout that saves weights at the seed root (no final_checkpoint/ subdir).
    group = tmp_path / "grp"
    (group / "seed_5").mkdir(parents=True)
    (group / "seed_5" / "model.safetensors").write_text("x", encoding="utf-8")
    (seed, ckpt), = discover_checkpoints(str(group))
    assert seed == 5 and ckpt.name == "seed_5"  # the dir itself is the checkpoint


# --- planning -------------------------------------------------------------------

def test_plan_preserves_seed_in_output_layout(tmp_path):
    group = _write_ckpt_ensemble(tmp_path / "grp", [42, 7])
    out = tmp_path / "eval"
    jobs = plan_evaluations(str(group), out, benchmarks=["faitheval"], num_samples=5)
    by_seed = {j.seed: j for j in jobs}
    assert by_seed[42].out_dir == out / "seed_42"
    cmd = by_seed[42].command
    assert any(t == f"model.id={by_seed[42].checkpoint}" for t in cmd)
    assert f"output_dir={out / 'seed_42'}" in cmd
    assert "run=[faitheval]" in cmd and "num_samples=5" in cmd


# --- execution ------------------------------------------------------------------

def test_dry_run_touches_nothing(tmp_path, monkeypatch):
    calls = _fake_launcher(monkeypatch)
    group = _write_ckpt_ensemble(tmp_path / "grp", [1, 2])
    jobs = plan_evaluations(str(group), tmp_path / "eval")
    run_evaluations(jobs, dry_run=True)
    assert calls == []                         # launcher never invoked
    assert not (tmp_path / "eval").exists()    # no output dirs created


def test_run_produces_analysis_ready_ensemble(tmp_path, monkeypatch):
    """The produced layout round-trips through the real discover_arm, paired by seed."""
    _fake_launcher(monkeypatch)
    group = _write_ckpt_ensemble(tmp_path / "grp", [42, 7, 99])
    out = tmp_path / "eval"
    jobs = plan_evaluations(str(group), out, benchmarks=["faitheval"])
    run_evaluations(jobs)

    # Each seed dir carries the benchmark output AND a seed-tagged run_metadata.json.
    assert (out / "seed_42" / "faitheval").is_dir()
    assert (out / "seed_42" / "run_metadata.json").is_file()

    pairs = dict((p.name, seed) for p, seed in discover_arm(str(out / "seed_*")))
    assert pairs == {"seed_42": 42, "seed_7": 7, "seed_99": 99}


def test_analyze_forwards_benchmarks_and_other_flags(tmp_path, monkeypatch):
    """--analyze must forward --benchmarks (and friends) to analysis.cli, not just the
    eval step -- otherwise an --arm scoped to more benchmarks than were evaluated here
    silently leaks extra benchmarks into the analysis (see eval_checkpoints._run_analysis).
    """
    _fake_launcher(monkeypatch)
    group = _write_ckpt_ensemble(tmp_path / "grp", [42])
    out = tmp_path / "eval"

    captured_argv = {}

    def fake_analysis_main(argv):
        captured_argv["argv"] = argv
        return 0

    monkeypatch.setattr("analysis.cli.main", fake_analysis_main)

    rc = eval_checkpoints.main([
        "--checkpoints", str(group), "--out", str(out),
        "--benchmarks", "faitheval", "--analyze", "--name", "sft",
        "--exclude", "ragtruth", "--rng-seed", "7", "--allow-seed-mismatch",
    ])
    assert rc == 0
    argv = captured_argv["argv"]
    assert "--benchmarks" in argv and argv[argv.index("--benchmarks") + 1] == "faitheval"
    assert "--exclude" in argv and argv[argv.index("--exclude") + 1] == "ragtruth"
    assert "--rng-seed" in argv and argv[argv.index("--rng-seed") + 1] == "7"
    assert "--allow-seed-mismatch" in argv


def test_continue_on_error_keeps_going(tmp_path, monkeypatch):
    _fake_launcher(monkeypatch, fail_seeds=[7])
    group = _write_ckpt_ensemble(tmp_path / "grp", [42, 7])
    out = tmp_path / "eval"
    run_evaluations(plan_evaluations(str(group), out, benchmarks=["faitheval"]))
    # The failing seed still leaves a discoverable, seed-tagged dir; the other completes.
    assert (out / "seed_42" / "faitheval").is_dir()
    assert (out / "seed_7" / "run_metadata.json").is_file()  # stamped before the launcher
    assert not (out / "seed_7" / "faitheval").exists()       # its benchmark never wrote


# --- resume ---------------------------------------------------------------------

def test_completed_benchmarks_needs_a_summary(tmp_path):
    """A summary marks completion; incrementally-written predictions do not."""
    (tmp_path / "faitheval").mkdir()
    (tmp_path / "faitheval" / "counterfactual_summary.json").write_text("{}", encoding="utf-8")
    # halueval got as far as appending per-sample rows but never wrote its summary.
    (tmp_path / "halueval").mkdir()
    (tmp_path / "halueval" / "qa_m_results.json").write_text("{}", encoding="utf-8")

    done = completed_benchmarks(tmp_path, ["faitheval", "halueval", "harness"])
    assert done == ["faitheval"]


def test_resume_skips_only_finished_seeds(tmp_path, monkeypatch):
    calls = _fake_launcher(monkeypatch)
    group = _write_ckpt_ensemble(tmp_path / "grp", [42, 7])
    out = tmp_path / "eval"

    # First pass: seed 7 completes, then the "walltime kill" leaves seed 42 untouched.
    run_evaluations(plan_evaluations(str(group / "seed_7"), out, benchmarks=["faitheval"]))
    assert len(calls) == 1

    # Re-submitting the whole ensemble with --resume must only run the missing seed.
    run_evaluations(
        plan_evaluations(str(group), out, benchmarks=["faitheval"]), resume=True
    )
    assert len(calls) == 2
    ran = [c for c in calls[1] if c.startswith("output_dir=")][0]
    assert ran.endswith("seed_42")
    assert (out / "seed_42" / "faitheval").is_dir()


def test_resume_reruns_a_seed_missing_one_benchmark(tmp_path, monkeypatch):
    """A seed killed between benchmarks is incomplete, so resume must redo it."""
    calls = _fake_launcher(monkeypatch)
    group = _write_ckpt_ensemble(tmp_path / "grp", [42])
    out = tmp_path / "eval"
    # faitheval finished; harness never started.
    fixtures.write_faitheval(out / "seed_42" / "faitheval", {"counterfactual": 0.5})

    run_evaluations(
        plan_evaluations(str(group), out, benchmarks=["faitheval", "harness"]), resume=True
    )
    assert len(calls) == 1  # not skipped
