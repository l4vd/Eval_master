"""The PBS job writers of the optional overview (and the VARIANTS path of qsub_eval_day.sh).

Print mode only writes job scripts, so the planning and the guards run anywhere with bash.
The written jobs are then executed against a stub Eval_master (fake `module`, `python` and
wrapper scripts that only log their arguments), which exercises the job bodies without HPC.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1]


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None or "system32" in bash.lower():  # WSL's bash cannot see Windows paths
        pytest.skip("no POSIX bash")
    probe = subprocess.run([bash, "-c", "declare -A x=([a]=1); echo ${x[a]}"], capture_output=True, text=True)
    if probe.stdout.strip() != "1":
        pytest.skip("bash >= 4 needed")
    return bash


def _p(path: Path) -> str:
    return path.as_posix()


def _run(script: str, *args, env: dict, cwd: Path | None = None) -> subprocess.CompletedProcess:
    full = {**os.environ, **{k: str(v) for k, v in env.items()}}
    return subprocess.run([_bash(), _p(ANALYSIS / script), *map(str, args)], capture_output=True, text=True,
                          env=full, cwd=cwd)


def _day(tmp_path: Path) -> Path:
    day = tmp_path / "train" / "2026-09-10"
    for name in ("08-00-00_dpo_ensemble", "09-00-00_off_sp_dpo_asc_ensemble"):
        (day / name / "seed_42" / "final_checkpoint").mkdir(parents=True)
    (day / "10-00-00_profile").mkdir()  # no seeds: not an ensemble
    return day


def _stub_master(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fake Eval_master and a PATH dir; every tool appends its argv to calls.log."""
    master, bindir, log = tmp_path / "master", tmp_path / "bin", tmp_path / "calls.log"
    (master / ".venv" / "bin").mkdir(parents=True)
    (master / ".venv" / "bin" / "activate").write_text("", encoding="utf-8")
    (master / "analysis").mkdir()
    bindir.mkdir()
    tool = f'#!/usr/bin/env bash\necho "$(basename "$0") $*" >> "{_p(log)}"\n'
    for path in (master / "run_all.sh", master / "analysis" / "run_derive.sh",
                 master / "analysis" / "run_analysis.sh", master / "analysis" / "run_eval_checkpoints.sh",
                 bindir / "python", bindir / "module", bindir / "dos2unix"):
        path.write_text(tool, encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return master, bindir, log


def _calls(log: Path) -> list[str]:
    return [c for c in log.read_text(encoding="utf-8").splitlines()
            if not c.startswith(("module ", "dos2unix "))]


def _exec_job(job: Path, master: Path, bindir: Path, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    text = job.read_text(encoding="utf-8")
    # The job cd's to the checkout that wrote it; point it at the stub instead.
    lines = [f"cd {_p(master)}" if line.startswith("cd ") else line for line in text.splitlines()]
    stubbed = job.with_name(job.stem + ".stub.sh")
    stubbed.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return subprocess.run([_bash(), _p(stubbed), *args], capture_output=True, text=True, env=env)


# --- qsub_eval_day.sh: VARIANTS ----------------------------------------------------------------

def test_eval_day_variants_list_and_guards(tmp_path):
    day, jobs = _day(tmp_path), tmp_path / "jobs"
    env = {"JOB_DIR": _p(jobs), "VARIANTS": "halueval.constrained,faitheval.mc", "EXTRA": "model.dtype=bfloat16"}
    proc = _run("qsub_eval_day.sh", _p(day), "print", env=env)
    assert proc.returncode == 0, proc.stderr
    job = (jobs / "08-00-00_dpo_ensemble.pbs").read_text(encoding="utf-8")
    assert "EXTRA=('model.dtype=bfloat16' 'variants=[halueval.constrained,faitheval.mc]')" in job
    assert "OUT=outputs/eval/v4-2/dpo_ensemble" in job and "export EVAL_MASTER_REV=" in job
    assert not (jobs / "10-00-00_profile.pbs").exists()

    for bad in ({"VARIANTS": "all", "EXTRA": "faitheval.strict_match=true"},
                {"VARIANTS": "all", "EXTRA": "variants=[all]"},
                {"VARIANTS": "all; rm -rf x"},
                {"EXTRA": "halueval.scoring=both"}):  # legacy guard, no modified root
        assert _run("qsub_eval_day.sh", _p(day), "print", env={"JOB_DIR": _p(jobs), **bad}).returncode == 2, bad


def test_eval_day_reads_the_code_version_from_eval_block(tmp_path):
    day, jobs, root = _day(tmp_path), tmp_path / "jobs", tmp_path / "root"
    root.mkdir()
    (root / "EVAL_BLOCK.txt").write_text("eval_master abc123\nsp_dpo_base def\n", encoding="utf-8")
    env = {"JOB_DIR": _p(jobs), "VARIANTS": "all", "OUT_ROOT": _p(root), "EVAL_MASTER_REV": ""}
    assert _run("qsub_eval_day.sh", _p(day), "print", env=env).returncode == 0
    assert "export EVAL_MASTER_REV=abc123" in (jobs / "08-00-00_dpo_ensemble.pbs").read_text(encoding="utf-8")


def test_eval_day_without_variants_writes_the_legacy_job(tmp_path):
    day, jobs = _day(tmp_path), tmp_path / "jobs"
    assert _run("qsub_eval_day.sh", _p(day), "print", env={"JOB_DIR": _p(jobs)}).returncode == 0
    job = (jobs / "08-00-00_dpo_ensemble.pbs").read_text(encoding="utf-8")
    assert "EXTRA=()" in job and "variants=" not in job
    assert "run_eval_checkpoints.sh --checkpoints \"$CKPT\"" in job and "--resume" in job


# --- qsub_overview_day.sh ------------------------------------------------------------------------

def test_overview_day_writes_ensemble_base_and_derive_jobs(tmp_path):
    day, jobs = _day(tmp_path), tmp_path / "jobs"
    env = {"JOB_DIR": _p(jobs), "BASE_MODEL": "/models/Qwen", "EXTRA": "model.dtype=bfloat16 harness.batch_size=8",
           "RECOMPUTE_STALE": "1"}
    proc = _run("qsub_overview_day.sh", _p(day), "print", env=env)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in jobs.glob("*.pbs")) == [
        "08-00-00_dpo_ensemble.pbs", "09-00-00_off_sp_dpo_asc_ensemble.pbs", "base.pbs", "derive.pbs"]
    ensemble = (jobs / "08-00-00_dpo_ensemble.pbs").read_text(encoding="utf-8")
    assert "'recompute_stale=true' 'variants=[all]'" in ensemble
    assert "overview: 4 job(s)" in proc.stdout

    for job in jobs.glob("*.pbs"):
        assert subprocess.run([_bash(), "-n", _p(job)]).returncode == 0, job.name

    master, bindir, log = _stub_master(tmp_path)
    assert _exec_job(jobs / "base.pbs", master, bindir, "--dry-run").returncode == 0
    call = _calls(log)[-1]
    assert call.startswith("run_all.sh model.id=/models/Qwen output_dir=outputs/eval/v4-2/base/run_0")
    assert "run=[faitheval,halueval,harness] model.dtype=bfloat16 harness.batch_size=8" in call
    assert call.endswith("recompute_stale=true variants=[all] resume=true dry_run=true")


def test_overview_day_derive_job_runs_derive_then_the_status_table(tmp_path):
    jobs, root = tmp_path / "jobs", tmp_path / "root"
    root.mkdir()
    proc = _run("qsub_overview_day.sh", "-", "print",
                env={"JOB_DIR": _p(jobs), "DERIVE_ONLY": "1", "OUT_ROOT": _p(root)})
    assert proc.returncode == 0, proc.stderr
    assert [p.name for p in jobs.glob("*.pbs")] == ["derive.pbs"]
    master, bindir, log = _stub_master(tmp_path)
    run = _exec_job(jobs / "derive.pbs", master, bindir)
    assert run.returncode == 0, run.stdout + run.stderr
    calls = _calls(log)
    assert calls[0] == f"run_derive.sh --root {_p(root)} --variants all"
    assert calls[1].startswith(f"python -m analysis.variants --root {_p(root)} --variants all")
    assert not any(p for p in root.iterdir())  # nothing written into the eval root


def test_overview_day_needs_resources_to_submit_and_a_valid_mode(tmp_path):
    day = _day(tmp_path)
    env = {"JOB_DIR": _p(tmp_path / "jobs"), "QSUB_RES": ""}
    assert _run("qsub_overview_day.sh", _p(day), "submit", env=env).returncode == 2
    assert _run("qsub_overview_day.sh", _p(day), "nope", env=env).returncode == 2


# --- qsub_overview_compare.sh ---------------------------------------------------------------------

def _eval_root(tmp_path: Path, arms) -> Path:
    root = tmp_path / "eval" / "v9"
    for arm in arms:
        (root / (arm if arm == "base" else f"{arm}_ensemble-2epochs") / ("run_0" if arm == "base" else "seed_42")
         ).mkdir(parents=True)
    return root


def test_overview_compare_plans_claim_pairs_and_families(tmp_path):
    root = _eval_root(tmp_path, ["base", "sft", "dpo", "off_sp_dpo_asc", "off_sp_dpo_shuf"])
    jobs, out = tmp_path / "jobs", tmp_path / "analysis"
    env = {"OUT_ROOT": _p(root), "SUFFIX": "-2epochs", "JOB_DIR": _p(jobs), "ANALYSIS_ROOT": _p(out)}
    proc = _run("qsub_overview_compare.sh", "print", env=env)
    assert proc.returncode == 0, proc.stderr
    # sft belongs to both families, so family_orpo is still written
    assert "left out:    claimA_orpo sign_dpo sign_orpo claimB_orpo" + "\n" in proc.stdout
    assert "missing arm: off_sp_dpo_desc" in proc.stdout
    job = jobs / "v9-2epochs.pbs"
    body = job.read_text(encoding="utf-8")
    assert "run claimA_dpo --arm off_sp_dpo_asc=" in body and "--reference off_sp_dpo_shuf" in body
    assert "run family_dpo --reference base --arm base=" in body and "run all_arms" not in body
    assert subprocess.run([_bash(), "-n", _p(job)]).returncode == 0

    master, bindir, log = _stub_master(tmp_path)
    run = _exec_job(job, master, bindir)
    assert run.returncode == 0, run.stdout + run.stderr
    calls = _calls(log)
    assert calls[0] == f"run_derive.sh --root {_p(root)} --variants all"
    assert calls[1].startswith("python -m analysis.variants")
    analyses = [c for c in calls if c.startswith("run_analysis.sh")]
    assert [c.rsplit("/", 1)[-1] for c in analyses] == ["claimA_dpo", "claimB_dpo", "family_dpo", "family_orpo"]
    assert all("--variants-overview --no-compare --benchmarks faitheval,halueval,harness" in c for c in analyses)
    assert analyses[2].endswith(f"--out {_p(out)}/family_dpo")
    assert (out / "status.tsv").exists()


def test_overview_compare_dry_run_and_options(tmp_path):
    root = _eval_root(tmp_path, ["base", "dpo", "orpo"])
    jobs = tmp_path / "jobs"
    env = {"OUT_ROOT": _p(root), "SUFFIX": "-2epochs", "JOB_DIR": _p(jobs), "SETS": "all", "PLOT": "0",
           "DERIVE": "0", "ANALYSIS_ROOT": _p(tmp_path / "a")}
    assert _run("qsub_overview_compare.sh", "print", env=env).returncode == 0
    body = (jobs / "v9-2epochs.pbs").read_text(encoding="utf-8")
    assert "run all_arms --reference base --arm base=" in body and "--arm orpo=" in body

    master, bindir, log = _stub_master(tmp_path)
    run = _exec_job(jobs / "v9-2epochs.pbs", master, bindir, "--dry-run")
    assert run.returncode == 0 and "--no-plot" in run.stdout
    assert not any(c.startswith(("run_analysis.sh", "run_derive.sh"))
                   for c in _calls(log))


def test_overview_compare_rejects_bad_input(tmp_path):
    root = _eval_root(tmp_path, ["base"])
    env = {"OUT_ROOT": _p(root), "JOB_DIR": _p(tmp_path / "jobs")}
    assert _run("qsub_overview_compare.sh", "print", env={**env, "SETS": "claims bogus"}).returncode == 2
    assert _run("qsub_overview_compare.sh", "submit", env={**env, "QSUB_RES": ""}).returncode == 2
    assert _run("qsub_overview_compare.sh", "print",
                env={**env, "OUT_ROOT": _p(tmp_path / "nope")}).returncode == 1
