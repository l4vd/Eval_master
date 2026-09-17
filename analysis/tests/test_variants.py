"""The optional all-variants overview: unit status, planning, and the launcher's overview mode.

Offline: no model, no GPU. The launcher is exercised through Hydra's compose API with its
subprocess runner stubbed out.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from analysis import fixtures
from analysis import variants as V

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir  # noqa: E402

EVAL_MASTER = Path(__file__).resolve().parents[2]
LABEL = "_m_ckpt"


def _cfg(*overrides):
    import run_benchmarks  # noqa: F401  (registers the auto_* resolvers)

    with initialize_config_dir(config_dir=str(EVAL_MASTER / "conf"), version_base=None):
        return compose(config_name="config", overrides=["model.id=/m/ckpt", "python=python",
                                                        "run=[faitheval,halueval,harness]", *overrides])


def _halueval_results(run_dir: Path, task: str, *, raw=True, n=4, directory="halueval", suffix=""):
    d = run_dir / directory
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"index": i, "ground_truth": "Yes" if i % 2 else "No", "judgement": "Yes"} for i in range(n)]
    if raw:
        for r in rows:
            r["raw_judgement"] = "Yes"
    if suffix:
        rows = [{"index": i, "ground_truth": "Yes" if i % 2 else "No", "logp_yes": -1.0, "logp_no": -2.0}
                for i in range(n)]
    results = d / f"{task}_{LABEL}{suffix}_results.json"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = {"task": task, "model": LABEL, "backend": "hf", "batch_size": 8, "sort_by_length": "desc",
               "seed": 42, "num_samples_requested": None, "max_new_tokens": 128, "accuracy": 0.5,
               **({"scoring": "constrained", "auroc": 0.5} if suffix else {})}
    (d / f"{task}_{LABEL}{suffix}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return results


def _faitheval_run(run_dir: Path, task: str, directory="faitheval", **summary_extra):
    d = run_dir / directory
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{task}_predictions.jsonl").write_text(
        json.dumps({"index": 0, "question": "q", "prediction": "unknown", "correct": True}) + "\n",
        encoding="utf-8")
    summary = {"task": task, "num_examples": 1, "accuracy": 1.0, "batch_size": 8,
               "sort_by_length": "desc", "max_new_tokens": 256, **summary_extra}
    (d / f"{task}_summary.json").write_text(json.dumps(summary), encoding="utf-8")


# --- registry ------------------------------------------------------------------------

def test_run_label_matches_halueval():
    import sys

    sys.path.insert(0, str(EVAL_MASTER / "HaluEval-reproduce" / "evaluation"))
    spec = importlib.util.spec_from_file_location(
        "halueval_evaluate_label", EVAL_MASTER / "HaluEval-reproduce" / "evaluation" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    for path in ("/gpfs/a/b/final_checkpoint", r"C:\x\y", "org/model", 'a:b*c?"d<e>f|g'):
        assert V.run_label(path) == evaluate._run_label(path)


def test_every_variant_dir_is_its_analysis_benchmark_and_every_cpu_rule_its_suffix():
    for name, v in V.REGISTRY.items():
        assert v.name == name and name.split(".")[0] == v.base
        if v.kind == V.CPU:
            assert v.source in V.REGISTRY
            assert v.rule == "decontam" or name.endswith("." + v.rule)


def test_resolve_pulls_in_sources_as_non_explicit():
    chosen, explicit = V.resolve_variants(["halueval.constrained_decontam"], ["halueval"])
    assert [v.name for v in chosen] == ["halueval.constrained", "halueval.constrained_decontam"]
    assert explicit == {"halueval.constrained_decontam"}
    all_, _ = V.resolve_variants(["all"], ["faitheval"])
    assert {v.name for v in all_} == {"faitheval", "faitheval.mc", "faitheval.strict",
                                      "faitheval.wordmatch", "faitheval.contains"}
    with pytest.raises(ValueError, match="unknown"):
        V.resolve_variants(["halueval.nope"], ["halueval"])


def test_a_pulled_in_source_is_planned_only_for_the_tasks_its_dependant_needs(tmp_path):
    chosen, explicit = V.resolve_variants(["halueval.decontam"], ["halueval"])
    units = V.plan(tmp_path, chosen, V.Context(), explicit=explicit)
    assert [u.key for u in units] == [("halueval", "summarization"), ("halueval.decontam", "summarization")]


# --- status --------------------------------------------------------------------------

def test_status_matrix_and_cpu_units_waiting_on_sources(tmp_path):
    ctx = V.Context(halueval_label=LABEL)
    _halueval_results(tmp_path, "qa")
    chosen, explicit = V.resolve_variants(["all"], ["halueval"])
    status = {u.key: u.status for u in V.plan(tmp_path, chosen, ctx, explicit=explicit)}
    assert status[("halueval", "qa")].state == V.DONE
    assert status[("halueval", "dialogue")].state == V.MISSING
    assert status[("halueval.parsed", "qa")] == V.Status(V.MISSING)
    assert status[("halueval.parsed", "dialogue")] == V.Status(V.MISSING, "waiting on halueval")
    assert status[("halueval.constrained_decontam", "summarization")].reason == "waiting on halueval.constrained"
    table = V.format_matrix(list(V.plan(tmp_path, chosen, ctx, explicit=explicit)))
    assert "halueval.parsed" in table and "missing(waiting on halueval)" in table


def test_gpu_stale_on_num_samples_and_label(tmp_path):
    _halueval_results(tmp_path, "qa")
    expected = {"halueval": {"num_samples": None, "seed": 42, "batch_size": 8}}
    ctx = V.Context(halueval_label=LABEL, expected=expected)
    assert V.unit_status(tmp_path, "halueval", "qa", ctx).state == V.DONE
    ctx.expected["halueval"]["num_samples"] = 5
    assert V.unit_status(tmp_path, "halueval", "qa", ctx) == V.Status(V.STALE, "num_samples None != 5")
    other = V.Context(halueval_label="_other_ckpt")
    assert V.unit_status(tmp_path, "halueval", "qa", other).state == V.STALE


def test_cpu_stale_when_the_source_changes(tmp_path):
    results = _halueval_results(tmp_path, "qa")
    out = tmp_path / "halueval.parsed"
    out.mkdir()
    summary = {"source_sha256": hashlib.sha256(results.read_bytes()).hexdigest()}
    (out / f"qa_{LABEL}_parsed_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    ctx = V.Context(halueval_label=LABEL)
    assert V.unit_status(tmp_path, "halueval.parsed", "qa", ctx).state == V.DONE
    with results.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"index": 9, "ground_truth": "No", "judgement": "No", "raw_judgement": "No"}) + "\n")
    assert V.unit_status(tmp_path, "halueval.parsed", "qa", ctx) == V.Status(V.STALE, "source changed")


def test_lenient_without_raw_judgement_is_unavailable_not_missing(tmp_path):
    _halueval_results(tmp_path, "qa", raw=False)
    ctx = V.Context(halueval_label=LABEL)
    status = V.unit_status(tmp_path, "halueval.lenient", "qa", ctx)
    assert status == V.Status(V.UNAVAILABLE, "no raw_judgement") and status.settled
    unit = V.Unit(V.REGISTRY["halueval.lenient"], "qa", status)
    assert not V.to_compute(unit, resume=True, recompute_stale=True)


def test_old_faitheval_summary_without_the_dataset_is_unverified(tmp_path):
    _faitheval_run(tmp_path, "unanswerable")  # no num_samples recorded
    expected = {"faitheval": {"num_samples": None, "batch_size": 8}}
    ctx = V.Context(expected=expected, faitheval_data_dir=tmp_path / "no-data")
    status = V.unit_status(tmp_path, "faitheval", "unanswerable", ctx)
    assert status == V.Status(V.UNVERIFIED, "num_samples") and status.counts_as_done
    ctx.strict_provenance = True
    assert V.unit_status(tmp_path, "faitheval", "unanswerable", ctx).state == V.STALE

    data = tmp_path / "data" / "FaithEval-unanswerable-v1.0"
    data.mkdir(parents=True)
    (data / "test.jsonl").write_text('{"q": 1}\n{"q": 2}\n', encoding="utf-8")
    ctx = V.Context(expected=expected, faitheval_data_dir=tmp_path / "data")
    assert V.unit_status(tmp_path, "faitheval", "unanswerable", ctx).state == V.STALE  # 1 row of 2
    ctx.expected["faitheval"]["num_samples"] = 1
    assert V.unit_status(tmp_path, "faitheval", "unanswerable", ctx).state == V.DONE


def test_a_strict_match_run_in_the_original_dir_is_stale(tmp_path):
    _faitheval_run(tmp_path, "unanswerable", strict_match=True, num_samples=None)
    assert V.unit_status(tmp_path, "faitheval", "unanswerable").state == V.STALE


def test_harness_needs_both_files(tmp_path):
    fixtures.write_harness(tmp_path / "harness", [{"task": "truthfulqa_mc2", "value": 0.4}])
    assert V.unit_status(tmp_path, "harness", V.WHOLE).state == V.MISSING
    (tmp_path / "harness" / "lm_eval_results.json").write_text("{}", encoding="utf-8")
    assert V.unit_status(tmp_path, "harness", V.WHOLE).state == V.DONE


def test_derive_writes_only_sibling_dirs_and_is_idempotent(tmp_path):
    _halueval_results(tmp_path, "qa")
    _faitheval_run(tmp_path, "unanswerable")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    chosen, explicit = V.resolve_variants(["halueval.parsed", "halueval.lenient", "faitheval.strict",
                                           "faitheval.wordmatch"], ["halueval", "faitheval"])
    ctx = V.Context(halueval_label=LABEL)
    units = V.plan(tmp_path, chosen, ctx, explicit=explicit)
    result = V.derive_run_dir(tmp_path, units, ctx)
    assert sorted(result["ok"]) == [("faitheval.strict", "unanswerable"), ("faitheval.wordmatch", "unanswerable"),
                                    ("halueval.lenient", "qa"), ("halueval.parsed", "qa")]
    assert all(p.read_bytes() == b for p, b in before.items())
    new = [p for p in tmp_path.rglob("*") if p.is_file() and p not in before]
    assert all("." in p.parent.name for p in new) and len(new) == 4
    again = V.derive_run_dir(tmp_path, V.plan(tmp_path, chosen, ctx, explicit=explicit), ctx)
    assert again["ok"] == [] and again["failed"] == []


# --- launcher: overview mode ---------------------------------------------------------

def test_both_only_when_both_scorings_are_missing(tmp_path):
    import run_benchmarks as rb

    cfg = _cfg("variants=[halueval.constrained]", "run=[halueval]")
    _halueval_results(tmp_path, "qa")  # generate done for qa only
    ctx = rb.overview_context(cfg)
    chosen, explicit = V.resolve_variants(["halueval", "halueval.constrained"], ["halueval"])
    units = [u for u in V.plan(tmp_path, chosen, ctx, explicit=explicit)
             if V.to_compute(u, resume=True, recompute_stale=False)]
    commands = {tuple(u.key for u in c["units"]): c["cmd"] for c in rb.gpu_commands(cfg, tmp_path, units)}
    qa = commands[(("halueval.constrained", "qa"),)]
    assert qa[qa.index("--scoring") + 1] == "constrained"
    assert qa[qa.index("--output-dir") + 1] == str(tmp_path / "halueval.constrained")
    assert "--constrained-output-dir" not in qa
    dialogue = commands[(("halueval", "dialogue"), ("halueval.constrained", "dialogue"))]
    assert dialogue[dialogue.index("--scoring") + 1] == "both"
    assert dialogue[dialogue.index("--output-dir") + 1] == str(tmp_path / "halueval")
    assert dialogue[dialogue.index("--constrained-output-dir") + 1] == str(tmp_path / "halueval.constrained")
    assert dialogue[dialogue.index("--task") + 1] == "dialogue"


def test_run_suite_dry_run_on_an_empty_dir_plans_everything(tmp_path, capsys):
    import run_benchmarks as rb

    cfg = _cfg("variants=[all]", "resume=true", "dry_run=true")
    calls = []
    assert rb.run_suite(cfg, tmp_path / "run", runner=lambda *a, **k: calls.append(a)) == 0
    out = capsys.readouterr().out
    assert calls == [] and not (tmp_path / "run").exists()
    assert out.count("--scoring both") == 3 and "faitheval.mc/counterfactual_mc" in out
    assert "after GPU]  derive: faitheval.contains/counterfactual" in out


def test_run_suite_refuses_legacy_switches(tmp_path):
    import run_benchmarks as rb

    for override in ("halueval.scoring=both", "halueval.decontam.enabled=true", "faitheval.strict_match=true",
                     "faitheval.tasks=[unanswerable,counterfactual_mc]"):
        assert rb.run_suite(_cfg("variants=[all]", override), tmp_path) == 2, override


def test_run_suite_refuses_stale_gpu_units_unless_asked(tmp_path):
    import run_benchmarks as rb

    run = tmp_path / "run"
    _faitheval_run(run, "unanswerable", num_samples=5)
    cfg = _cfg("variants=[faitheval]", "run=[faitheval]", "resume=true", "faitheval.tasks=[unanswerable]")
    calls = []
    runner = lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0)  # noqa: E731
    assert rb.run_suite(cfg, run, runner=runner) == 2 and calls == []
    cfg = _cfg("variants=[faitheval]", "run=[faitheval]", "resume=true", "faitheval.tasks=[unanswerable]",
               "recompute_stale=true")
    assert rb.run_suite(cfg, run, runner=runner) == 0 and len(calls) == 1


def test_markers_are_deleted_before_the_launch_and_a_crash_reads_as_missing(tmp_path, monkeypatch):
    import run_benchmarks as rb

    run = tmp_path / "run"
    _faitheval_run(run, "unanswerable", num_samples=5)
    cfg = _cfg("variants=[faitheval]", "run=[faitheval]", "faitheval.tasks=[unanswerable]", "num_samples=5")
    seen = []

    def runner(cmd, **kwargs):
        seen.append((run / "faitheval" / "unanswerable_summary.json").exists())
        return subprocess.CompletedProcess(cmd, 1)  # the run dies

    monkeypatch.setenv("EVAL_MASTER_REV", "abc123")
    assert rb.run_suite(cfg, run, runner=runner) == 1  # resume=false: everything re-runs
    assert seen == [False]  # the old summary was gone before the process started
    assert V.unit_status(run, "faitheval", "unanswerable").state == V.MISSING
    entry = json.loads((run / "launches.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert entry["code_version"] == "abc123" and entry["code_version_source"] == "EVAL_MASTER_REV"
    assert entry["ran"] == [{"unit": "faitheval/unanswerable", "was": "done", "result": "FAILED (exit 1)"}]


def test_code_version_falls_back_to_git_then_null(monkeypatch):
    import run_benchmarks as rb

    monkeypatch.setenv("EVAL_MASTER_REV", "fromenv")
    assert rb.code_version() == ("fromenv", "EVAL_MASTER_REV")
    monkeypatch.delenv("EVAL_MASTER_REV")
    monkeypatch.setattr(rb.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="deadbeef\n"))
    assert rb.code_version() == ("deadbeef", "git")
    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    assert rb.code_version() == (None, "none")


# The last commit of run_benchmarks.py before the overview existed.
PRE_OVERVIEW_COMMIT = "5c8cf56"


def _committed_launcher():
    """run_benchmarks.py as of PRE_OVERVIEW_COMMIT, loaded under another module name."""
    proc = subprocess.run(["git", "-C", str(EVAL_MASTER), "show", f"{PRE_OVERVIEW_COMMIT}:run_benchmarks.py"],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        pytest.skip(f"run_benchmarks.py at {PRE_OVERVIEW_COMMIT} unavailable (no git history)")
    source = re.sub(r"register_new_resolver\((.*)\)", r"register_new_resolver(\1, replace=True)", proc.stdout)
    assert "def run_suite" not in source
    module = type(importlib)("run_benchmarks_head")
    module.__file__ = str(EVAL_MASTER / "run_benchmarks.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("overrides", [
    [], ["num_samples=5"], ["halueval.scoring=both"], ["halueval.decontam.enabled=true"],
    ["faitheval.strict_match=true", "faitheval.tasks=[unanswerable,counterfactual_mc]"],
    ["halueval.max_new_tokens=16", "halueval.extra_args=[--foo]",
     "run=[faitheval,truthfulqa,halueval,ragtruth,harness]"],
])
def test_legacy_commands_are_unchanged_without_variants(tmp_path, overrides):
    import run_benchmarks as rb

    head = _committed_launcher()
    cfg = _cfg(*overrides)
    assert cfg.variants is None
    for name in rb.BUILDERS:
        assert rb.BUILDERS[name](cfg, tmp_path) == head.BUILDERS[name](cfg, tmp_path), name


# --- ensembles: eval_checkpoints --resume with variants= -------------------------------

def _ensemble(tmp_path, extra):
    from analysis import eval_checkpoints

    (tmp_path / "grp" / "seed_1" / "final_checkpoint").mkdir(parents=True)
    return eval_checkpoints.plan_evaluations(str(tmp_path / "grp"), tmp_path / "eval",
                                             benchmarks=["faitheval"], extra=extra)


def _record_calls(monkeypatch):
    from analysis import eval_checkpoints

    calls = []
    monkeypatch.setattr(eval_checkpoints.subprocess, "run",
                        lambda cmd, *a, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    return calls


def test_overview_resume_runs_a_seed_with_any_missing_unit_and_forwards_resume(tmp_path, monkeypatch):
    from analysis import eval_checkpoints

    jobs = _ensemble(tmp_path, ["variants=[faitheval,faitheval.strict]", "faitheval.tasks=[unanswerable]"])
    run = jobs[0].out_dir
    _faitheval_run(run, "unanswerable")  # the original finished; the strict variant did not
    # the legacy check would call this seed complete
    assert eval_checkpoints.completed_benchmarks(run, ["faitheval"], jobs[0].command) == ["faitheval"]
    assert eval_checkpoints.overview_pending(jobs[0]) == ["faitheval.strict/unanswerable"]
    calls = _record_calls(monkeypatch)
    eval_checkpoints.run_evaluations(jobs, resume=True)
    assert len(calls) == 1 and calls[0][-1] == "resume=true"


def test_overview_resume_skips_a_seed_whose_units_are_all_done(tmp_path, monkeypatch):
    from analysis import eval_checkpoints

    jobs = _ensemble(tmp_path, ["variants=[faitheval.strict]", "faitheval.tasks=[unanswerable]"])
    run = jobs[0].out_dir
    _faitheval_run(run, "unanswerable")
    ctx = V.Context(tasks={"faitheval": ("unanswerable",)})
    chosen, explicit = V.resolve_variants(["faitheval.strict"], ["faitheval"])
    assert V.derive_run_dir(run, V.plan(run, chosen, ctx, explicit=explicit), ctx)["ok"]
    calls = _record_calls(monkeypatch)
    eval_checkpoints.run_evaluations(jobs, resume=True)
    assert calls == []


def test_legacy_resume_is_unchanged_without_variants(tmp_path, monkeypatch):
    from analysis import eval_checkpoints

    jobs = _ensemble(tmp_path, [])
    _faitheval_run(jobs[0].out_dir, "unanswerable")
    calls = _record_calls(monkeypatch)
    eval_checkpoints.run_evaluations(jobs, resume=True)
    assert calls == []
    assert eval_checkpoints.overview_variants(jobs[0].command) is None
