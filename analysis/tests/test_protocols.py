"""Modified protocols never leak into the original numbers.

Routing by the summary's own ``scoring`` / ``variant``, base-vs-dotted benchmark
selection, merged ``--arm`` specs, the separate ``modified/`` output tree (with the
original tree byte-identical to what it was without it), and variant-aware ``--resume``.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from analysis import eval_checkpoints, fixtures
from analysis.cli import main
from analysis.eval_checkpoints import completed_benchmarks
from analysis.model import MetricRecord, RecordSet, benchmark_selected, protocol_of
from analysis.parse import parse_run_dir
from analysis.spec import AnalysisConfig, build_records, parse_arm_arg


def _by_key(records):
    return {(r.benchmark, r.task, r.metric): r for r in records}


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- names ---------------------------------------------------------------------------

def test_protocol_is_read_from_the_benchmark_name():
    assert protocol_of("halueval") == "original"
    assert protocol_of("halueval.constrained") == "modified"


def test_a_base_name_selects_its_variants_and_a_dotted_name_only_itself():
    assert benchmark_selected("halueval.constrained", ["halueval"])
    assert benchmark_selected("halueval.constrained", ["halueval.constrained"])
    assert not benchmark_selected("halueval", ["halueval.constrained"])
    assert not benchmark_selected("halueval.decontam", ["halueval.constrained"])
    assert not benchmark_selected("faitheval.mc", ["halueval"])


def test_recordset_selection_and_protocol_split():
    def rec(benchmark):
        return MetricRecord("a", 1, benchmark, "t", "m", 0.5, None, True, 10)

    rs = RecordSet([rec("halueval"), rec("halueval.constrained"), rec("faitheval.mc")])
    assert [r.benchmark for r in rs.include_benchmarks(["halueval"], None)] == ["halueval", "halueval.constrained"]
    assert [r.benchmark for r in rs.include_benchmarks(None, ["halueval"])] == ["faitheval.mc"]
    assert [r.benchmark for r in rs.by_protocol("original")] == ["halueval"]
    assert len(rs.by_protocol("modified")) == 2


# --- parsing -----------------------------------------------------------------------

def test_halueval_variants_are_routed_and_originals_parse_as_before(tmp_path):
    bench = tmp_path / "halueval"
    fixtures.write_halueval(bench, {"summarization": 0.5, "qa": 0.6})
    fixtures.write_halueval_constrained(bench, {"summarization": 0.7})
    fixtures.write_halueval_decontam(bench, {"summarization": (0.52, 0.40)})
    fixtures.write_halueval_decontam(bench, {"summarization": (0.71, 0.65)}, constrained=True)
    recs = parse_run_dir(tmp_path, "arm", 1)
    m = _by_key(recs)

    assert {r.benchmark for r in recs} == {
        "halueval", "halueval.constrained", "halueval.decontam", "halueval.constrained_decontam"}
    # The original benchmark sees only the two original summaries, mean included.
    assert m[("halueval", "summarization", "accuracy")].value == 0.5
    assert m[("halueval", "mean", "accuracy")].value == pytest.approx(0.55)
    assert sorted(r.task for r in recs if r.benchmark == "halueval") == ["mean", "qa", "summarization"]

    auroc = m[("halueval.constrained", "summarization", "auroc")]
    assert auroc.is_primary and auroc.stderr == 0.02 and auroc.n_samples == 100
    assert not m[("halueval.constrained", "summarization", "mean_verdict_mass")].is_primary
    assert m[("halueval.constrained", "summarization", "frac_mass_below_half")].higher_is_better is False

    decontam = m[("halueval.decontam", "summarization", "accuracy")]
    assert decontam.is_primary and decontam.value == 0.52 and decontam.n_samples == 90
    contrast = m[("halueval.decontam", "summarization", "seen_minus_clean")]
    assert contrast.value == pytest.approx(-0.12) and contrast.stderr == 0.02 and not contrast.is_primary
    seen = m[("halueval.decontam", "summarization", "accuracy_seen")]
    assert (seen.value, seen.n_samples, seen.stderr) == (0.40, 10, None)

    assert m[("halueval.constrained_decontam", "summarization", "auroc")].is_primary
    assert m[("halueval.constrained_decontam", "summarization", "auroc_seen")].stderr == 0.01


def test_faitheval_mc_is_routed_out_of_the_original_mean(tmp_path):
    fixtures.write_faitheval(tmp_path / "faitheval", {"counterfactual": 0.1, "unanswerable": 0.3})
    fixtures.write_faitheval_mc(tmp_path / "faitheval", 0.6, accuracy_norm=0.55)
    m = _by_key(parse_run_dir(tmp_path, "arm", 1))
    assert m[("faitheval", "mean", "accuracy")].value == pytest.approx(0.2)
    assert m[("faitheval.mc", "counterfactual_mc", "accuracy")].is_primary
    assert m[("faitheval.mc", "counterfactual_mc", "accuracy_norm")].value == 0.55
    assert ("faitheval", "counterfactual_mc", "accuracy") not in m


def test_parse_run_dir_selection_understands_variants(tmp_path):
    fixtures.write_halueval(tmp_path / "halueval", {"qa": 0.5})
    fixtures.write_halueval_constrained(tmp_path / "halueval", {"qa": 0.6})
    only = parse_run_dir(tmp_path, "arm", 1, benchmarks=["halueval.constrained"])
    assert {r.benchmark for r in only} == {"halueval.constrained"}
    both = parse_run_dir(tmp_path, "arm", 1, benchmarks=["halueval"])
    assert {r.benchmark for r in both} == {"halueval", "halueval.constrained"}


# --- arms and trees ------------------------------------------------------------------

def _roots(tmp_path, *, modified: bool):
    """Two 3-seed arms in an original root, and (optionally) their modified-protocol root."""
    orig, mod = tmp_path / "eval", tmp_path / "eval_mod"
    for arm, base in (("ref", 0.40), ("cur", 0.45)):
        for k, seed in enumerate((1, 2, 3)):
            fixtures.write_full_run(
                orig / arm / f"seed_{seed}", seed=seed,
                faitheval={"counterfactual": base + 0.01 * k},
                halueval={"summarization": base + 0.02 * k},
                harness_rows=[{"task": "truthfulqa_mc2", "metric": "acc", "value": base + 0.03 * k, "stderr": 0.015}],
            )
            if modified:
                fixtures.write_full_run(
                    mod / arm / f"seed_{seed}", seed=seed,
                    halueval_constrained={"summarization": base + 0.2 + 0.01 * k},
                    faitheval_mc=base + 0.1 + 0.01 * k,
                )
    return orig, mod


_TREE_FILES = ("records.jsonl", "aggregate.json", "aggregate.tex", "comparisons.json", "comparisons.tex")


def test_original_tree_is_byte_identical_when_modified_roots_are_added(tmp_path):
    orig, mod = _roots(tmp_path, modified=True)
    plain = tmp_path / "plain"
    assert main(["--arm", f"ref={orig / 'ref'}", "--arm", f"cur={orig / 'cur'}", "--reference", "ref",
                 "--no-plot", "--out", str(plain)]) == 0
    assert not (plain / "modified").exists()  # nothing modified in, no modified tree out

    merged = tmp_path / "merged"
    assert main(["--arm", f"ref={orig / 'ref'}", "--arm", f"ref={mod / 'ref'}",
                 "--arm", f"cur={orig / 'cur'}", "--arm", f"cur={mod / 'cur'}",
                 "--reference", "ref", "--no-plot", "--out", str(merged)]) == 0
    for name in _TREE_FILES:
        assert (merged / name).read_bytes() == (plain / name).read_bytes(), name

    tree = merged / "modified"
    assert {r["benchmark"] for r in _jsonl(tree / "records.jsonl")} == {"halueval.constrained", "faitheval.mc"}
    comparisons = json.loads((tree / "comparisons.json").read_text(encoding="utf-8"))
    assert comparisons and {c["benchmark"] for c in comparisons} == {"halueval.constrained", "faitheval.mc"}
    assert (tree / "aggregate.tex").read_text(encoding="utf-8").startswith("% Protocol: modified")
    assert (tree / "comparisons.tex").read_text(encoding="utf-8").startswith("% Protocol: modified")
    assert not (merged / "aggregate.tex").read_text(encoding="utf-8").startswith("%")


def test_protocol_flag_selects_which_trees_are_written(tmp_path):
    orig, mod = _roots(tmp_path, modified=True)
    arms = ["--arm", f"ref={orig / 'ref'}", "--arm", f"ref={mod / 'ref'}",
            "--arm", f"cur={orig / 'cur'}", "--arm", f"cur={mod / 'cur'}", "--reference", "ref", "--no-plot"]
    main([*arms, "--protocol", "original", "--out", str(tmp_path / "o")])
    assert (tmp_path / "o" / "records.jsonl").is_file() and not (tmp_path / "o" / "modified").exists()
    main([*arms, "--protocol", "modified", "--out", str(tmp_path / "m")])
    assert not (tmp_path / "m" / "records.jsonl").exists()
    assert (tmp_path / "m" / "modified" / "records.jsonl").is_file()


def test_benchmark_filter_carries_a_base_names_variants_into_the_modified_tree(tmp_path):
    orig, mod = _roots(tmp_path, modified=True)
    out = tmp_path / "out"
    main(["--arm", f"cur={orig / 'cur'}", "--arm", f"cur={mod / 'cur'}", "--no-compare", "--no-plot",
          "--benchmarks", "halueval", "--out", str(out)])
    assert {r["benchmark"] for r in _jsonl(out / "records.jsonl")} == {"halueval"}
    assert {r["benchmark"] for r in _jsonl(out / "modified" / "records.jsonl")} == {"halueval.constrained"}


def test_a_reference_without_modified_runs_skips_only_the_modified_comparisons(tmp_path):
    orig, mod = _roots(tmp_path, modified=True)
    out = tmp_path / "out"
    assert main(["--arm", f"ref={orig / 'ref'}", "--arm", f"cur={orig / 'cur'}", "--arm", f"cur={mod / 'cur'}",
                 "--reference", "ref", "--no-plot", "--out", str(out)]) == 0
    assert (out / "comparisons.json").is_file()
    assert (out / "modified" / "records.jsonl").is_file() and not (out / "modified" / "comparisons.json").exists()


def test_merged_arms_keep_seeds_and_fixed_points(tmp_path):
    orig, mod = _roots(tmp_path, modified=True)
    base_orig, base_mod = orig / "base" / "run_0", mod / "base" / "run_0"
    fixtures.write_halueval(base_orig / "halueval", {"summarization": 0.5})
    fixtures.write_halueval_constrained(base_mod / "halueval", {"summarization": 0.5})
    build = build_records(AnalysisConfig(arms=[
        parse_arm_arg(f"base={base_orig}"), parse_arm_arg(f"base={base_mod}"),
        parse_arm_arg(f"cur={orig / 'cur'}"), parse_arm_arg(f"cur={mod / 'cur'}"),
    ], compare=False, plot=False))
    assert build.arm_meta["base"].is_fixed_point is True
    assert build.arm_meta["cur"].seeds == [1, 2, 3] and len(build.arm_meta["cur"].run_dirs) == 6
    assert build.arm_meta["cur"].is_fixed_point is False


def test_merged_arm_refuses_overlapping_roots(tmp_path):
    orig, _ = _roots(tmp_path, modified=False)
    with pytest.raises(ValueError, match="must not overlap"):
        main(["--arm", f"cur={orig / 'cur'}", "--arm", f"cur={orig / 'cur'}", "--no-compare", "--no-plot",
              "--out", str(tmp_path / "x")])


# --- completion / resume ---------------------------------------------------------------

def test_halueval_completion_follows_the_scoring_mode(tmp_path):
    fixtures.write_halueval(tmp_path / "halueval", {"qa": 0.5})
    assert completed_benchmarks(tmp_path, ["halueval"]) == ["halueval"]
    assert completed_benchmarks(tmp_path, ["halueval"], ["halueval.scoring=constrained"]) == []
    fixtures.write_halueval_constrained(tmp_path / "halueval", {"qa": 0.6})
    assert completed_benchmarks(tmp_path, ["halueval"], ["halueval.scoring=constrained"]) == ["halueval"]
    assert completed_benchmarks(tmp_path, ["halueval"], ["halueval.scoring=both"]) == ["halueval"]

    constrained_only = tmp_path / "c"
    fixtures.write_halueval_constrained(constrained_only / "halueval", {"qa": 0.6})
    assert completed_benchmarks(constrained_only, ["halueval"]) == []  # not the original protocol
    assert completed_benchmarks(constrained_only, ["halueval"], ["halueval.scoring=both"]) == []


def test_faitheval_mc_and_ragtruth_stage_completion(tmp_path):
    fixtures.write_faitheval(tmp_path / "faitheval", {"counterfactual": 0.1})
    mc = ["faitheval.tasks=[counterfactual_mc]"]
    assert completed_benchmarks(tmp_path, ["faitheval"], mc) == []
    fixtures.write_faitheval_mc(tmp_path / "faitheval", 0.5)
    assert completed_benchmarks(tmp_path, ["faitheval"], mc) == ["faitheval"]
    mc_only = tmp_path / "mc"
    fixtures.write_faitheval_mc(mc_only / "faitheval", 0.5)
    assert completed_benchmarks(mc_only, ["faitheval"]) == []

    (tmp_path / "ragtruth").mkdir()
    (tmp_path / "ragtruth" / "generation_summary.json").write_text("{}", encoding="utf-8")
    assert completed_benchmarks(tmp_path, ["ragtruth"], ["ragtruth.stage=generate"]) == ["ragtruth"]
    assert completed_benchmarks(tmp_path, ["ragtruth"]) == []  # stage=all needs detection's summary
    (tmp_path / "ragtruth" / "summary.json").write_text("{}", encoding="utf-8")
    assert completed_benchmarks(tmp_path, ["ragtruth"]) == ["ragtruth"]


def test_resume_reads_the_variant_from_the_jobs_own_overrides(tmp_path, monkeypatch):
    (tmp_path / "grp" / "seed_1" / "final_checkpoint").mkdir(parents=True)
    out = tmp_path / "eval"
    fixtures.write_halueval(out / "seed_1" / "halueval", {"qa": 0.5})  # the original run finished
    calls = []
    monkeypatch.setattr(eval_checkpoints.subprocess, "run",
                        lambda cmd, *a, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))

    constrained = eval_checkpoints.plan_evaluations(str(tmp_path / "grp"), out, benchmarks=["halueval"],
                                                    extra=["halueval.scoring=constrained"])
    eval_checkpoints.run_evaluations(constrained, resume=True)
    assert len(calls) == 1  # an original summary does not finish a constrained run

    original = eval_checkpoints.plan_evaluations(str(tmp_path / "grp"), out, benchmarks=["halueval"])
    eval_checkpoints.run_evaluations(original, resume=True)
    assert len(calls) == 1  # ...but it does finish the original one


# --- the optional all-variants overview: sibling <bench>.<variant>/ dirs ----------------

def _with_siblings(run_dir, base=0.4):
    fixtures.write_halueval_constrained(run_dir / "halueval.constrained", {"qa": base + 0.3})
    fixtures.write_halueval_variant(run_dir / "halueval.parsed", {"qa": base + 0.1}, "parsed")
    fixtures.write_halueval_variant(run_dir / "halueval.lenient", {"qa": base + 0.05}, "lenient")
    fixtures.write_faitheval_rescored(run_dir / "faitheval.strict", {"unanswerable": base - 0.2}, "strict")
    fixtures.write_faitheval_rescored(run_dir / "faitheval.contains", {"counterfactual": base - 0.1}, "contains",
                                      strata={"1_5": 0.6, "61_plus": 0.1})
    fixtures.write_faitheval_mc(run_dir / "faitheval.mc", base + 0.2)


def test_sibling_dirs_are_read_only_with_the_overview_flag(tmp_path):
    run = fixtures.write_full_run(tmp_path / "run", seed=1, faitheval={"unanswerable": 0.3},
                                  halueval={"qa": 0.5})
    _with_siblings(run)
    off = parse_run_dir(run, "arm", 1)
    assert {r.benchmark for r in off} == {"faitheval", "halueval"}
    on = _by_key(parse_run_dir(run, "arm", 1, variants_overview=True))
    assert {b for b, _, _ in on} == {"faitheval", "halueval", "halueval.constrained", "halueval.parsed",
                                     "halueval.lenient", "faitheval.strict", "faitheval.contains",
                                     "faitheval.mc"}
    assert on[("halueval.parsed", "qa", "accuracy")].is_primary
    assert on[("faitheval.strict", "unanswerable", "accuracy")].value == pytest.approx(0.2)
    assert not on[("faitheval.strict", "unanswerable", "mean_prediction_words")].is_primary
    stratum = on[("faitheval.contains", "counterfactual", "accuracy_len_1_5")]
    assert (stratum.value, stratum.n_samples, stratum.is_primary) == (0.6, 10, False)
    # the original records are the same objects either way
    assert [r for r in on.values() if r.benchmark in ("faitheval", "halueval")] == off


def test_a_summary_in_the_wrong_variant_dir_is_skipped_with_a_warning(tmp_path):
    run = fixtures.write_full_run(tmp_path / "run", seed=1, halueval={"qa": 0.5})
    fixtures.write_halueval_variant(run / "halueval.lenient", {"qa": 0.7}, "parsed")
    with pytest.warns(UserWarning, match="wrong variant dir"):
        recs = parse_run_dir(run, "arm", 1, variants_overview=True)
    assert {r.benchmark for r in recs} == {"halueval"}


def test_a_variant_in_both_locations_is_an_error(tmp_path):
    run = fixtures.write_full_run(tmp_path / "run", seed=1, halueval={"qa": 0.5},
                                  halueval_constrained={"qa": 0.6})
    fixtures.write_halueval_constrained(run / "halueval.constrained", {"qa": 0.6})
    assert {r.benchmark for r in parse_run_dir(run, "arm", 1)} == {"halueval", "halueval.constrained"}
    with pytest.raises(ValueError, match="both"):
        parse_run_dir(run, "arm", 1, variants_overview=True)


def test_strict_match_run_routes_to_faitheval_strict(tmp_path):
    fixtures.write_faitheval(tmp_path / "faitheval", {"unanswerable": 0.12}, strict_match=True)
    fixtures.write_faitheval(tmp_path / "f2" / "faitheval", {"unanswerable": 0.3}, strict_match=False)
    assert {r.benchmark for r in parse_run_dir(tmp_path, "arm", 1)} == {"faitheval.strict"}
    assert {r.benchmark for r in parse_run_dir(tmp_path / "f2", "arm", 1)} == {"faitheval"}


def test_a_run_dir_with_only_sibling_dirs_is_found_only_with_the_flag(tmp_path):
    from analysis.discover import expand_spec

    fixtures.write_faitheval_mc(tmp_path / "grp" / "x" / "faitheval.mc", 0.5)
    assert expand_spec(str(tmp_path / "grp" / "x")) == []
    assert expand_spec(str(tmp_path / "grp" / "x"), variants_overview=True) == [tmp_path / "grp" / "x"]


def _overview_roots(tmp_path):
    orig, _ = _roots(tmp_path, modified=False)
    plain = tmp_path / "plain_copy"
    import shutil

    shutil.copytree(orig, plain)
    for arm in ("ref", "cur"):
        for seed in (1, 2, 3):
            _with_siblings(orig / arm / f"seed_{seed}", base=0.4 + 0.01 * seed + (0.05 if arm == "cur" else 0))
    return orig, plain


_ORIGINAL_TREE = ("records.jsonl", "aggregate.json", "aggregate.tex", "comparisons.json", "comparisons.tex")


@pytest.mark.parametrize("flag", [[], ["--variants-overview"]])
def test_original_tree_is_byte_identical_with_sibling_dirs_present(tmp_path, flag):
    orig, plain = _overview_roots(tmp_path)
    untouched, overview = tmp_path / "a", tmp_path / "b"
    common = ["--reference", "ref", "--no-plot"]
    assert main(["--arm", f"ref={plain / 'ref'}", "--arm", f"cur={plain / 'cur'}", *common,
                 "--out", str(untouched)]) == 0
    assert main(["--arm", f"ref={orig / 'ref'}", "--arm", f"cur={orig / 'cur'}", *common, *flag,
                 "--out", str(overview)]) == 0
    for name in _ORIGINAL_TREE:
        assert (overview / name).read_bytes() == (untouched / name).read_bytes(), name
    if not flag:
        assert not (overview / "modified").exists() and not (overview / "overview").exists()
        return
    benchmarks = {r["benchmark"] for r in _jsonl(overview / "modified" / "records.jsonl")}
    assert {"halueval.parsed", "halueval.lenient", "faitheval.strict", "faitheval.contains",
            "halueval.constrained", "faitheval.mc"} <= benchmarks
    assert (overview / "overview" / "overview.tsv").is_file()
    assert not list((overview / "overview").rglob("comparisons.json"))
