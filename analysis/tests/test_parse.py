"""Each parser yields the right MetricRecords with correct direction / primary tags."""

from __future__ import annotations

import warnings

from analysis import fixtures
from analysis.parse import parse_run_dir


def _by_key(records):
    return {(r.benchmark, r.task, r.metric): r for r in records}


def test_faitheval_per_task_and_mean(tmp_path):
    fixtures.write_faitheval(tmp_path / "faitheval",
                             {"counterfactual": 0.6, "inconsistent": 0.4})
    recs = parse_run_dir(tmp_path, "arm", 42)
    m = _by_key(recs)
    assert m[("faitheval", "counterfactual", "accuracy")].value == 0.6
    assert m[("faitheval", "counterfactual", "accuracy")].higher_is_better is True
    assert m[("faitheval", "counterfactual", "accuracy")].is_primary is True
    # convenience task-mean is present but NOT primary
    assert abs(m[("faitheval", "mean", "accuracy")].value - 0.5) < 1e-9
    assert m[("faitheval", "mean", "accuracy")].is_primary is False


def test_halueval_accuracy(tmp_path):
    fixtures.write_halueval(tmp_path / "halueval", {"qa": 0.7, "dialogue": 0.5})
    recs = parse_run_dir(tmp_path, "arm", 1)
    m = _by_key(recs)
    assert m[("halueval", "qa", "accuracy")].value == 0.7
    assert m[("halueval", "qa", "accuracy")].is_primary is True


def test_ragtruth_lower_is_better(tmp_path):
    fixtures.write_ragtruth(tmp_path / "ragtruth", 0.3,
                            per_task_rate={"QA": 0.2, "Summary": 0.4},
                            gold_f1={"precision": 0.8, "recall": 0.6, "f1": 0.69})
    recs = parse_run_dir(tmp_path, "arm", 1)
    m = _by_key(recs)
    overall = m[("ragtruth", "overall", "hallucination_rate")]
    assert overall.value == 0.3
    assert overall.higher_is_better is False       # the one inverted metric
    assert overall.is_primary is True
    assert overall.signed_value == -0.3            # sign-flipped for cross-metric deltas
    assert m[("ragtruth", "QA", "hallucination_rate")].is_primary is False
    assert m[("ragtruth", "overall", "gold_f1")].higher_is_better is True


def test_truthfulqa_csv_branch(tmp_path):
    fixtures.write_truthfulqa(tmp_path / "truthfulqa",
                              {"MC1": 0.31, "MC2": 0.52, "bleu acc": 0.4})
    recs = parse_run_dir(tmp_path, "arm", 1)
    m = _by_key(recs)
    assert m[("truthfulqa", "overall", "MC2")].value == 0.52
    assert m[("truthfulqa", "overall", "MC1")].is_primary is True
    assert m[("truthfulqa", "overall", "bleu acc")].is_primary is False
    assert all(r.higher_is_better for r in recs if r.benchmark == "truthfulqa")


def test_harness_rows_and_string_values(tmp_path):
    fixtures.write_harness(tmp_path / "harness", [
        {"task": "truthfulqa_mc1", "metric": "acc", "value": 0.28, "stderr": 0.01,
         "higher_is_better": True},
        {"task": "truthfulqa_mc2", "metric": "acc", "value": 0.5, "higher_is_better": True},
        {"task": "some_task", "metric": "acc", "value": 0.9, "higher_is_better": False},
    ])
    recs = parse_run_dir(tmp_path, "arm", 1)
    m = _by_key(recs)
    # values arrive as strings on disk; parser coerces to float
    assert m[("harness", "truthfulqa_mc1", "acc")].value == 0.28
    assert m[("harness", "truthfulqa_mc1", "acc")].stderr == 0.01
    assert m[("harness", "truthfulqa_mc1", "acc")].is_primary is True
    assert m[("harness", "some_task", "acc")].is_primary is False
    # per-row direction is authoritative
    assert m[("harness", "some_task", "acc")].higher_is_better is False


def test_harness_filter_disambiguation(tmp_path):
    fixtures.write_harness(tmp_path / "harness", [
        {"task": "t", "metric": "acc", "filter": "none", "value": 0.5},
        {"task": "t", "metric": "acc", "filter": "strict", "value": 0.4},
    ])
    recs = parse_run_dir(tmp_path, "arm", 1)
    metrics = {r.metric for r in recs}
    assert "acc" in metrics and "acc::strict" in metrics


def test_missing_benchmark_dir_warns_and_skips(tmp_path):
    fixtures.write_faitheval(tmp_path / "faitheval", {"counterfactual": 0.6})
    # no other benchmark dirs -> parse_run_dir simply returns faitheval records
    recs = parse_run_dir(tmp_path, "arm", 1)
    assert {r.benchmark for r in recs} == {"faitheval"}


def test_corrupt_summary_warns(tmp_path):
    (tmp_path / "ragtruth").mkdir()
    (tmp_path / "ragtruth" / "summary.json").write_text("{not json", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recs = parse_run_dir(tmp_path, "arm", 1)
    assert recs == []
    assert any("ragtruth" in str(w.message) for w in caught)


def test_benchmark_selection(tmp_path):
    fixtures.write_full_run(tmp_path / "run", seed=1,
                            faitheval={"counterfactual": 0.6},
                            truthfulqa={"MC2": 0.5})
    recs = parse_run_dir(tmp_path / "run", "arm", 1, benchmarks=["faitheval"])
    assert {r.benchmark for r in recs} == {"faitheval"}
