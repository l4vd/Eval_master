"""Result-normalisation tests. Pure fixture dicts — no lm_eval, no downloads."""

import json

from harness_eval.results import build_summary, flatten_results, write_outputs


class _Unserializable:
    """Stands in for a numpy scalar: str() works, but json can't serialise it."""

    def __str__(self):
        return "0.5"


def _results(**overrides):
    base = {
        "results": {
            "truthfulqa_mc1": {
                "alias": "truthfulqa_mc1",
                "acc,none": 0.42,
                "acc_stderr,none": 0.017,
            }
        },
        "versions": {"truthfulqa_mc1": 2.0},
        "n-shot": {"truthfulqa_mc1": 0},
        "higher_is_better": {"truthfulqa_mc1": {"acc": True}},
    }
    base.update(overrides)
    return base


def test_flatten_splits_metric_and_filter_keys():
    (row,) = flatten_results(_results())
    assert row["metric"] == "acc"
    assert row["filter"] == "none"
    assert row["value"] == 0.42


def test_flatten_ignores_non_metric_keys():
    rows = flatten_results(_results())
    assert all(r["metric"] != "alias" for r in rows)
    assert len(rows) == 1


def test_flatten_pairs_stderr_with_its_metric():
    (row,) = flatten_results(_results())
    assert row["stderr"] == 0.017


def test_flatten_sets_stderr_none_when_absent():
    res = {"results": {"t": {"acc,none": 0.5}}}
    (row,) = flatten_results(res)
    assert row["stderr"] is None


def test_flatten_coerces_na_stderr_to_none():
    res = {"results": {"t": {"acc,none": 0.5, "acc_stderr,none": "N/A"}}}
    (row,) = flatten_results(res)
    assert row["stderr"] is None


def test_flatten_attaches_version_and_num_fewshot():
    (row,) = flatten_results(_results())
    assert row["version"] == 2.0
    assert row["num_fewshot"] == 0
    assert row["higher_is_better"] is True


def test_missing_groups_key_is_not_an_error():
    # The TruthfulQA tags don't aggregate, so `groups` is absent.
    rows = flatten_results(_results())
    assert len(rows) == 1


def test_group_subtasks_recorded_when_present():
    res = _results(group_subtasks={"truthfulqa": ["truthfulqa_mc1", "truthfulqa_mc2"]})
    summary = build_summary(res, provenance={})
    assert summary["group_subtasks"]["truthfulqa"] == ["truthfulqa_mc1", "truthfulqa_mc2"]


def test_flatten_uses_task_key_not_indented_alias():
    # Group-member aliases arrive indented (" - truthfulqa_mc1"); we key on the task.
    res = {"results": {"truthfulqa_mc1": {"alias": " - truthfulqa_mc1", "acc,none": 0.5}}}
    (row,) = flatten_results(res)
    assert row["task"] == "truthfulqa_mc1"


def test_summary_records_provenance_fields():
    provenance = {
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "limit": None,
        "num_fewshot": None,
        "resolved_tasks": ["truthfulqa"],
    }
    summary = build_summary(_results(lm_eval_version="0.4.12"), provenance)
    for key in provenance:
        assert key in summary
    assert summary["lm_eval_version"] == "0.4.12"


def test_samples_jsonl_written_one_record_per_doc(tmp_path):
    res = {
        "results": {"t": {"acc,none": 0.5}},
        "samples": {"t": [{"doc_id": 0}, {"doc_id": 1}]},
    }
    write_outputs(res, build_summary(res, {}), tmp_path, log_samples=True)
    lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["task"] == "t"


def test_samples_jsonl_preserves_non_ascii(tmp_path):
    res = {"results": {}, "samples": {"truthfulqa_ar_mc1": [{"question": "ما هي عاصمة فرنسا؟"}]}}
    write_outputs(res, build_summary(res, {}), tmp_path, log_samples=True)
    raw = (tmp_path / "samples.jsonl").read_text(encoding="utf-8")
    assert "عاصمة" in raw  # literal, not \uXXXX escapes


def test_summary_serializes_numpy_like_scalars(tmp_path):
    # lm_eval results carry numpy scalars; write_outputs must pass default=str.
    res = {"results": {"t": {"acc,none": _Unserializable()}}, "samples": {}}
    write_outputs(res, build_summary(res, {}), tmp_path, log_samples=False)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["results"][0]["value"] == "0.5"


def test_lm_eval_results_dump_excludes_samples(tmp_path):
    res = {"results": {"t": {"acc,none": 0.5}}, "samples": {"t": [{"doc_id": 0}]}}
    write_outputs(res, build_summary(res, {}), tmp_path, log_samples=True)
    raw = json.loads((tmp_path / "lm_eval_results.json").read_text(encoding="utf-8"))
    assert "samples" not in raw
