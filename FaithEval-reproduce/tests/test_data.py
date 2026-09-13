"""Tests for local (JSONL) FaithEval dataset loading."""

import json

import pytest
from faitheval import data


def test_dataset_slug_takes_last_path_component():
    assert data.dataset_slug("Salesforce/FaithEval-unanswerable-v1.0") == "FaithEval-unanswerable-v1.0"
    assert data.dataset_slug("FaithEval-inconsistent-v1.0") == "FaithEval-inconsistent-v1.0"
    assert data.dataset_slug("org/name/") == "name"


def test_local_split_path_layout(tmp_path):
    path = data.local_split_path("Salesforce/FaithEval-unanswerable-v1.0", "test", tmp_path)
    assert path == tmp_path / "FaithEval-unanswerable-v1.0" / "test.jsonl"


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_load_split_reads_local_jsonl(tmp_path):
    records = [
        {"context": "c1", "question": "q1", "answer": ["a1"]},
        {"context": "c2", "question": "q2", "answer": ["a2"]},
    ]
    _write_jsonl(tmp_path / "FaithEval-unanswerable-v1.0" / "test.jsonl", records)

    dataset = data._load_split("Salesforce/FaithEval-unanswerable-v1.0", "test", tmp_path)
    assert len(dataset) == 2
    assert dataset[0]["question"] == "q1"
    assert dataset[1]["answer"] == ["a2"]


def test_load_split_skips_blank_lines(tmp_path):
    path = tmp_path / "FaithEval-inconsistent-v1.0" / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"question": "q1"}) + "\n\n" + json.dumps({"question": "q2"}) + "\n",
        encoding="utf-8",
    )
    dataset = data._load_split("Salesforce/FaithEval-inconsistent-v1.0", "test", tmp_path)
    assert len(dataset) == 2


def test_load_split_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_datasets"):
        data._load_split("Salesforce/FaithEval-unanswerable-v1.0", "test", tmp_path)


def test_load_split_empty_file_raises(tmp_path):
    path = tmp_path / "FaithEval-unanswerable-v1.0" / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        data._load_split("Salesforce/FaithEval-unanswerable-v1.0", "test", tmp_path)


# --- strict-match configuration ---------------------------------------------------

def test_strict_match_without_strict_phrases_is_rejected():
    """Otherwise every example scores wrong and the run reports accuracy 0.0 as a result."""
    import pytest
    from faitheval.config import EvalConfig, TaskConfig

    task = TaskConfig(
        dataset_name="x", scoring="phrase_match", valid_phrases=["unknown"],
        strict_valid_phrases=[],
    )
    with pytest.raises(ValueError, match="strict_valid_phrases"):
        EvalConfig(task="unanswerable", task_config=task, model_id="m", strict_match=True)

    # ...and the same config is fine without --strict-match.
    cfg = EvalConfig(task="unanswerable", task_config=task, model_id="m")
    assert cfg.active_valid_phrases == ["unknown"]


def test_shipped_phrase_match_configs_support_strict_match():
    """Every shipped phrase_match task must survive --strict-match (run_benchmarks passes it).

    Note what this can and cannot assert. Reaching the body at all already proves the
    ``EvalConfig`` guard passed, and that guard is exactly ``strict_match and
    scoring == PHRASE_MATCH and not strict_valid_phrases`` -- so a bare
    ``assert cfg.active_valid_phrases`` restates the check that just ran and can never
    fire. The content assertions below are the part that can: that ``active_valid_phrases``
    actually switches to the *strict* list rather than falling through to the permissive
    one, which is the behaviour --strict-match is bought for.
    """
    from pathlib import Path

    from faitheval.config import PHRASE_MATCH, EvalConfig, load_task_config

    configs = Path(__file__).resolve().parents[1] / "configs"
    yamls = sorted(configs.glob("*.yaml"))
    # Anti-vacuity: a moved configs/ dir, a renamed suffix or a scoring-value change would
    # otherwise empty the loop and leave this test passing while checking nothing.
    assert {p.stem for p in yamls} == {"counterfactual", "counterfactual_mc", "inconsistent", "unanswerable"}

    checked = []
    for path in yamls:
        task = load_task_config(path)
        if task.scoring != PHRASE_MATCH:
            continue
        checked.append(path.stem)
        cfg = EvalConfig(task=path.stem, task_config=task, model_id="m", strict_match=True)
        # The flag must select the strict list, not silently keep the permissive one.
        assert cfg.active_valid_phrases == task.strict_valid_phrases, path.name
        # ...and strict must actually be stricter, or the flag is decorative on this task.
        assert set(task.strict_valid_phrases) < set(task.valid_phrases), path.name

        without = EvalConfig(task=path.stem, task_config=task, model_id="m")
        assert without.active_valid_phrases == task.valid_phrases, path.name
        assert without.active_valid_phrases != cfg.active_valid_phrases, path.name

    # counterfactual is answer_match, so it is skipped by design -- pinned so that a task
    # dropping out of phrase_match scoring shows up here instead of shrinking the loop.
    assert checked == ["inconsistent", "unanswerable"]
