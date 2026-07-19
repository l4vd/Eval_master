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
