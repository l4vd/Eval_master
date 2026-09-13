"""Eval-side decontamination: row sets, their scores, and where the summary may be written.

The excluded rows are removed from a *copy* of the scoring; the original results and summary
are never touched, and the offline CLI refuses to write the variant next to them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def _load(module_name, filename):
    sys.path.insert(0, str(_EVALUATION_DIR))
    spec = importlib.util.spec_from_file_location(module_name, _EVALUATION_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_results = _load("halueval_score_results_decontam", "score_results.py")


def _list(tmp_path, excluded, unseen, n_rows, task="qa"):
    path = tmp_path / "halueval__qa.json"
    path.write_text(json.dumps({
        "schema_version": 1, "train": "ragtruth", "benchmark": "halueval", "task": task,
        "policy": "exact/near-dup with train/val exposure", "report_sha256": "abc", "n_rows": n_rows,
        "indices": excluded, "unseen_exposed_indices": unseen,
    }), encoding="utf-8")
    return path


def _results(tmp_path, name="qa_label_results.json", n=10):
    """Rows 0-5 judged correctly, rows 6-9 unparseable."""
    path = tmp_path / "run" / "halueval" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            truth = "Yes" if i % 2 else "No"
            f.write(json.dumps({"index": i, "ground_truth": truth,
                                "judgement": truth if i < 6 else "failed!",
                                "raw_judgement": truth if i < 6 else "maybe"}) + "\n")
    return path


def test_row_sets_partition_the_rows_and_score_as_computed_by_hand(tmp_path):
    list_path = _list(tmp_path, excluded=[0, 1, 2], unseen=[3], n_rows=10)
    summary = score_results.decontam_summary(_results(tmp_path), score_results.load_exclusion_list(list_path))

    sets = summary["row_sets"]
    assert [sets[k]["num_examples"] for k in score_results.ROW_SETS] == [10, 7, 3, 6, 1]
    assert (summary["num_examples"], summary["n_excluded"], summary["n_unseen_exposed"]) == (7, 3, 1)
    assert summary["complete"] is True and summary["variant"] == "decontam" and summary["scoring"] == "generate"
    # seen {0,1,2}: all correct; clean {4..9}: 4,5 correct; decontaminated {3..9}: 3,4,5 correct
    assert sets["seen"]["strict"]["accuracy"] == 1.0
    assert sets["clean"]["strict"]["accuracy"] == pytest.approx(2 / 6)
    assert summary["accuracy"] == pytest.approx(3 / 7)
    assert summary["accuracy_se"] == pytest.approx(math.sqrt((3 / 7) * (4 / 7) / 7))
    contrast = summary["contrasts"]["seen_minus_clean"]["strict"]
    assert contrast["diff"] == pytest.approx(1 - 2 / 6)
    assert contrast["se"] == pytest.approx(math.sqrt((2 / 6) * (4 / 6) / 6))
    assert summary["exclusion_list_sha256"] == hashlib.sha256(list_path.read_bytes()).hexdigest()


def test_provenance_is_copied_from_the_runs_own_summary(tmp_path):
    results = _results(tmp_path)
    results.with_name("qa_label_summary.json").write_text(
        json.dumps({"model": "the-label", "seed": 42, "batch_size": 8, "accuracy": 0.6}), encoding="utf-8")
    summary = score_results.decontam_summary(
        results, score_results.load_exclusion_list(_list(tmp_path, [0], [], 10)))
    assert (summary["model"], summary["seed"], summary["batch_size"]) == ("the-label", 42, 8)
    assert score_results.decontam_summary_name(results) == "qa_label_decontam_summary.json"


def test_constrained_results_get_a_constrained_decontam_summary(tmp_path):
    path = tmp_path / "qa_label_constrained_results.json"
    with path.open("w", encoding="utf-8") as f:
        for i in range(8):
            truth = "Yes" if i % 2 else "No"
            yes = math.log(0.8) if truth == "Yes" else math.log(0.1)
            f.write(json.dumps({"index": i, "ground_truth": truth, "logp_yes": yes, "logp_no": math.log(0.15)}) + "\n")
    summary = score_results.decontam_summary(path, score_results.load_exclusion_list(_list(tmp_path, [0, 1], [], 8)))
    assert summary["scoring"] == "constrained" and summary["auroc"] == 1.0 and summary["model"] == "label"
    assert summary["row_sets"]["seen"]["constrained"]["auroc"] == 1.0
    assert score_results.decontam_summary_name(path) == "qa_label_constrained_decontam_summary.json"


def test_an_incomplete_run_and_a_wrong_task_are_flagged(tmp_path):
    summary = score_results.decontam_summary(
        _results(tmp_path), score_results.load_exclusion_list(_list(tmp_path, [0], [], 10_000)))
    assert summary["complete"] is False
    with pytest.raises(ValueError, match="task"):
        score_results.decontam_summary(
            _results(tmp_path), score_results.load_exclusion_list(_list(tmp_path, [0], [], 10, task="summarization")))


def test_cli_writes_the_summary_elsewhere_and_refuses_the_results_dir(tmp_path, monkeypatch):
    results = _results(tmp_path)
    list_path = _list(tmp_path, [0, 1, 2], [3], 10)
    out = tmp_path / "modified" / "run" / "halueval"

    monkeypatch.setattr(sys, "argv", ["score_results.py", str(results), "--exclude", str(list_path),
                                      "--emit-summary", str(out)])
    score_results.main()
    written = json.loads((out / "qa_label_decontam_summary.json").read_text(encoding="utf-8"))
    assert written["num_examples"] == 7

    monkeypatch.setattr(sys, "argv", ["score_results.py", str(results), "--exclude", str(list_path),
                                      "--emit-summary", str(results.parent)])
    with pytest.raises(SystemExit, match="refusing"):
        score_results.main()
    assert sorted(p.name for p in results.parent.iterdir()) == ["qa_label_results.json"]


def test_cli_skips_results_of_another_task(tmp_path, monkeypatch, capsys):
    other = _results(tmp_path, name="dialogue_label_results.json")
    monkeypatch.setattr(sys, "argv", ["score_results.py", str(other), "--exclude", str(_list(tmp_path, [0], [], 10)),
                                      "--emit-summary", str(tmp_path / "out")])
    score_results.main()
    assert "skip dialogue_label_results.json" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
