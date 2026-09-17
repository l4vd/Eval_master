"""`score_results.py --emit-variant`: the descriptive re-scorings of the optional overview."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


score_results = _load("halueval_score_results_emit", "score_results.py")


def _results(tmp_path, *, raw=True):
    """10 rows: 0-5 parsed (4 correct), 6-7 strict-unparseable but lenient-readable, 8-9 unreadable."""
    path = tmp_path / "run" / "halueval" / "qa_label_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(10):
        truth = "Yes" if i % 2 else "No"
        if i < 6:
            verdict = truth if i < 4 else ("No" if truth == "Yes" else "Yes")
            row = {"index": i, "ground_truth": truth, "judgement": verdict, "raw_judgement": verdict}
        elif i < 8:
            row = {"index": i, "ground_truth": truth, "judgement": "failed!",
                   "raw_judgement": f"{truth.lower()}, Nothing else"}
        else:
            row = {"index": i, "ground_truth": truth, "judgement": "failed!", "raw_judgement": "maybe"}
        if not raw:
            row.pop("raw_judgement")
        rows.append(row)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    path.with_name("qa_label_summary.json").write_text(
        json.dumps({"model": "label", "seed": 42, "batch_size": 8, "accuracy": 0.4}), encoding="utf-8")
    return path


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["score_results.py", *map(str, argv)])
    return score_results.main()


def test_parsed_accuracy_times_compliance_is_the_strict_accuracy(tmp_path):
    path = _results(tmp_path)
    parsed = score_results.variant_summary(path, "parsed")
    strict = score_results.score_file(path)["strict"]
    assert parsed["accuracy"] == pytest.approx(4 / 6)
    assert parsed["accuracy"] * parsed["format_compliance"] == pytest.approx(strict["accuracy"])
    assert (parsed["num_examples"], parsed["num_scored"], parsed["num_failed"]) == (10, 6, 4)


def test_lenient_recovers_the_word_boundary_verdicts(tmp_path):
    lenient = score_results.variant_summary(_results(tmp_path), "lenient")
    assert lenient["accuracy"] == pytest.approx(6 / 10) and lenient["num_failed"] == 2


def test_summary_carries_source_hash_and_provenance(tmp_path):
    path = _results(tmp_path)
    summary = score_results.variant_summary(path, "parsed")
    assert summary["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (summary["variant"], summary["scoring"], summary["model"], summary["seed"]) == (
        "parsed", "generate", "label", 42)
    assert summary["source_results"] == path.name


def test_cli_writes_into_the_variant_dir(tmp_path, monkeypatch):
    path = _results(tmp_path)
    out = tmp_path / "run" / "halueval.parsed"
    _main(monkeypatch, path, "--emit-variant", "parsed", "--emit-summary", out)
    assert json.loads((out / "qa_label_parsed_summary.json").read_text(encoding="utf-8"))["variant"] == "parsed"


def test_lenient_without_raw_judgement_exits_3_and_writes_nothing(tmp_path, monkeypatch):
    path = _results(tmp_path, raw=False)
    out = tmp_path / "run" / "halueval.lenient"
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, path, "--emit-variant", "lenient", "--emit-summary", out)
    assert exc.value.code == score_results.EXIT_UNAVAILABLE == 3
    assert not out.exists()
    # ...while parsed still works on the same legacy file
    assert score_results.variant_summary(path, "parsed")["accuracy"] == pytest.approx(4 / 6)


def test_cli_refuses_the_results_dir_and_bad_combinations(tmp_path, monkeypatch):
    path = _results(tmp_path)
    with pytest.raises(SystemExit, match="refusing"):
        _main(monkeypatch, path, "--emit-variant", "parsed", "--emit-summary", path.parent)
    assert sorted(p.name for p in path.parent.iterdir()) == ["qa_label_results.json", "qa_label_summary.json"]
    with pytest.raises(SystemExit):
        _main(monkeypatch, path, "--emit-variant", "parsed")
    with pytest.raises(SystemExit):
        _main(monkeypatch, path, "--emit-summary", tmp_path / "x")


def test_constrained_files_are_not_re_scored(tmp_path):
    path = tmp_path / "qa_label_constrained_results.json"
    path.write_text(json.dumps({"index": 0, "ground_truth": "Yes", "logp_yes": -1, "logp_no": -2}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="constrained"):
        score_results.variant_summary(path, "parsed")


def test_decontam_summary_records_the_source_hash(tmp_path):
    path = _results(tmp_path)
    listing = tmp_path / "halueval__qa.json"
    listing.write_text(json.dumps({"task": "qa", "n_rows": 10, "indices": [0]}), encoding="utf-8")
    summary = score_results.decontam_summary(path, score_results.load_exclusion_list(listing))
    assert summary["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
