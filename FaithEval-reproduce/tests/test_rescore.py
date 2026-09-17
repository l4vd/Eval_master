"""Offline re-scoring (`faitheval.rescore`): rules, the lenient reproduction, and no heavy imports.

Runs without torch/datasets, so it also passes under the Eval_master launcher venv.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from faitheval import rescore
from faitheval.metrics import phrase_match

REPO = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


UNANSWERABLE = ["Unknown.", "I cannot tell from the context.", "Paris is the capital.",
                "There is no information about that.", "The answer is not given.", "It is notable."]


def _unanswerable_run(tmp_path, *, correct=None):
    phrases = ["unknown", "no answer", "no information", "not", "unclear"]
    rows = [{"index": i, "question": f"q{i}", "prediction": p,
             "correct": phrase_match(p, phrases) if correct is None else correct[i]}
            for i, p in enumerate(UNANSWERABLE)]
    path = _write_jsonl(tmp_path / "run" / "faitheval" / "unanswerable_predictions.jsonl", rows)
    (path.parent / "unanswerable_summary.json").write_text(
        json.dumps({"task": "unanswerable", "batch_size": 8, "max_new_tokens": 256, "accuracy": 0.0}),
        encoding="utf-8")
    return path


def test_cannot_matches_lenient_but_not_wordmatch():
    phrases = ["not"]
    assert phrase_match("I cannot tell.", phrases)
    assert not rescore.word_match("I cannot tell.", phrases)
    assert rescore.word_match("It is not stated.", phrases)
    assert rescore.word_match("There is no information here.", ["no information"])


def test_strict_only_matches_unknown(tmp_path):
    summary = rescore.rescore(_unanswerable_run(tmp_path), "strict")
    assert summary["num_correct"] == 1 and summary["phrases"] == ["unknown"]
    assert summary["accuracy"] == pytest.approx(1 / 6)


def test_wordmatch_drops_the_substring_hits(tmp_path):
    path = _unanswerable_run(tmp_path)
    lenient = rescore.rescore(path, "lenient")
    wordmatch = rescore.rescore(path, "wordmatch")
    # lenient: unknown, cannot, no information, not, notable -> 5; wordmatch: unknown, no information, not -> 3
    assert (lenient["num_correct"], wordmatch["num_correct"]) == (5, 3)


def test_lenient_reproduces_the_stored_correct_and_flags_a_mismatch(tmp_path):
    assert rescore.rescore(_unanswerable_run(tmp_path), "lenient")["num_mismatched"] == 0
    tampered = _unanswerable_run(tmp_path / "t", correct=[True] * 6)
    assert rescore.rescore(tampered, "lenient")["num_mismatched"] == 1
    assert rescore.main([str(tampered), "--rule", "lenient"]) == 1


def test_summary_carries_source_hash_provenance_and_length(tmp_path):
    path = _unanswerable_run(tmp_path)
    summary = rescore.rescore(path, "wordmatch")
    assert summary["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (summary["task"], summary["variant"], summary["scoring"]) == ("unanswerable", "wordmatch", "generate")
    assert (summary["batch_size"], summary["max_new_tokens"]) == (8, 256)
    assert summary["mean_prediction_words"] == pytest.approx(sum(len(p.split()) for p in UNANSWERABLE) / 6)


def _counterfactual_run(tmp_path):
    data = tmp_path / "data"
    gold = ["Planetary gravity will become stronger.", "blue", "Mount Everest"]
    _write_jsonl(data / "FaithEval-counterfactual-v1.0" / "test.jsonl",
                 [{"question": f"q{i}", "answer": a} for i, a in enumerate(gold)])
    preds = ["Planetary gravity will become stronger.",            # exact, 5 words
             "Based on the context, the sky is blue " + "x " * 20,  # contained, long
             "The tallest is K2."]                                   # wrong, 4 words
    rows = [{"index": i, "question": f"q{i}", "prediction": p, "correct": i == 0} for i, p in enumerate(preds)]
    return _write_jsonl(tmp_path / "run" / "faitheval" / "counterfactual_predictions.jsonl", rows), data


def test_contains_is_length_stratified_with_exact_match_alongside(tmp_path):
    path, data = _counterfactual_run(tmp_path)
    summary = rescore.rescore(path, "contains", data_dir=data)
    assert summary["accuracy"] == pytest.approx(2 / 3) and summary["exact_match"] == pytest.approx(1 / 3)
    strata = summary["length_strata"]
    assert strata["1_5"] == {"n": 2, "accuracy": 0.5, "exact_match": 0.5}
    assert strata["21_60"] == {"n": 1, "accuracy": 1.0, "exact_match": 0.0}
    assert strata["61_plus"]["n"] == 0 and strata["61_plus"]["accuracy"] is None
    assert rescore.rescore(path, "lenient", data_dir=data)["num_mismatched"] == 0


def test_contains_needs_word_boundaries():
    assert not rescore.contains_match("The bluebird sang.", ["blue"])
    assert rescore.contains_match("It was blue.", ["Blue"])


def test_rules_refuse_the_wrong_task_kind(tmp_path):
    path, data = _counterfactual_run(tmp_path)
    with pytest.raises(ValueError, match="phrase_match"):
        rescore.rescore(path, "strict", data_dir=data)
    with pytest.raises(ValueError, match="answer_match"):
        rescore.rescore(_unanswerable_run(tmp_path), "contains")


def test_cli_writes_the_variant_dir_and_refuses_the_original(tmp_path):
    path = _unanswerable_run(tmp_path)
    out = tmp_path / "run" / "faitheval.strict"
    assert rescore.main([str(path), "--rule", "strict", "--output-dir", str(out)]) == 0
    assert json.loads((out / "unanswerable_summary.json").read_text(encoding="utf-8"))["variant"] == "strict"
    with pytest.raises(SystemExit, match="refusing"):
        rescore.main([str(path), "--rule", "strict", "--output-dir", str(path.parent)])
    with pytest.raises(SystemExit):
        rescore.main([str(path), "--rule", "lenient", "--output-dir", str(out)])


def test_importing_it_loads_neither_torch_nor_datasets():
    code = ("import sys; import faitheval.rescore; "
            "bad = [m for m in ('torch', 'datasets', 'transformers') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO / "src"), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_entry_point_runs_as_a_script(tmp_path):
    path = _unanswerable_run(tmp_path)
    proc = subprocess.run([sys.executable, "src/rescore.py", str(path), "--rule", "lenient"],
                          cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "reproduces every stored" in proc.stdout
