"""The published metric must survive the instrumentation added around it.

`evaluate.py` gained a failure count, per-class recall, raw-output capture and an
optional label seed. None of that is allowed to move `accuracy`, because that number is
what makes this fork comparable to the HaluEval paper. These tests pin the boundary:
the scoring branch behaves exactly as upstream's, and everything new is derived from it.
"""

from __future__ import annotations

import importlib.util
import json
import random
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


evaluate = _load("halueval_evaluate", "evaluate.py")
score_results = _load("halueval_score_results", "score_results.py")


# --- The upstream scoring rule ---------------------------------------------------

# Verdict text -> what the published rule makes of it. The awkward ones are deliberate:
# upstream matches CAPITALISED substrings without word boundaries, so "Note"/"November"
# collide with a "Yes" verdict, and a lowercase answer parses as nothing at all. These
# are upstream's semantics, reproduced on purpose — not bugs to fix here.
_STRICT_CASES = [
    ("No", "No"),
    ("Yes", "Yes"),
    ("Yes.", "Yes"),                                     # the "." strip
    ("**No**", "No"),
    ("No, the answer is factually correct", "No"),
    ("Yes, the answer contains fabricated details", "Yes"),
    ("yes", None),                                       # case-sensitive
    ("NO", None),
    ("Yes. Note that the lengths differ", None),         # "Note" contains "No"
    ("Yes, the claim about November 1994 is false", None),
    ("I cannot determine this", None),
    ("", None),
]


@pytest.mark.parametrize("text,expected", _STRICT_CASES)
def test_strict_parser_matches_upstream_semantics(text, expected):
    assert score_results.parse_strict(text) == expected


@pytest.mark.parametrize("text,expected", _STRICT_CASES)
def test_evaluate_scoring_branch_agrees_with_the_standalone_parser(text, expected):
    """`score_results.parse_strict` is a copy of evaluate.py's inline branch.

    Two copies of one rule is the risk this test exists for: re-scoring a stored run
    offline is only meaningful while the offline parser is the live one.
    """
    ans = text.replace(".", "")
    if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
        inline = None
    else:
        inline = "Yes" if "Yes" in ans else "No"
    assert inline == expected


def test_lenient_parser_recovers_what_the_strict_one_discards():
    assert score_results.parse_lenient("yes") == "Yes"
    assert score_results.parse_lenient("NO") == "No"
    assert score_results.parse_lenient("Yes. Note that the lengths differ") == "Yes"
    # Word boundaries: "Not"/"November" must not read as a "No" verdict.
    assert score_results.parse_lenient("Not hallucinated at all") is None
    assert score_results.parse_lenient("I cannot determine this") is None


# --- The tally ------------------------------------------------------------------

def test_failed_rows_count_as_incorrect_and_stay_in_the_denominator():
    """Upstream's arithmetic. Excluding failures would inflate accuracy and break
    comparability with the published numbers."""
    tally = evaluate._Tally()
    tally.record("Yes", "Yes")
    tally.record("No", "Yes")
    tally.record_failure("No")

    stats = tally.stats(3)
    assert stats["num_correct"] == 1
    assert stats["num_incorrect"] == 2
    assert stats["accuracy"] == pytest.approx(1 / 3)
    assert stats["num_failed"] == 1
    assert stats["format_compliance"] == pytest.approx(2 / 3)


def test_accuracy_matches_the_pre_instrumentation_formula():
    """Property check against `correct / n`, the expression this replaced."""
    rng = random.Random(0)
    tally = evaluate._Tally()
    correct = 0
    n = 200
    for _ in range(n):
        ground_truth = rng.choice(["Yes", "No"])
        outcome = rng.choice(["Yes", "No", "failed"])
        if outcome == "failed":
            tally.record_failure(ground_truth)
        else:
            tally.record(ground_truth, outcome)
            correct += outcome == ground_truth

    assert tally.stats(n)["accuracy"] == pytest.approx(correct / n)


def test_constant_judge_is_exposed_by_tpr_tnr_but_not_by_accuracy():
    """THE reason the diagnostics exist: a judge answering "No" to everything scores
    the base rate and looks mediocre rather than broken."""
    tally = evaluate._Tally()
    for _ in range(50):
        tally.record("Yes", "No")
    for _ in range(50):
        tally.record("No", "No")

    stats = tally.stats(100)
    assert stats["accuracy"] == pytest.approx(0.5)      # looks unremarkable
    assert stats["format_compliance"] == 1.0            # nothing failed to parse
    assert stats["tpr"] == 0.0                          # ... but caught no hallucination
    assert stats["tnr"] == 1.0
    assert stats["judged_yes_rate"] == 0.0


def test_stats_are_safe_on_an_empty_run():
    stats = evaluate._Tally().stats(0)
    assert stats["accuracy"] == 0.0
    assert stats["tpr"] == 0.0
    assert stats["format_compliance"] == 0.0


# --- Seeding --------------------------------------------------------------------

class _ConstantJudge:
    batch_size = 2

    def prompt_token_lengths(self, requests):
        return [len(prompt.split()) for _messages, prompt in requests]

    def generate_many(self, requests):
        return ["No"] * len(requests)


def _qa_fixture(path, n=12):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "knowledge": f"k{i}", "question": f"q{i} " + "word " * (i % 5),
                "hallucinated_answer": f"bad {i}", "right_answer": f"good {i}",
            }) + "\n")
    return path


def _run(tmp_path, data, tag, seed):
    out = tmp_path / f"results_{tag}.json"
    stats = evaluate.evaluation_qa_dataset(
        "stub", str(data), "INSTRUCTION", str(out),
        backend="hf", generator=_ConstantJudge(), seed=seed, sort_by_length="desc",
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stats, rows


def test_same_seed_gives_the_same_label_draw(tmp_path):
    data = _qa_fixture(tmp_path / "qa.json")
    stats_a, rows_a = _run(tmp_path, data, "a", seed=42)
    stats_b, rows_b = _run(tmp_path, data, "b", seed=42)

    assert stats_a == stats_b
    assert {r["index"]: r["ground_truth"] for r in rows_a} == \
           {r["index"]: r["ground_truth"] for r in rows_b}


def test_different_seeds_give_a_different_label_draw(tmp_path):
    """Guards the test above: identical labels under different seeds would make it vacuous."""
    data = _qa_fixture(tmp_path / "qa.json", n=40)
    _, rows_a = _run(tmp_path, data, "a", seed=42)
    _, rows_b = _run(tmp_path, data, "c", seed=1337)

    assert {r["index"]: r["ground_truth"] for r in rows_a} != \
           {r["index"]: r["ground_truth"] for r in rows_b}


def test_seed_none_keeps_upstreams_unseeded_behaviour(tmp_path):
    """`--seed` unset must not silently pin the draw — the module-level RNG stays in
    charge, so a caller seeding it themselves still controls the run."""
    data = _qa_fixture(tmp_path / "qa.json")

    random.seed(7)
    _, rows_a = _run(tmp_path, data, "a", seed=None)
    random.seed(7)
    _, rows_b = _run(tmp_path, data, "b", seed=None)

    assert {r["index"]: r["ground_truth"] for r in rows_a} == \
           {r["index"]: r["ground_truth"] for r in rows_b}


def test_raw_judgement_is_stored_for_every_row(tmp_path):
    """Without the raw text a failed row is undiagnosable after the fact — the whole
    point of capturing it."""
    data = _qa_fixture(tmp_path / "qa.json")
    _, rows = _run(tmp_path, data, "a", seed=42)

    assert all("raw_judgement" in row for row in rows)
    assert {row["raw_judgement"] for row in rows} == {"No"}


def test_run_label_is_filename_safe():
    """A Windows `--model-path` must not smuggle a ":" into the artifact name: NTFS
    would read it as an alternate data stream and the run would report success while
    writing nothing. Linux paths must come through unchanged."""
    assert evaluate._run_label("/gpfs/project/models/final_checkpoint") == \
        "_gpfs_project_models_final_checkpoint"
    assert ":" not in evaluate._run_label(r"c:\Users\me\ckpt")
    assert evaluate._run_label(r"c:\Users\me\ckpt") == "c__Users_me_ckpt"


# --- Offline re-scoring ---------------------------------------------------------

def test_score_file_reproduces_the_live_runs_accuracy(tmp_path):
    """The contract that makes offline re-scoring trustworthy."""
    data = _qa_fixture(tmp_path / "qa.json", n=20)
    stats, _ = _run(tmp_path, data, "a", seed=42)

    scored = score_results.score_file(tmp_path / "results_a.json")

    assert scored["strict"]["accuracy"] == pytest.approx(stats["accuracy"])
    assert scored["strict"]["num_failed"] == stats["num_failed"]
    assert scored["strict"]["tpr"] == pytest.approx(stats["tpr"])
    assert scored["strict"]["tnr"] == pytest.approx(stats["tnr"])


def test_score_file_sizes_the_parser_gap(tmp_path):
    """A run whose judge answers in lowercase: every row unparseable under the published
    rule, all of them recoverable under the lenient one. The gap between the two is the
    share of the score that is formatting rather than judgement."""
    path = tmp_path / "lower_results.json"
    with path.open("w", encoding="utf-8") as f:
        for i in range(4):
            ground_truth = "Yes" if i % 2 else "No"
            f.write(json.dumps({
                "index": i, "ground_truth": ground_truth,
                "judgement": "failed!", "raw_judgement": ground_truth.lower(),
            }) + "\n")

    scored = score_results.score_file(path)

    assert scored["strict"]["accuracy"] == 0.0
    assert scored["strict"]["num_failed"] == 4
    assert scored["lenient"]["accuracy"] == 1.0
    assert scored["lenient"]["num_failed"] == 0


def test_score_file_handles_runs_without_raw_judgement(tmp_path):
    """Legacy results (and the shipped gpt-3.5 reference file) must still score strictly."""
    path = tmp_path / "legacy_results.json"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"index": 0, "ground_truth": "Yes", "judgement": "Yes"}) + "\n")
        f.write(json.dumps({"index": 1, "ground_truth": "No", "judgement": "failed!"}) + "\n")
        f.write("\n")  # a blank line must not be scored as a row

    scored = score_results.score_file(path)

    assert scored["has_raw_judgement"] is False
    assert "lenient" not in scored
    assert scored["strict"]["num_examples"] == 2
    assert scored["strict"]["accuracy"] == pytest.approx(0.5)
    assert scored["strict"]["num_failed"] == 1


def test_shipped_gpt35_reference_reproduces_the_published_number_exactly():
    """A canary on the scoring path, free of any model or API key.

    The ChatGPT run bundled with the repo re-scores to 6,259 correct out of the split's
    10,000 rows = 0.6259, which is *exactly* the 62.59 the HaluEval paper reports for
    ChatGPT on QA. That makes this file a ground-truth fixture for the scoring rule: if
    this test moves, the rule changed and the fork's numbers left the published scale.

    The file also carries 3 empty rows — an artefact of the append-per-row writer, which
    leaves a partial file behind when a run is interrupted. They are unscorable, so the
    scorer skips them and its own denominator is 9,997; the published figure divides by
    the full 10,000 (upstream uses `len(data)`). Both are asserted so the distinction
    stays explicit rather than being rediscovered later as a rounding mystery.
    """
    reference = _EVALUATION_DIR / "qa" / "qa_gpt-3.5-turbo_result.json"
    if not reference.is_file():
        pytest.skip("reference result file not present")

    scored = score_results.score_file(reference)

    assert scored["num_rows"] == 10000
    assert scored["num_skipped"] == 3
    assert scored["strict"]["num_examples"] == 9997
    assert scored["strict"]["num_correct"] == 6259
    assert scored["strict"]["num_correct"] / scored["num_rows"] == pytest.approx(0.6259)
    assert scored["strict"]["num_failed"] == 0
