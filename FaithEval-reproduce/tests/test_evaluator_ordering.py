"""Length-sorted batching must change only WHEN an example is generated, never the result.

Reordering the dataset is a throughput optimization, so the property that matters is that
it is invisible in the artifacts: identical accuracy, identical per-example scoring, and a
predictions file that comes back in dataset order regardless of the execution order. These
tests pin exactly that, with a stub generator so no model is loaded.
"""

from __future__ import annotations

import json

import pytest
from faitheval import evaluator
from faitheval.config import PHRASE_MATCH, EvalConfig, TaskConfig
from faitheval.model import _chat_template_ids


def test_chat_template_ids_normalizes_every_stack_shape():
    """`apply_chat_template(tokenize=True)` returns a plain id list on transformers 4.41
    (the HPC pin) but a BatchEncoding on 5.x.

    Worth pinning because the failure is invisible on one stack: `len(BatchEncoding)` is the
    number of KEYS (2), so every prompt would report the same length and the sorter would
    silently degrade to a no-op locally while working correctly on the cluster.
    """
    ids = [11, 12, 13, 14, 15]

    assert _chat_template_ids(ids) == ids  # 4.41: flat list
    assert _chat_template_ids({"input_ids": ids, "attention_mask": [1] * 5}) == ids  # 5.x
    assert _chat_template_ids({"input_ids": [ids]}) == ids  # batch-wrapped
    assert _chat_template_ids([]) == []


class _StubTokenizer:
    """Whitespace tokenizer, so prompt "length" is predictable from the fixture text."""

    def __call__(self, text, add_special_tokens=False):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [list(range(len(t.split()))) for t in texts]
        return {"input_ids": ids[0] if isinstance(text, str) else ids}


class _StubGenerator:
    """Records the order prompts arrived in, and answers deterministically per prompt.

    The answer depends only on the prompt, never on batch position — so any difference in
    the scored results between two orderings is a real ordering bug, not decoding noise.
    """

    def __init__(self, *args, **kwargs):
        self.tokenizer = _StubTokenizer()
        self.seen: list[str] = []

    @staticmethod
    def _text(messages):
        return " ".join(m["content"] for m in messages)

    def prompt_token_lengths(self, batch):
        return [len(self._text(m).split()) for m in batch]

    def generate_many(self, batch, params):
        out = []
        for messages in batch:
            text = self._text(messages)
            self.seen.append(text)
            # "unknown" for every third question, so both correct and incorrect rows exist.
            out.append("unknown" if "q2" in text or "q5" in text else "some answer")
        return out


def _config(tmp_path, sort_by_length, num_samples=None):
    task_config = TaskConfig(
        dataset_name="stub", scoring=PHRASE_MATCH, valid_phrases=["unknown"], strict_valid_phrases=["unknown"]
    )
    return EvalConfig(
        task="unanswerable",
        task_config=task_config,
        model_id="stub",
        num_samples=num_samples,
        max_new_tokens=8,
        output_dir=str(tmp_path),
        batch_size=3,
        sort_by_length=sort_by_length,
    )


# Ragged on purpose: context length is unrelated to index, so file order and length order
# are genuinely different permutations.
_EXAMPLES = [
    {"context": "word " * 7, "question": "q0", "answer": ["a0"]},
    {"context": "word " * 1, "question": "q1", "answer": ["a1"]},
    {"context": "word " * 40, "question": "q2", "answer": ["a2"]},
    {"context": "word " * 3, "question": "q3", "answer": ["a3"]},
    {"context": "word " * 22, "question": "q4", "answer": ["a4"]},
    {"context": "word " * 11, "question": "q5", "answer": ["a5"]},
    {"context": "word " * 2, "question": "q6", "answer": ["a6"]},
]


@pytest.fixture
def stub_run(monkeypatch):
    """Run `run_evaluation` against the stub generator and fixture dataset."""
    generators: list[_StubGenerator] = []

    def _make(*args, **kwargs):
        generators.append(_StubGenerator())
        return generators[-1]

    monkeypatch.setattr(evaluator, "HFChatGenerator", _make)
    monkeypatch.setattr(evaluator, "load_task_dataset", lambda tc, split, n: list(_EXAMPLES)[: n or len(_EXAMPLES)])

    def run(tmp_path, sort_by_length, num_samples=None):
        summary = evaluator.run_evaluation(_config(tmp_path, sort_by_length, num_samples))
        records = [
            json.loads(line)
            for line in (tmp_path / "unanswerable_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return summary, records, generators[-1]

    return run


def test_batch_order_sorts_by_length_and_is_stable():
    lengths = [7, 1, 40, 3]
    assert evaluator.batch_order(lengths, "none") == [0, 1, 2, 3]
    assert evaluator.batch_order(lengths, "asc") == [1, 3, 0, 2]
    assert evaluator.batch_order(lengths, "desc") == [2, 0, 3, 1]
    # Ties keep dataset order, so a uniform-length split is the identity permutation.
    assert evaluator.batch_order([5, 5, 5], "desc") == [0, 1, 2]


def test_sorting_changes_execution_order(tmp_path, stub_run):
    """Guards the test itself: if the generator saw the same order either way, the
    equivalence assertions below would be vacuous."""
    _, _, gen_none = stub_run(tmp_path / "a", "none")
    _, _, gen_desc = stub_run(tmp_path / "b", "desc")

    assert gen_none.seen != gen_desc.seen
    assert sorted(gen_none.seen) == sorted(gen_desc.seen)  # same prompts, different order
    # Longest first: each batch is internally homogeneous, which is the whole point.
    lengths = [len(t.split()) for t in gen_desc.seen]
    assert lengths == sorted(lengths, reverse=True)


@pytest.mark.parametrize("sort_by_length", ["none", "asc", "desc"])
def test_results_are_identical_under_every_ordering(tmp_path, stub_run, sort_by_length):
    baseline_summary, baseline_records, _ = stub_run(tmp_path / "baseline", "none")
    summary, records, _ = stub_run(tmp_path / sort_by_length, sort_by_length)

    for key in ("accuracy", "num_correct", "num_examples", "mean_prediction_tokens", "truncation_rate"):
        assert summary[key] == baseline_summary[key]
    assert records == baseline_records


def test_predictions_file_is_written_in_dataset_order(tmp_path, stub_run):
    """The artifact must not depend on the execution order, or a length-sorted run cannot
    be diffed row-for-row against one made without sorting."""
    _, records, _ = stub_run(tmp_path, "desc")

    assert [r["index"] for r in records] == list(range(len(_EXAMPLES)))
    assert [r["question"] for r in records] == [e["question"] for e in _EXAMPLES]


def test_num_samples_still_takes_the_first_n_in_dataset_order(tmp_path, stub_run):
    """Sorting happens AFTER the sub-sample, so `--num-samples 4` evaluates the same four
    examples it always did — otherwise a smoke test would silently change its own split."""
    _, records, _ = stub_run(tmp_path, "desc", num_samples=4)

    assert [r["index"] for r in records] == [0, 1, 2, 3]


def test_summary_records_the_ordering_and_token_metrics(tmp_path, stub_run):
    summary, _, _ = stub_run(tmp_path, "desc")

    assert summary["sort_by_length"] == "desc"
    # "unknown" is 1 token, "some answer" is 2, under the whitespace stub tokenizer.
    assert summary["mean_prediction_tokens"] == pytest.approx((2 * 1 + 5 * 2) / 7)
    assert summary["mean_prediction_words"] == pytest.approx((2 * 1 + 5 * 2) / 7)
    # Nothing reached max_new_tokens=8.
    assert summary["truncation_rate"] == 0.0


def test_truncation_rate_flags_predictions_that_hit_the_budget(tmp_path, monkeypatch):
    """The degeneration probe this metric exists for: a model that never stops."""

    class _Runaway(_StubGenerator):
        def generate_many(self, batch, params):
            return ["tok " * 20] * len(batch)

    monkeypatch.setattr(evaluator, "HFChatGenerator", lambda *a, **k: _Runaway())
    monkeypatch.setattr(evaluator, "load_task_dataset", lambda tc, split, n: list(_EXAMPLES))

    summary = evaluator.run_evaluation(_config(tmp_path, "desc"))

    assert summary["truncation_rate"] == 1.0
    assert summary["mean_prediction_tokens"] == 20.0
