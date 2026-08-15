"""Length-sorted batching must not disturb HaluEval's Yes/No ground-truth assignment.

This is the sharp edge of reordering here. Each row's label is decided by a `random.random()`
draw, and the batching loop used to make that draw inside the per-chunk loop — so the draws
were consumed in iteration order. Judging rows in length order under that design would have
handed each example a DIFFERENT coin flip, silently changing which answer the judge is shown
and therefore the benchmark's own labels, not just its throughput.

`evaluation/evaluate.py` now draws every ground truth up front in ascending index order and
reorders only execution. These tests pin that: same labels, same accuracy, different order.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

_EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"


def _load_evaluate():
    """Import evaluation/evaluate.py by path — the repo is scripts run in place, not a package."""
    sys.path.insert(0, str(_EVALUATION_DIR))
    spec = importlib.util.spec_from_file_location("halueval_evaluate", _EVALUATION_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = _load_evaluate()


class _StubGenerator:
    """A judge that answers from the prompt alone, and records the order it saw prompts in.

    Answering deterministically per prompt is what makes an ordering difference in the
    results attributable to the ordering rather than to decoding noise.
    """

    batch_size = 3

    def __init__(self):
        self.seen: list[str] = []

    def prompt_token_lengths(self, requests):
        return [len(prompt.split()) for _messages, prompt in requests]

    def generate_many(self, requests):
        out = []
        for _messages, prompt in requests:
            self.seen.append(prompt)
            # Judge "Yes" iff the answer text carries the marker the fixture puts on
            # hallucinated rows — a perfect judge, so accuracy is a pure function of labels.
            out.append("Yes" if "HALLUCINATED" in prompt else "No")
        return out


def _write_qa_fixture(path, n=11):
    """Ragged prompt lengths, uncorrelated with index, so length order != file order.

    The padding goes in the QUESTION, not in `knowledge`: `build_qa_request` composes the
    judge prompt from the instruction, the question and the answer only, so knowledge never
    reaches the tokenizer and padding it would leave every prompt the same length.
    """
    lengths = [7, 1, 40, 3, 22, 11, 2, 31, 5, 17, 9]
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "knowledge": "k%d" % i,
                        "question": "q%d " % i + "word " * lengths[i % len(lengths)],
                        "hallucinated_answer": "HALLUCINATED a%d" % i,
                        "right_answer": "correct a%d" % i,
                    }
                )
                + "\n"
            )
    return path


def _run(tmp_path, data_path, sort_by_length, seed=0):
    random.seed(seed)
    generator = _StubGenerator()
    out = tmp_path / f"results_{sort_by_length}.json"
    stats = evaluate.evaluation_qa_dataset(
        "stub", str(data_path), "INSTRUCTION", str(out),
        backend="hf", generator=generator, sort_by_length=sort_by_length,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stats, rows, generator


@pytest.fixture
def qa_data(tmp_path):
    return _write_qa_fixture(tmp_path / "qa_data.json")


def test_sorting_changes_execution_order(qa_data, tmp_path):
    """Guards the tests below: if the judge saw the same order either way, the equivalence
    assertions would be vacuous."""
    _, _, gen_none = _run(tmp_path, qa_data, "none")
    _, _, gen_desc = _run(tmp_path, qa_data, "desc")

    assert gen_none.seen != gen_desc.seen
    assert sorted(gen_none.seen) == sorted(gen_desc.seen)
    lengths = [len(p.split()) for p in gen_desc.seen]
    assert lengths == sorted(lengths, reverse=True)  # longest batch first


@pytest.mark.parametrize("sort_by_length", ["none", "asc", "desc"])
def test_ground_truth_per_index_is_independent_of_ordering(qa_data, tmp_path, sort_by_length):
    """THE assertion this module exists for.

    Under a fixed seed, example i must be shown the same answer no matter what order the
    rows are judged in. If the draws ever move back inside the batching loop, this fails.
    """
    _, baseline_rows, _ = _run(tmp_path, qa_data, "none")
    _, rows, _ = _run(tmp_path, qa_data, sort_by_length)

    baseline_truth = {r["index"]: r["ground_truth"] for r in baseline_rows}
    truth = {r["index"]: r["ground_truth"] for r in rows}

    assert truth == baseline_truth
    # ... and the answer text shown alongside it, not just the label.
    assert {r["index"]: r["answer"] for r in rows} == {r["index"]: r["answer"] for r in baseline_rows}


@pytest.mark.parametrize("sort_by_length", ["none", "asc", "desc"])
def test_accuracy_is_unchanged_by_ordering(qa_data, tmp_path, sort_by_length):
    baseline_stats, _, _ = _run(tmp_path, qa_data, "none")
    stats, _, _ = _run(tmp_path, qa_data, sort_by_length)

    assert stats == baseline_stats


def test_every_row_is_written_with_its_index(qa_data, tmp_path):
    """Rows are appended in execution order, so `index` is the only way back to the
    dataset — the per-sample file is unusable for row-level comparison without it."""
    _, rows, _ = _run(tmp_path, qa_data, "desc")

    assert sorted(r["index"] for r in rows) == list(range(11))


def test_num_samples_still_takes_the_first_n_in_dataset_order(qa_data, tmp_path):
    """Sorting happens after the sub-sample, so a smoke test keeps evaluating the same rows."""
    random.seed(0)
    out = tmp_path / "sub.json"
    evaluate.evaluation_qa_dataset(
        "stub", str(qa_data), "INSTRUCTION", str(out),
        backend="hf", generator=_StubGenerator(), num_samples=4, sort_by_length="desc",
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert sorted(r["index"] for r in rows) == [0, 1, 2, 3]


def test_chat_template_ids_normalizes_every_stack_shape():
    """`apply_chat_template(tokenize=True)` returns a plain id list on transformers 4.41
    (the HPC pin) but a BatchEncoding on 5.x.

    Worth pinning because the failure is invisible on one stack: `len(BatchEncoding)` is the
    number of KEYS (2), so every prompt would report the same length and the sorter would
    silently degrade to a no-op locally while working correctly on the cluster.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("halueval_hf_local", _EVALUATION_DIR / "hf_local.py")
    hf_local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hf_local)

    ids = [11, 12, 13, 14, 15]
    assert hf_local._chat_template_ids(ids) == ids  # 4.41: flat list
    assert hf_local._chat_template_ids({"input_ids": ids, "attention_mask": [1] * 5}) == ids  # 5.x
    assert hf_local._chat_template_ids({"input_ids": [ids]}) == ids  # batch-wrapped
    assert hf_local._chat_template_ids([]) == []


def test_openai_backend_is_never_reordered(qa_data, tmp_path):
    """The OpenAI path is serial and unbatched, so reordering buys nothing and would only
    scramble the progress output. _batch_order must return the identity there."""
    pending = [(i, "req %d" % i) for i in range(5)]

    assert evaluate._batch_order(pending, "openai", None, "desc") == [0, 1, 2, 3, 4]
    assert evaluate._batch_order(pending, "hf", None, "none") == [0, 1, 2, 3, 4]
