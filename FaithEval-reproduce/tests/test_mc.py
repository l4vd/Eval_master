"""counterfactual_mc: the task config, option scoring from log-likelihoods, dataset-order
predictions, gold from answerKey — and, with transformers, the log-likelihoods themselves."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from faitheval import evaluator
from faitheval.config import (
    CHOICE_LOGLIK,
    SUPPORTED_TASKS,
    EvalConfig,
    TaskConfig,
    load_task_config,
)

REPO = Path(__file__).resolve().parents[1]


def test_mc_config_is_counterfactual_with_option_scoring():
    mc = load_task_config(REPO / "configs" / "counterfactual_mc.yaml")
    cf = load_task_config(REPO / "configs" / "counterfactual.yaml")
    assert mc.scoring == CHOICE_LOGLIK and "counterfactual_mc" in SUPPORTED_TASKS
    assert (mc.dataset_name, mc.context_column, mc.question_column, mc.task_specific_prompt) == \
        (cf.dataset_name, cf.context_column, cf.question_column, cf.task_specific_prompt)
    assert (mc.choices_column, mc.answer_key_column) == ("choices", "answerKey")


def test_choice_loglik_requires_its_columns():
    with pytest.raises(ValueError, match="choices_column"):
        TaskConfig(dataset_name="x", scoring=CHOICE_LOGLIK)
    with pytest.raises(ValueError, match="Unknown scoring"):
        TaskConfig(dataset_name="x", scoring="loglik")


def test_parse_choices_and_resolve_gold():
    assert evaluator.parse_choices('{"label": ["A", "B"], "text": ["x", "y"]}') == (["A", "B"], ["x", "y"])
    assert evaluator.parse_choices({"label": ["A"], "text": ["x"]}) == (["A"], ["x"])
    assert evaluator.resolve_gold(["A", "B"], " B ") == 1
    with pytest.raises(ValueError, match="not one of"):
        evaluator.resolve_gold(["A", "B"], "D")
    with pytest.raises(ValueError, match="malformed"):
        evaluator.parse_choices({"label": ["A"], "text": []})
    assert evaluator.argmax([1.0, 3.0, 3.0]) == 1  # earliest on a tie


def test_resolve_gold_reads_a_letter_key_over_numeric_labels_as_a_position():
    # The NYSEDREGENTS_* rows: labels 1-4, key A-D, confirmed by the answer text.
    labels, texts = ["1", "2", "3", "4"], ["sunlight", "oxygen", "carbon dioxide", "dead organisms"]
    assert evaluator.resolve_gold(labels, "C", texts=texts, answer="carbon dioxide") == 2
    assert evaluator.resolve_gold(labels, "3", texts=texts, answer="carbon dioxide") == 2
    with pytest.raises(ValueError, match="not one of"):  # position disagrees with the answer
        evaluator.resolve_gold(labels, "B", texts=texts, answer="carbon dioxide")
    with pytest.raises(ValueError, match="not one of"):  # out of range
        evaluator.resolve_gold(labels, "E", texts=texts, answer="carbon dioxide")
    with pytest.raises(ValueError, match="not one of"):  # nothing to confirm against
        evaluator.resolve_gold(labels, "C", texts=texts)
    with pytest.raises(ValueError, match="not one of"):  # letter labels never fall back
        evaluator.resolve_gold(["A", "B", "C"], "D", texts=["x", "y", "z"], answer="x")


_LOCAL_SPLIT = REPO / "data" / "faitheval" / "FaithEval-counterfactual-v1.0" / "test.jsonl"


@pytest.mark.skipif(not _LOCAL_SPLIT.exists(), reason="local counterfactual split not present")
def test_every_counterfactual_row_resolves_to_its_answer_text():
    task_config = load_task_config(REPO / "configs" / "counterfactual_mc.yaml")
    rows = [json.loads(line) for line in _LOCAL_SPLIT.read_text(encoding="utf-8").splitlines() if line.strip()]
    numeric = 0
    for row in rows:
        labels, texts = evaluator.parse_choices(row[task_config.choices_column])
        answer = row[task_config.answer_column]
        gold = evaluator.resolve_gold(labels, row[task_config.answer_key_column], texts=texts, answer=answer)
        assert texts[gold].strip() == str(answer).strip(), row["id"]
        numeric += labels[0].isdigit()
    assert len(rows) == 1000 and numeric == 19


# (logprob, n_tokens, n_chars) per option text, keyed by question.
_TABLE = {
    # summed: A wins (wrong); per character: B wins (right)
    "q0": {"short": (-2.0, 1, 5), "a much longer option": (-3.0, 4, 20)},
    # both pick A (right)
    "q1": {"alpha": (-1.0, 1, 5), "bravo": (-5.0, 1, 5)},
    # a tie between A and B resolves to A (wrong either way)
    "q2": {"one": (-4.0, 2, 8), "two": (-4.0, 2, 8), "three": (-9.0, 3, 12)},
}
_EXAMPLES = [
    {"context": "word " * 7, "question": "q0", "answerKey": "B",
     "choices": json.dumps({"label": ["A", "B"], "text": ["short", "a much longer option"]})},
    {"context": "word " * 30, "question": "q1", "answerKey": "A",
     "choices": json.dumps({"label": ["A", "B"], "text": ["alpha", "bravo"]})},
    {"context": "word " * 2, "question": "q2", "answerKey": "C",
     "choices": {"label": ["A", "B", "C"], "text": ["one", "two", "three"]}},
]


class _StubScorer:
    prompt_format = "chat_template"

    def __init__(self):
        self.seen: list[str] = []

    def prompt_token_lengths(self, batch):
        return [len(" ".join(m["content"] for m in messages).split()) for messages in batch]

    def choice_logprobs(self, messages_batch, choices_batch):
        out = []
        for messages, choices in zip(messages_batch, choices_batch):
            question = next(line for line in messages[-1]["content"].splitlines()
                            if line.startswith("Question: "))[len("Question: "):]
            self.seen.append(question)
            out.append([_TABLE[question][text] for text in choices])
        return out


@pytest.fixture
def mc_run(monkeypatch, tmp_path):
    stub = _StubScorer()
    monkeypatch.setattr(evaluator, "HFChatGenerator", lambda *a, **k: stub)
    monkeypatch.setattr(evaluator, "load_task_dataset", lambda tc, split, n: list(_EXAMPLES))
    config = EvalConfig(task="counterfactual_mc", task_config=load_task_config(REPO / "configs" / "counterfactual_mc.yaml"),
                        model_id="stub", output_dir=str(tmp_path), batch_size=2, sort_by_length="desc")
    summary = evaluator.run_evaluation(config)
    lines = (tmp_path / "counterfactual_mc_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return summary, [json.loads(line) for line in lines if line.strip()], stub, tmp_path


def test_accuracy_and_accuracy_norm_from_option_loglikelihoods(mc_run):
    summary, records, _, tmp_path = mc_run
    assert summary["scoring"] == CHOICE_LOGLIK and summary["num_examples"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3) and summary["accuracy_norm"] == pytest.approx(2 / 3)
    assert [r["predicted"] for r in records] == ["A", "A", "A"]
    assert [r["predicted_norm"] for r in records] == ["B", "A", "A"]
    assert json.loads((tmp_path / "counterfactual_mc_summary.json").read_text(encoding="utf-8")) == summary


def test_predictions_are_in_dataset_order_with_gold_from_the_answer_key(mc_run):
    _, records, stub, _ = mc_run
    assert [r["index"] for r in records] == [0, 1, 2]
    assert [r["gold"] for r in records] == ["B", "A", "C"]
    assert stub.seen == ["q1", "q0", "q2"]  # executed longest prompt first
    assert [c["label"] for c in records[2]["choice_scores"]] == ["A", "B", "C"]


_CHAT_TEMPLATE = ("{% for m in messages %}{{ m['content'] }} {% endfor %}"
                  "{% if add_generation_prompt %}Answer : {% endif %}")


@pytest.mark.slow
def test_choice_logprobs_equal_a_manual_forward_at_any_batch_size(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from faitheval.model import HFChatGenerator
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["[PAD]", "[UNK]", "[EOS]", "Answer", ":", *[f"w{i}" for i in range(40)]]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                     pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = _CHAT_TEMPLATE
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                      max_position_embeddings=512, pad_token_id=0, bos_token_id=None,
                                      eos_token_id=2, initializer_range=0.5)
    transformers.LlamaForCausalLM(config).save_pretrained(tmp_path / "tiny")
    tokenizer.save_pretrained(tmp_path / "tiny")

    messages = [[{"role": "user", "content": "w1 w2 w3 " * k}] for k in (1, 5)]
    choices = [["w4 w5", "w6"], ["w7", "w8 w9 w10", "w11"]]
    scored = {}
    for batch_size in (1, 4):
        generator = HFChatGenerator(model_id=str(tmp_path / "tiny"), device_map="cpu", dtype="float32",
                                    batch_size=batch_size)
        scored[batch_size] = generator.choice_logprobs(messages, choices)

    model = generator._generator.model
    for i, (msgs, options) in enumerate(zip(messages, choices)):
        context = generator.prompt_token_ids([msgs])[0]
        for j, text in enumerate(options):
            continuation = generator.tokenizer(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                logits = model(torch.tensor([context + continuation])).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            expected = sum(logprobs[len(context) - 1 + k, token].item() for k, token in enumerate(continuation))
            for batch_size in (1, 4):
                total, n_tokens, n_chars = scored[batch_size][i][j]
                assert total == pytest.approx(expected, abs=1e-4)
                assert (n_tokens, n_chars) == (len(continuation), len(text))
