"""Constrained Yes/No scoring: the same labels as generate mode, its own artifacts, a
threshold-free metric — and, with transformers installed, exact prefill log-probabilities.

The prefill must read the same prompt tokens a generate run conditions on, at the right
positions under left padding, from the model's raw logits. The slow tests check exactly
that on a tiny random Llama, so nothing is downloaded.
"""

from __future__ import annotations

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


evaluate = _load("halueval_evaluate_constrained", "evaluate.py")
score_results = _load("halueval_score_results_constrained", "score_results.py")


class _Judge:
    """Says "Yes" to the rows carrying the hallucination marker, with log-probs to match."""

    batch_size = 3

    def prompt_token_lengths(self, requests):
        return [len(prompt.split()) for _messages, prompt in requests]

    def generate_many(self, requests):
        return ["Yes" if "HALLUCINATED" in prompt else "No" for _messages, prompt in requests]

    def verdict_logprobs(self, requests):
        return [(math.log(0.7), math.log(0.2)) if "HALLUCINATED" in prompt else (math.log(0.1), math.log(0.8))
                for _messages, prompt in requests]


def _qa(path, n=13):
    lengths = [7, 1, 40, 3, 22, 11, 2]
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"knowledge": f"k{i}", "question": f"q{i} " + "w " * lengths[i % 7],
                                "hallucinated_answer": f"HALLUCINATED a{i}", "right_answer": f"right a{i}"}) + "\n")
    return path


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(tmp_path, data, scoring, tag):
    out, constrained_out = tmp_path / f"{tag}_results.json", tmp_path / f"{tag}_constrained_results.json"
    result = evaluate.evaluation_qa_dataset(
        "stub", str(data), "INSTRUCTION", str(out), backend="hf", generator=_Judge(), seed=42,
        sort_by_length="desc", scoring=scoring, constrained_output_path=str(constrained_out),
    )
    return result, out, constrained_out


def test_constrained_labels_equal_generate_labels(tmp_path):
    data = _qa(tmp_path / "qa.json")
    _, generate_out, _ = _run(tmp_path, data, "generate", "g")
    _, _, constrained_out = _run(tmp_path, data, "constrained", "c")
    labels = {r["index"]: r["ground_truth"] for r in _rows(generate_out)}
    assert {r["index"]: r["ground_truth"] for r in _rows(constrained_out)} == labels
    assert len(set(labels.values())) == 2  # guards the equality: both labels occur


def test_constrained_never_opens_the_original_results(tmp_path):
    data = _qa(tmp_path / "qa.json")
    result, out, constrained_out = _run(tmp_path, data, "constrained", "c")
    assert not out.exists()
    assert result["generate"] is None and result["constrained"]["num_examples"] == 13
    row = _rows(constrained_out)[0]
    assert set(row) == {"index", "ground_truth", "logp_yes", "logp_no", "verdict_mass", "constrained_judgement"}


def test_both_writes_both_sets_and_leaves_generate_untouched(tmp_path):
    data = _qa(tmp_path / "qa.json")
    generate_stats, generate_out, _ = _run(tmp_path, data, "generate", "g")
    result, both_out, constrained_out = _run(tmp_path, data, "both", "b")
    assert result["generate"] == generate_stats
    assert _rows(both_out) == _rows(generate_out)
    assert len(_rows(constrained_out)) == 13


def test_generate_mode_still_returns_the_plain_stats_dict(tmp_path):
    stats, _, constrained_out = _run(tmp_path, _qa(tmp_path / "qa.json"), "generate", "g")
    assert "accuracy" in stats and not constrained_out.exists()


def test_live_constrained_stats_equal_the_offline_rescore(tmp_path):
    result, _, constrained_out = _run(tmp_path, _qa(tmp_path / "qa.json"), "constrained", "c")
    live = result["constrained"]
    assert live == score_results.score_constrained(_rows(constrained_out))
    assert live["auroc"] == 1.0 and live["accuracy_argmax"] == 1.0
    assert live["mean_verdict_mass"] == pytest.approx(0.9)
    assert score_results.score_file(constrained_out)["scoring"] == "constrained"


def test_constrained_scoring_needs_the_local_judge(tmp_path):
    with pytest.raises(ValueError, match="local HF judge"):
        evaluate.evaluation_qa_dataset("gpt", str(_qa(tmp_path / "qa.json")), "I", str(tmp_path / "r.json"),
                                       backend="openai", scoring="constrained",
                                       constrained_output_path=str(tmp_path / "c.json"))


# --- the metric --------------------------------------------------------------------

def test_auroc_on_perfect_inverted_tied_and_constant_scores():
    labels = [0, 0, 1, 1]
    assert score_results.auroc([0.1, 0.2, 0.8, 0.9], labels) == 1.0
    assert score_results.auroc([0.9, 0.8, 0.2, 0.1], labels) == 0.0
    assert score_results.auroc([0.5, 0.5, 0.5, 0.5], labels) == 0.5
    assert score_results.auroc([0.1, 0.4, 0.35, 0.8], labels) == 0.75  # the textbook example
    assert score_results.auroc([1.0, 0.0, 1.0, 0.0], [1, 0, 0, 1]) == 0.5  # ties at average rank
    assert score_results.auroc([0.3, 0.7], [1, 1]) is None  # one class only


def test_hanley_mcneil_standard_error():
    assert score_results.auroc_se(1.0, 50, 50) == 0.0
    q1, q2 = 0.75 / 1.25, 2 * 0.5625 / 1.75
    expected = math.sqrt((0.75 * 0.25 + (q1 - 0.5625) + (q2 - 0.5625)) / 4)
    assert score_results.auroc_se(0.75, 2, 2) == pytest.approx(expected)
    assert score_results.auroc_se(None, 2, 2) is None


def test_argmax_ties_read_as_no_and_mass_below_half_is_counted():
    rows = [{"ground_truth": "Yes", "logp_yes": math.log(0.2), "logp_no": math.log(0.2)},  # mass 0.4
            {"ground_truth": "No", "logp_yes": math.log(0.05), "logp_no": math.log(0.9)}]
    stats = score_results.score_constrained(rows)
    assert stats["judged_yes_rate"] == 0.0 and stats["tpr"] == 0.0 and stats["tnr"] == 1.0
    assert stats["frac_mass_below_half"] == 0.5


# --- the verdict tokens ----------------------------------------------------------------

class _Vocab:
    """A tokenizer stub: (raw vocabulary token, decoded text) pairs, ids in list order."""

    all_special_ids = [0]

    def __init__(self, entries):
        self._entries = entries

    def get_vocab(self):
        return {token: i for i, (token, _text) in enumerate(self._entries)}

    def decode(self, ids):
        return "".join(self._entries[i][1] for i in ids)


def test_verdict_tokens_pool_each_id_once_over_every_spelling():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    hf_local = _load("halueval_hf_local_verdicts", "hf_local.py")
    # "▁Yes" decodes to the same text as "Yes" (SentencePiece), yet is its own token.
    vocab = _Vocab([("<no>", "no"), ("Yes", "Yes"), ("ĠYes", " Yes"), ("▁Yes", "Yes"), ("YES", "YES"),
                    ("ĠNO", " NO"), ("no", "no"), ("Ċno", "\nno"), ("no.", "no."), ("Yesterday", "Yesterday"),
                    ("ĠNobody", " Nobody"), ("yES", "yES"), ("ĠĠYes", "  Yes")])
    ids, tokens = hf_local._resolve_verdict_tokens(vocab)
    assert ids == {"yes": [1, 2, 3, 4], "no": [5, 6]}
    assert tokens == {"yes": ["Yes", "ĠYes", "▁Yes", "YES"], "no": ["ĠNO", "no"]}


# --- the real prefill, on a tiny random model -----------------------------------------

_CHAT_TEMPLATE = ("{% for m in messages %}{{ m['content'] }} {% endfor %}"
                  "{% if add_generation_prompt %}Answer : {% endif %}")


def _tiny_judge(tmp_path, *, chat):
    path, torch = _tiny_model_dir(tmp_path, chat=chat)
    hf_local = _load("halueval_hf_local_constrained", "hf_local.py")
    return hf_local.HFChatGenerator(model_id=str(path), device_map="cpu", dtype="float32", batch_size=3), torch


def _tiny_model_dir(tmp_path, *, chat):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["[PAD]", "[UNK]", "[EOS]", "Yes", "No", "yes", "no", "Answer", "Question", "Judgement",
             "#", ":", *[f"w{i}" for i in range(40)]]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    if chat:
        tokenizer.chat_template = _CHAT_TEMPLATE
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
        pad_token_id=0, bos_token_id=None, eos_token_id=2, initializer_range=0.5)
    path = tmp_path / ("chat" if chat else "concat")
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path, torch


def _requests():
    return [evaluate.build_qa_request("w1 " * k + "w2", "w3 w4", "Question") for k in (1, 9, 4)]


@pytest.mark.slow
@pytest.mark.parametrize("chat", [True, False])
def test_prefill_logprobs_are_identical_at_batch_one_and_three(tmp_path, chat):
    judge, torch = _tiny_judge(tmp_path, chat=chat)
    requests = _requests()
    assert len({len(ids) for ids in judge.prompt_token_ids(requests)}) == 3  # ragged, so padding matters
    batched = judge.last_token_logprobs(requests)
    single = torch.cat([judge.last_token_logprobs([r]) for r in requests])
    assert torch.allclose(batched, single, atol=1e-5)
    assert judge.verdict_tokens == {"yes": ["Yes", "yes"], "no": ["No", "no"]}
    for logp_yes, logp_no in judge.verdict_logprobs(requests):
        assert logp_yes < 0 and logp_no < 0


def _run_cli(tmp_path, model, *extra):
    import subprocess

    out = tmp_path / "run" / "halueval"
    cmd = [sys.executable, "evaluate.py", "--task", "qa", "--model-path", str(model), "--device-map", "cpu",
           "--dtype", "float32", "--num-samples", "4", "--seed", "1", "--max-new-tokens", "2",
           "--output-dir", str(out), *extra]
    proc = subprocess.run(cmd, cwd=str(_EVALUATION_DIR), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out, evaluate._run_label(str(model))


@pytest.mark.slow
def test_constrained_output_dir_keeps_the_variant_out_of_the_original_dir(tmp_path):
    model, _ = _tiny_model_dir(tmp_path, chat=True)
    constrained_dir = tmp_path / "run" / "halueval.constrained"
    out, label = _run_cli(tmp_path, model, "--scoring", "both", "--constrained-output-dir", str(constrained_dir))
    assert sorted(p.name for p in out.iterdir()) == [f"qa_{label}_results.json", f"qa_{label}_summary.json"]
    assert sorted(p.name for p in constrained_dir.iterdir()) == [
        f"qa_{label}_constrained_results.json", f"qa_{label}_constrained_summary.json"]
    summary = json.loads((constrained_dir / f"qa_{label}_constrained_summary.json").read_text(encoding="utf-8"))
    assert summary["scoring"] == "constrained" and summary["num_examples"] == 4


@pytest.mark.slow
def test_without_constrained_output_dir_both_sets_share_the_output_dir(tmp_path):
    model, _ = _tiny_model_dir(tmp_path, chat=True)
    out, label = _run_cli(tmp_path, model, "--scoring", "both")
    assert {p.name for p in out.iterdir()} == {
        f"qa_{label}_results.json", f"qa_{label}_summary.json",
        f"qa_{label}_constrained_results.json", f"qa_{label}_constrained_summary.json"}


@pytest.mark.slow
def test_argmax_matches_greedy_first_token_without_repetition_penalty(tmp_path):
    judge, torch = _tiny_judge(tmp_path, chat=True)
    model = judge._generator.model
    for request in _requests():
        ids = torch.tensor([judge.prompt_token_ids([request])[0]])
        greedy = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False,
                                repetition_penalty=1.0, pad_token_id=0)
        assert greedy[0, -1].item() == judge.last_token_logprobs([request])[0].argmax().item()
