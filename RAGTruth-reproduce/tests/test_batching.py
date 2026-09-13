"""Batched Stage 1 / Stage 2 and the shared detection job.

Stubs pin the bookkeeping (execution order, dataset-order artifacts, completion markers,
one detector load, per-directory seeding); a tiny random Llama pins that batched greedy
generation reproduces batch 1 exactly.
"""

from __future__ import annotations

import json
import logging
import random

import pytest
from ragtruth_eval import cli, detect, generate
from ragtruth_eval.model import GenerationParams, generation_order


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def dataset(tmp_path):
    """Sources 0-3 are test (prompt lengths 2, 5, 8, 11 words); source 4 is train."""
    ds = tmp_path / "dataset"
    _write_jsonl(ds / "source_info.jsonl", [
        {"source_id": str(i), "task_type": "Summary", "source": "CNN/DM", "source_info": "doc " * (3 * i + 1),
         "prompt": "Summarize: " + "doc " * (3 * i + 1)} for i in range(5)])
    _write_jsonl(ds / "response.jsonl", [
        {"id": str(i), "source_id": str(i), "split": "train" if i == 4 else "test", "quality": "good", "labels": []}
        for i in range(5)])
    return ds


class _Stub:
    """Tokenizes one id per word; each output names its prompt length and a random draw."""

    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1
        self.calls: list[list[int]] = []

    def chat_ids(self, messages):
        return [1] * len(messages[-1]["content"].split())

    def completion_ids(self, text):
        return [1] * len(text.split())

    def generate_ids(self, sequences, params):
        self.calls.append([len(s) for s in sequences])
        return [f'{{"hallucination list": []}} len={len(s)} draw={random.random():.8f}' for s in sequences]


def test_generation_order_is_dataset_order_at_batch_one_and_longest_first_otherwise():
    assert generation_order([3, 9, 1, 9], 1) == [0, 1, 2, 3]
    assert generation_order([3, 9, 1, 9], 2) == [1, 3, 0, 2]


@pytest.mark.parametrize("batch_size,calls", [(2, [[11, 8], [5, 2]]), (1, [[2], [5], [8], [11]])])
def test_generation_batches_and_writes_dataset_order_with_a_completion_marker(tmp_path, dataset, monkeypatch,
                                                                              batch_size, calls):
    stub = _Stub()
    monkeypatch.setattr(generate, "HFGenerator", lambda *a, **k: stub)
    out = tmp_path / "run"
    generate.run_generation(dataset, out, model_id="m", split="test", batch_size=batch_size,
                            gen_params=GenerationParams(max_new_tokens=7))

    assert stub.calls == calls
    rows = _jsonl(out / "generations.jsonl")
    assert [r["source_id"] for r in rows] == ["0", "1", "2", "3"]
    assert set(rows[0]) == {"source_id", "task_type", "prompt", "reference", "question", "response"}
    assert "len=2 " in rows[0]["response"]
    summary = json.loads((out / "generation_summary.json").read_text(encoding="utf-8"))
    assert (summary["split"], summary["batch_size"], summary["num_generations"]) == ("test", batch_size, 4)
    assert summary["decoding"]["max_new_tokens"] == 7 and summary["elapsed_s"] >= 0
    assert not (out / "generations.jsonl.partial").exists()


def test_cli_warns_when_the_whole_release_is_generated(tmp_path, monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(generate, "run_generation", lambda **kw: calls.append(kw))
    with caplog.at_level(logging.WARNING):
        cli.main(["--stage", "generate", "--model-id", "m", "--output-dir", str(tmp_path)])
    assert "2,965" in caplog.text and calls[0]["split"] is None
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        cli.main(["--stage", "generate", "--model-id", "m", "--split", "test", "--batch-size", "4",
                  "--output-dir", str(tmp_path)])
    assert "2,965" not in caplog.text and calls[1]["batch_size"] == 4


def test_cli_keeps_output_dirs_to_the_detect_stage():
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "all", "--output-dirs", "x"])


def _generated(run_dir, n=3):
    d = run_dir / "ragtruth"
    _write_jsonl(d / "generations.jsonl", [
        {"source_id": str(i), "task_type": "Summary", "prompt": "p", "reference": "ref " * (i + 1),
         "question": "", "response": "resp " * (2 * i + 1)} for i in range(n)])
    (d / "generation_summary.json").write_text(json.dumps(
        {"split": "test", "model_id": "m", "batch_size": 2, "decoding": {}, "elapsed_s": 1.5, "num_generations": n}),
        encoding="utf-8")
    return d


def test_detection_job_loads_the_detector_once_and_skips_finished_dirs(tmp_path, monkeypatch, caplog):
    _Stub.instances = 0
    monkeypatch.setattr(detect, "HFGenerator", _Stub)
    root = tmp_path / "eval"
    first, second = _generated(root / "arm" / "seed_1"), _generated(root / "arm" / "seed_2")
    finished = _generated(root / "arm" / "seed_3")
    (finished / "summary.json").write_text('{"done": true}', encoding="utf-8")
    (root / "arm" / "seed_4" / "ragtruth").mkdir(parents=True)  # generate never ran
    pattern = str(root / "arm" / "seed_*" / "ragtruth")

    with caplog.at_level(logging.INFO):
        results = detect.run_detection_job([pattern], detector_model_id="det", batch_size=2, seed=42)

    assert _Stub.instances == 1 and "loaded once for 2 run directories" in caplog.text
    assert [d for d, _ in results] == [first, second]
    assert json.loads((finished / "summary.json").read_text(encoding="utf-8")) == {"done": True}
    assert not (root / "arm" / "seed_4" / "ragtruth" / "summary.json").exists()

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert (summary["split"], summary["detector_seed"], summary["detector_batch_size"]) == ("test", 42, 2)
    assert summary["rate"]["total"] == 3 and summary["elapsed_s"] >= 1.5
    assert summary["detector_decoding"]["temperature"] == 0.05
    assert [r["source_id"] for r in _jsonl(first / "detections.jsonl")] == ["0", "1", "2"]
    # Re-seeded per directory: the same generations get the same draws in both.
    assert (first / "detections.jsonl").read_text(encoding="utf-8") == \
        (second / "detections.jsonl").read_text(encoding="utf-8")

    _Stub.instances = 0
    assert detect.run_detection_job([pattern], detector_model_id="det") == []  # resubmitting is a no-op
    assert _Stub.instances == 0


@pytest.mark.slow
def test_batched_greedy_generation_matches_batch_one_on_a_tiny_model(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from ragtruth_eval.model import HFGenerator
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["[PAD]", "[UNK]", "[EOS]", "Answer", ":", *[f"w{i}" for i in range(40)]]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                     pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = ("{% for m in messages %}{{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}Answer : {% endif %}")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                      max_position_embeddings=512, pad_token_id=0, bos_token_id=None,
                                      eos_token_id=2, initializer_range=0.5)
    transformers.LlamaForCausalLM(config).save_pretrained(tmp_path / "tiny")
    tokenizer.save_pretrained(tmp_path / "tiny")

    generator = HFGenerator(model_id=str(tmp_path / "tiny"), device_map="cpu", dtype="float32")
    prompts = [[{"role": "user", "content": " ".join(f"w{j % 40}" for j in range(k))}] for k in (3, 17, 9, 25, 6)]
    params = GenerationParams(max_new_tokens=12)
    single = [generator.chat(messages, params) for messages in prompts]

    sequences = [generator.chat_ids(messages) for messages in prompts]
    order = generation_order([len(s) for s in sequences], 3)
    batched = [""] * len(prompts)
    for start in range(0, len(order), 3):
        indices = order[start : start + 3]
        for i, text in zip(indices, generator.generate_ids([sequences[i] for i in indices], params)):
            batched[i] = text
    assert batched == single
    assert any(single)  # guards the equality: the model produced text
