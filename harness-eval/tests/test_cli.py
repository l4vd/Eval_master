"""CLI parsing / wiring tests. No lm_eval, no downloads."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness_eval.cli import build_config, parse_args, resolve_device
from harness_eval.evaluator import run_evaluation

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_parse_args_defaults_apply_chat_template_false():
    args = parse_args(["--model-id", "gpt2", "--tasks", "truthfulqa"])
    assert args.apply_chat_template is False  # published-protocol default


def test_fewshot_as_multiturn_without_chat_template_errors():
    # Reinstates at our layer the ValueError lm_eval dropped after 0.4.5.
    with pytest.raises(SystemExit):
        parse_args(["--model-id", "gpt2", "--fewshot-as-multiturn"])


def test_list_tasks_mode_does_not_require_model_id():
    args = parse_args(["--list-tasks", "truthfulqa"])
    assert args.list_tasks == "truthfulqa"
    assert args.model_id is None


def test_model_id_required_without_list_tasks():
    with pytest.raises(SystemExit):
        parse_args(["--tasks", "truthfulqa"])


def test_device_index_minus_one_maps_to_cpu():
    assert resolve_device("-1") == "cpu"


def test_device_index_zero_maps_to_cuda0():
    assert resolve_device("0") == "cuda:0"


def test_device_string_passes_through():
    assert resolve_device("mps") == "mps"


def test_build_config_maps_device_and_tasks_tuple():
    args = parse_args(["--model-id", "gpt2", "--tasks", "truthfulqa", "truthfulqa-multi", "--device", "0"])
    config = build_config(args)
    assert config.device == "cuda:0"
    assert config.tasks == ("truthfulqa", "truthfulqa-multi")


def test_help_does_not_import_lm_eval():
    # Importing cli and parsing args must not drag in lm_eval (torch + datasets).
    # Runs in a subprocess so it is unaffected by other tests importing lm_eval.
    code = (
        "import sys; from harness_eval import cli; "
        "cli.parse_args(['--model-id', 'gpt2', '--tasks', 'truthfulqa']); "
        "assert 'lm_eval' not in sys.modules, "
        "sorted(m for m in sys.modules if 'lm_eval' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": SRC},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_evaluation_passes_expected_kwargs(tmp_path):
    # The only test needing a fake; the evaluate_fn seam lets it run with lm_eval
    # absent. Asserts the behaviour-bearing kwargs are passed EXPLICITLY.
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {
            "results": {"truthfulqa_mc1": {"alias": "truthfulqa_mc1", "acc,none": 0.25}},
            "samples": {"truthfulqa_mc1": [{"doc_id": 0}]},
            "lm_eval_version": "0.4.12",
        }

    args = parse_args(
        ["--model-id", "Qwen/Qwen2.5-0.5B-Instruct", "--tasks", "truthfulqa_mc1",
         "--limit", "5", "--log-samples", "--output-dir", str(tmp_path)]
    )
    summary = run_evaluation(build_config(args), evaluate_fn=fake_evaluate)

    assert captured["model"] == "hf"
    assert captured["model_args"]["pretrained"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert captured["tasks"] == ["truthfulqa_mc1"]
    assert captured["limit"] == 5
    # Explicit, not left to an lm_eval default (whose value drifted across versions):
    assert "apply_chat_template" in captured and captured["apply_chat_template"] is False
    assert "fewshot_as_multiturn" in captured and captured["fewshot_as_multiturn"] is False
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "samples.jsonl").is_file()
    assert summary["results"][0]["value"] == 0.25
