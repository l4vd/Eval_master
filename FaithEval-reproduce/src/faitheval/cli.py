"""Command-line entry point for FaithEval evaluation.

Usage:
    python src/run_eval.py --task unanswerable --model-id meta-llama/Meta-Llama-3.1-8B-Instruct

or, once installed (`pip install -e .`):
    faitheval-eval --task unanswerable --model-id meta-llama/Meta-Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from faitheval.config import SUPPORTED_TASKS, EvalConfig, load_task_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for a FaithEval evaluation run."""
    parser = argparse.ArgumentParser(
        description="Evaluate a language model's contextual faithfulness on a FaithEval task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS, help="FaithEval task to evaluate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a task config YAML file. Defaults to configs/<task>.yaml.",
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model id, or a local path to a full model / PEFT adapter checkpoint "
        "(e.g. a `final_checkpoint` directory produced by this project's training pipeline).",
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help="Base model id/path to load the adapter onto, if --model-id points at a PEFT/LoRA "
        "adapter checkpoint. Defaults to the base model recorded in the adapter's own config.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Tokenizer id/path, if different from --model-id (e.g. hand-merged weights saved "
        "without their own tokenizer files). Defaults to --model-id.",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory for model/tokenizer files.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--num-samples", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens generated per example.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (requires --do-sample).")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p (requires --do-sample).")
    parser.add_argument(
        "--strict-match",
        action="store_true",
        help="Use each task's strict valid-phrase list instead of the lenient one.",
    )
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt prepended to each example.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for predictions and summary files.")
    parser.add_argument("--device-map", default="auto", help="`device_map` passed to `from_pretrained`.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args(argv)
    if (args.temperature is not None or args.top_p is not None) and not args.do_sample:
        parser.error("--temperature/--top-p require --do-sample")
    return args


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Assemble an `EvalConfig` from parsed CLI arguments."""
    config_path = args.config or DEFAULT_CONFIG_DIR / f"{args.task}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No task config found at {config_path}")

    task_config = load_task_config(config_path)
    return EvalConfig(
        task=args.task,
        task_config=task_config,
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        tokenizer_id=args.tokenizer_id,
        cache_dir=args.cache_dir,
        split=args.split,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        strict_match=args.strict_match,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        device_map=args.device_map,
        dtype=args.dtype,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args, build config, run the evaluation."""
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = build_config(args)

    # Imported lazily so `--help` and config errors don't pay the torch/transformers import cost.
    from faitheval.evaluator import run_evaluation

    run_evaluation(config)


if __name__ == "__main__":
    main()
