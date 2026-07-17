"""Command-line entry point for harness-eval.

Usage:
    python src/run_eval.py --model-id Qwen/Qwen2.5-0.5B-Instruct --tasks truthfulqa
    python src/run_eval.py --list-tasks truthfulqa      # discover task/tag names

or, once installed (`pip install -e .`):
    harness-eval --model-id <id> --tasks truthfulqa truthfulqa_multilingual
"""

from __future__ import annotations

import argparse
import logging

from harness_eval.config import EvalConfig


def resolve_device(value: str) -> str:
    """Map a `device_index`-style value to an lm_eval device string.

    The launcher passes `model.device_index` (`-1` = CPU, `0` = cuda:0). An
    explicit device string (e.g. `cuda:1`, `mps`, `cpu`) is passed through
    unchanged, so the flag also works when called directly.
    """
    try:
        index = int(value)
    except (TypeError, ValueError):
        return value
    return "cpu" if index < 0 else f"cuda:{index}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for a harness-eval run."""
    parser = argparse.ArgumentParser(
        description="Evaluate a language model on TruthfulQA via lm-evaluation-harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id, or a local path to a full model / PEFT adapter "
        "checkpoint (e.g. a `final_checkpoint` from this project's training pipeline). "
        "Required unless --list-tasks is given.",
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help="Base model id/path for a PEFT/LoRA adapter --model-id. Defaults to the "
        "base recorded in the adapter's own config.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Tokenizer id/path, if different from the checkpoint's own tokenizer.",
    )
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["truthfulqa"],
        help="lm_eval task/tag names (glob patterns allowed, e.g. 'truthfulqa_de_*').",
    )
    parser.add_argument("--num-fewshot", type=int, default=None, help="Few-shot examples; None = task default.")
    parser.add_argument("--batch-size", default="1", help="Batch size: an int, or 'auto' / 'auto:N'.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N docs PER TASK.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="lm_eval device. Accepts a device_index (-1=CPU, 0=cuda:0) or a string (cuda:1, mps).",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype."
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Render prompts with the model's chat template. Default (off) is the published "
        "completion-style protocol; this MATERIALLY moves TruthfulQA scores.",
    )
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Present few-shot examples as separate chat turns (requires --apply-chat-template).",
    )
    parser.add_argument("--system-instruction", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow checkpoints with custom modeling code."
    )
    parser.add_argument("--log-samples", action="store_true", help="Write per-document records to samples.jsonl.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for summary and sample files.")
    parser.add_argument(
        "--list-tasks",
        nargs="?",
        const="",
        default=None,
        metavar="SUBSTRING",
        help="Print known lm_eval task/tag names (optionally filtered) and exit. No model is loaded.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity."
    )

    args = parser.parse_args(argv)
    if args.list_tasks is None:
        if not args.model_id:
            parser.error("--model-id is required (unless --list-tasks)")
        if args.fewshot_as_multiturn and not args.apply_chat_template:
            parser.error("--fewshot-as-multiturn requires --apply-chat-template")
    return args


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Assemble an :class:`EvalConfig` from parsed CLI arguments."""
    return EvalConfig(
        model_id=args.model_id,
        tasks=tuple(args.tasks),
        base_model_id=args.base_model_id,
        tokenizer_id=args.tokenizer_id,
        cache_dir=args.cache_dir,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        system_instruction=args.system_instruction,
        trust_remote_code=args.trust_remote_code,
        log_samples=args.log_samples,
        output_dir=args.output_dir,
        device=resolve_device(args.device),
        dtype=args.dtype,
    )


def _run_list_tasks(substring: str) -> None:
    """Print known task/tag names matching `substring` (empty = all)."""
    from lm_eval.tasks import TaskManager  # lazy: pulls in the heavy stack

    from harness_eval.tasks import list_tasks

    for name in list_tasks(substring, TaskManager()):
        print(name)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args, then either list tasks or run the evaluation."""
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.list_tasks is not None:
        _run_list_tasks(args.list_tasks)
        return

    config = build_config(args)
    # Imported lazily so `--help`, arg errors, and --list-tasks don't pay lm_eval's
    # (torch + datasets) import cost.
    from harness_eval.evaluator import run_evaluation

    run_evaluation(config)


if __name__ == "__main__":
    main()
