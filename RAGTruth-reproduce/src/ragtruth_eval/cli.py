"""Command-line entry point for the serverless two-stage RAGTruth evaluation.

Usage:
    python src/run_eval.py --stage all \
        --model-id meta-llama/Meta-Llama-3-8B-Instruct \
        --detector-model-id CodingLL/RAGTruth_Eval

or, once installed (`pip install -e .`):
    ragtruth-eval --stage all --model-id <gen> --detector-model-id <detector>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ragtruth_eval.data import TASK_TYPES
from ragtruth_eval.model import GenerationParams

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serverless two-stage RAGTruth evaluation: a generation model produces RAG "
        "responses, a detector model flags hallucinations. No TGI/Docker server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["generate", "detect", "all"], default="all",
                        help="Which stage(s) to run.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                        help="Directory holding source_info.jsonl (+ response.jsonl).")
    parser.add_argument("--output-dir", default="outputs/run",
                        help="Directory for generations.jsonl / detections.jsonl / summary.json.")

    # Generation model (Stage 1) — your own checkpoint / LoRA.
    parser.add_argument("--model-id", default=None,
                        help="Generation model: Hub id, local path, or PEFT/LoRA adapter. Required "
                             "for --stage generate/all.")
    parser.add_argument("--base-model-id", default=None,
                        help="Base model for a LoRA --model-id (if not resolvable from its config).")
    parser.add_argument("--tokenizer-id", default=None,
                        help="Tokenizer for --model-id, if not saved with the weights.")

    # Detector model (Stage 2).
    parser.add_argument("--detector-model-id", default=None,
                        help="Detector model: Hub id (e.g. CodingLL/RAGTruth_Eval), local path, or "
                             "PEFT/LoRA adapter. Required for --stage detect/all.")
    parser.add_argument("--detector-base-model-id", default=None,
                        help="Base model for a LoRA --detector-model-id.")
    parser.add_argument("--detector-tokenizer-id", default=None,
                        help="Tokenizer for --detector-model-id, if not saved with the weights.")

    # Shared loading options.
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto", help="device_map for from_pretrained.")

    # Data selection.
    parser.add_argument("--split", default=None,
                        help="Filter source items to this split (train/dev/test) via response.jsonl. "
                             "Omit or 'all' for the whole source_info.jsonl.")
    parser.add_argument("--num-samples", type=int, default=None, help="Use only the first N items.")
    parser.add_argument("--task-types", nargs="+", default=None, choices=list(TASK_TYPES),
                        help="Restrict to these task types (QA / Summary / Data2txt).")

    # Stage 1 decoding.
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", help="Sample instead of greedy (Stage 1).")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)

    # Gold-F1 reproduction mode (detector vs. the corpus's original responses/labels).
    parser.add_argument("--gold-f1", action="store_true",
                        help="Reproduction mode: run the detector on the ORIGINAL responses + gold "
                             "labels and report precision/recall/F1. Skips Stage 1.")
    parser.add_argument("--system-prompt", default=None, help="Optional Stage 1 system prompt.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args(argv)
    if (args.temperature is not None or args.top_p is not None or args.top_k is not None) and not args.do_sample:
        parser.error("--temperature/--top-p/--top-k require --do-sample")
    return args


def _print_summary(summary: dict) -> None:
    rate = summary["rate"]
    print("\n=== RAGTruth hallucination rate ===")
    print(f"overall: {rate['flagged']}/{rate['total']} flagged "
          f"= {rate['hallucination_rate']:.4f} (parse failures: {rate['parse_failures']})")
    for task, s in rate["per_task"].items():
        print(f"  {task:<9} {s['flagged']}/{s['total']} = {s['hallucination_rate']:.4f}")
    if "gold_f1" in summary:
        g = summary["gold_f1"]["overall"]
        print("\n=== detector vs gold (F1 reproduction) ===")
        print(f"overall precision/recall/f1: {g['precision']:.3f} / {g['recall']:.3f} / {g['f1']:.3f}")
        for task, s in summary["gold_f1"]["per_task"].items():
            print(f"  {task:<9} {s['precision']:.3f} / {s['recall']:.3f} / {s['f1']:.3f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    task_types = tuple(args.task_types) if args.task_types else None
    run_generate = args.stage in ("generate", "all") and not args.gold_f1
    run_detect = args.stage in ("detect", "all")

    if run_generate and not args.model_id:
        raise SystemExit("--model-id is required for the generate stage")
    if run_detect and not args.detector_model_id:
        raise SystemExit("--detector-model-id is required for the detect stage")

    # Imported lazily so `--help` / arg errors don't pay the torch import cost.
    if run_generate:
        from ragtruth_eval.generate import run_generation

        gen_params = GenerationParams(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        run_generation(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            model_id=args.model_id,
            base_model_id=args.base_model_id,
            tokenizer_id=args.tokenizer_id,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            dtype=args.dtype,
            split=args.split,
            num_samples=args.num_samples,
            task_types=task_types,
            system_prompt=args.system_prompt,
            gen_params=gen_params,
        )

    if run_detect:
        from ragtruth_eval.detect import run_detection

        summary = run_detection(
            output_dir=args.output_dir,
            detector_model_id=args.detector_model_id,
            base_model_id=args.detector_base_model_id,
            tokenizer_id=args.detector_tokenizer_id,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            dtype=args.dtype,
            gold_f1_mode=args.gold_f1,
            dataset_dir=args.dataset_dir,
            split=args.split or "test",
            num_samples=args.num_samples,
            task_types=task_types,
        )
        _print_summary(summary)
        print(f"\nSummary written to {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
