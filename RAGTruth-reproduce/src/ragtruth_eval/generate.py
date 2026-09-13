"""Stage 1 — generation.

Load the generation model (your own checkpoint / LoRA), iterate the RAGTruth
source items, generate a RAG response for each, and write them to
``<output_dir>/generations.jsonl`` as
``{source_id, task_type, prompt, reference, question, response}``.

The ``reference`` / ``question`` fields are carried through so Stage 2 (detection)
does not need to re-open the corpus.

Prompts are generated in batches of ``batch_size`` (left-padded, longest first) and
``generations.jsonl`` is written in dataset order once every item is done; while the stage
runs, finished responses stream to ``generations.jsonl.partial``. ``generation_summary.json``
is written last: it records split, decoding, batch size and wall clock, and is the marker a
finished generate-only run is recognised by (``analysis.eval_checkpoints --resume``, the
shared detection job).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ragtruth_eval.data import load_source_items
from ragtruth_eval.model import GenerationParams, HFGenerator, generation_order
from ragtruth_eval.prompts import build_generation_messages

logger = logging.getLogger(__name__)

GENERATION_SUMMARY = "generation_summary.json"


def run_generation(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    base_model_id: str | None = None,
    tokenizer_id: str | None = None,
    cache_dir: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    split: str | None = None,
    num_samples: int | None = None,
    task_types: tuple[str, ...] | None = None,
    system_prompt: str | None = None,
    gen_params: GenerationParams | None = None,
    batch_size: int = 1,
) -> Path:
    """Run Stage 1 and return the path to the written ``generations.jsonl``."""
    started = time.monotonic()
    items = load_source_items(dataset_dir, split=split, num_samples=num_samples, task_types=task_types)
    generator = HFGenerator(
        model_id=model_id,
        base_model_id=base_model_id,
        tokenizer_id=tokenizer_id,
        cache_dir=cache_dir,
        device_map=device_map,
        dtype=dtype,
    )
    gen_params = gen_params or GenerationParams()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"
    partial_path = output_dir / "generations.jsonl.partial"
    summary_path = output_dir / GENERATION_SUMMARY
    # A re-run must not leave the previous run's completion markers standing while it works.
    for stale in (summary_path, generations_path):
        if stale.exists():
            stale.unlink()

    sequences = [generator.chat_ids(build_generation_messages(item, system_prompt)) for item in items]
    order = generation_order([len(ids) for ids in sequences], batch_size)
    responses: list[str] = [""] * len(items)
    with partial_path.open("w", encoding="utf-8") as partial, \
            tqdm(total=len(items), desc="Stage 1 [generate]") as progress:
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            for index, response in zip(indices, generator.generate_ids([sequences[i] for i in indices], gen_params)):
                responses[index] = response
                partial.write(json.dumps({"index": index, "source_id": items[index]["source_id"],
                                          "response": response}, ensure_ascii=False) + "\n")
            progress.update(len(indices))

    with generations_path.open("w", encoding="utf-8") as out:
        for item, response in zip(items, responses):
            record: dict[str, Any] = {
                "source_id": item["source_id"],
                "task_type": item["task_type"],
                "prompt": item["prompt"],
                "reference": item["reference"],
                "question": item["question"],
                "response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    partial_path.unlink()

    summary = {
        "stage": "generate",
        "model_id": model_id,
        "base_model_id": base_model_id,
        "dtype": dtype,
        "split": split if split else "all",
        "task_types": list(task_types) if task_types else None,
        "num_samples_requested": num_samples,
        "num_generations": len(items),
        "decoding": dataclasses.asdict(gen_params),
        "batch_size": batch_size,
        "sort_by_length": "desc" if batch_size > 1 else "none",
        "system_prompt": system_prompt,
        # Wall clock of the whole stage, model load included: what a pilot budgets on.
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d generations to %s in %.0fs", len(items), generations_path, summary["elapsed_s"])
    return generations_path
