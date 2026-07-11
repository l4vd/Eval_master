"""Stage 1 — generation.

Load the generation model (your own checkpoint / LoRA), iterate the RAGTruth
source items, generate a RAG response for each, and stream them to
``<output_dir>/generations.jsonl`` as
``{source_id, task_type, prompt, reference, question, response}``.

The ``reference`` / ``question`` fields are carried through so Stage 2 (detection)
does not need to re-open the corpus.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ragtruth_eval.data import load_source_items
from ragtruth_eval.model import GenerationParams, HFGenerator
from ragtruth_eval.prompts import build_generation_messages

logger = logging.getLogger(__name__)


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
) -> Path:
    """Run Stage 1 and return the path to the written ``generations.jsonl``."""
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

    with generations_path.open("w", encoding="utf-8") as out:
        for item in tqdm(items, desc="Stage 1 [generate]"):
            messages = build_generation_messages(item, system_prompt)
            response = generator.chat(messages, gen_params)
            record: dict[str, Any] = {
                "source_id": item["source_id"],
                "task_type": item["task_type"],
                "prompt": item["prompt"],
                "reference": item["reference"],
                "question": item["question"],
                "response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d generations to %s", len(items), generations_path)
    return generations_path
