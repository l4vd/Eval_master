"""Evaluation loop tying together data loading, generation, and scoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from faitheval.config import ANSWER_MATCH, PHRASE_MATCH, EvalConfig
from faitheval.data import load_task_dataset
from faitheval.metrics import answer_match, phrase_match
from faitheval.model import GenerationParams, HFChatGenerator
from faitheval.prompting import build_messages

logger = logging.getLogger(__name__)


def score_prediction(prediction: str, example: dict[str, Any], config: EvalConfig) -> bool:
    """Score a single prediction according to the task's scoring rule."""
    task_config = config.task_config
    if task_config.scoring == PHRASE_MATCH:
        return phrase_match(prediction, config.active_valid_phrases)
    if task_config.scoring == ANSWER_MATCH:
        reference = example[task_config.answer_column]
        references = reference if isinstance(reference, list) else [reference]
        return answer_match(prediction, references)
    raise ValueError(f"Unknown scoring mode: {task_config.scoring}")  # pragma: no cover


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Run a full FaithEval evaluation and return a summary dict.

    Predictions are streamed to `<output_dir>/<task>_predictions.jsonl` as they
    are produced, and a `<output_dir>/<task>_summary.json` is written once the
    run completes.
    """
    dataset = load_task_dataset(config.task_config, config.split, config.num_samples)
    generator = HFChatGenerator(
        model_id=config.model_id,
        base_model_id=config.base_model_id,
        tokenizer_id=config.tokenizer_id,
        cache_dir=config.cache_dir,
        device_map=config.device_map,
        dtype=config.dtype,
    )
    gen_params = GenerationParams(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{config.task}_predictions.jsonl"

    num_correct = 0
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for example in tqdm(dataset, desc=f"Evaluating [{config.task}]"):
            messages = build_messages(example, config.task_config, config.system_prompt)
            prediction = generator.generate(messages, gen_params)
            correct = score_prediction(prediction, example, config)
            num_correct += int(correct)

            record = {
                "question": example[config.task_config.question_column],
                "prediction": prediction,
                "correct": correct,
            }
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    num_examples = len(dataset)
    accuracy = num_correct / num_examples if num_examples else 0.0
    summary = {
        "task": config.task,
        "model_id": config.model_id,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": accuracy,
    }

    summary_path = output_dir / f"{config.task}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Accuracy: %.4f (%d/%d)", accuracy, num_correct, num_examples)
    logger.info("Predictions written to %s", predictions_path)
    logger.info("Summary written to %s", summary_path)
    return summary
