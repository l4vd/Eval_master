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
        batch_size=config.batch_size,
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

    # Materialized so the loop can slice batches; the FaithEval splits are ~1k rows.
    examples = list(dataset)
    num_examples = len(examples)

    num_correct = 0
    total_words = 0
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        with tqdm(total=num_examples, desc=f"Evaluating [{config.task}]") as progress:
            for start in range(0, num_examples, config.batch_size):
                chunk = examples[start : min(start + config.batch_size, num_examples)]
                batch = [
                    build_messages(example, config.task_config, config.system_prompt)
                    for example in chunk
                ]
                predictions = generator.generate_many(batch, gen_params)

                for example, prediction in zip(chunk, predictions):
                    correct = score_prediction(prediction, example, config)
                    num_correct += int(correct)
                    total_words += len(prediction.split())

                    record = {
                        "question": example[config.task_config.question_column],
                        "prediction": prediction,
                        "correct": correct,
                    }
                    predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.update(len(chunk))

    accuracy = num_correct / num_examples if num_examples else 0.0
    # Mean answer length is a cheap degeneration probe: this task's prompt asks for
    # "the exact answer only" and is scored by exact match, so an arm whose mean runs
    # into the tens of words is ignoring the instruction — which shows up as a near-zero
    # accuracy AND as a multi-hour runtime, both from the same cause.
    mean_prediction_words = total_words / num_examples if num_examples else 0.0
    summary = {
        "task": config.task,
        "model_id": config.model_id,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "mean_prediction_words": mean_prediction_words,
        "batch_size": config.batch_size,
        "max_new_tokens": config.max_new_tokens,
    }

    summary_path = output_dir / f"{config.task}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Accuracy: %.4f (%d/%d)", accuracy, num_correct, num_examples)
    logger.info("Mean prediction length: %.1f words", mean_prediction_words)
    logger.info("Predictions written to %s", predictions_path)
    logger.info("Summary written to %s", summary_path)
    return summary
