"""Evaluation loop tying together data loading, generation, and scoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from faitheval.config import ANSWER_MATCH, CHOICE_LOGLIK, PHRASE_MATCH, EvalConfig
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


def batch_order(lengths: list[int], sort_by_length: str) -> list[int]:
    """Indices 0..n-1 in the order examples should be fed to the generator.

    `lengths` are prompt token lengths. Sorting by them groups similarly-sized prompts into
    the same padded forward pass: the pipeline pads every batch to its own longest prompt,
    so under file order — where length is effectively random — one long context drags a
    whole batch of short ones up to its size. Descending puts the peak-memory batch first,
    so a CUDA OOM surfaces immediately instead of after most of the run is already spent.

    Ties keep their original relative order (`sorted` is stable), so "none" and a split of
    uniform-length prompts both reduce to the identity permutation.
    """
    order = list(range(len(lengths)))
    if sort_by_length == "none":
        return order
    return sorted(order, key=lambda i: lengths[i], reverse=sort_by_length == "desc")


def parse_choices(value: Any) -> tuple[list[str], list[str]]:
    """``(labels, texts)`` of a FaithEval ``choices`` cell, ``{"label": [...], "text": [...]}``.

    The local JSONL stores the cell as a JSON string; a split loaded from the Hub yields
    the dict itself.
    """
    if isinstance(value, str):
        value = json.loads(value)
    labels = [str(label) for label in value["label"]]
    texts = [str(text) for text in value["text"]]
    if not texts or len(labels) != len(texts):
        raise ValueError(f"malformed choices: {len(labels)} labels for {len(texts)} options")
    return labels, texts


def resolve_gold(
    labels: list[str],
    answer_key: Any,
    *,
    texts: list[str] | None = None,
    answer: Any = None,
) -> int:
    """Position of the correct option, named by its label (FaithEval's ``answerKey``).

    The counterfactual split's NYSEDREGENTS_* rows label their options 1-4 but key them
    A-D. For those the letter is read as a position (A = first option), accepted only if
    the option there is the row's ``answer`` text, so a key that is merely out of range
    of the labels still fails loudly rather than being silently reinterpreted.
    """
    key = str(answer_key).strip()
    if key in labels:
        return labels.index(key)
    if len(key) == 1 and "A" <= key <= "Z" and all(label.isdigit() for label in labels):
        position = ord(key) - ord("A")
        if (
            position < len(labels)
            and texts is not None
            and answer is not None
            and texts[position].strip() == str(answer).strip()
        ):
            return position
    raise ValueError(
        f"answer key {answer_key!r} is not one of the option labels {labels}"
        " and does not name the answer text by position"
    )


def argmax(values: list[float]) -> int:
    """Index of the largest value; the earliest one on a tie."""
    return max(range(len(values)), key=lambda i: (values[i], -i))


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Run a full FaithEval evaluation and return a summary dict.

    Predictions are streamed to `<output_dir>/<task>_predictions.jsonl` as they
    are produced, and a `<output_dir>/<task>_summary.json` is written once the
    run completes.

    Examples are generated in length-sorted order (see `batch_order`) but the predictions
    file is restored to dataset order before the function returns, so the artifact does not
    depend on the execution order.
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

    # Built up front so the same rendering is used for the sort key and for generation.
    all_messages = [
        build_messages(example, config.task_config, config.system_prompt) for example in examples
    ]
    order = batch_order(generator.prompt_token_lengths(all_messages), config.sort_by_length)

    if config.task_config.scoring == CHOICE_LOGLIK:
        return _run_choice_loglik(config, generator, examples, all_messages, order, output_dir)

    num_correct = 0
    total_words = 0
    total_tokens = 0
    num_truncated = 0
    # Streamed rather than buffered so a job killed mid-run still leaves usable predictions;
    # `index` is what makes those partial rows attributable back to the dataset.
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        with tqdm(total=num_examples, desc=f"Evaluating [{config.task}]") as progress:
            for start in range(0, num_examples, config.batch_size):
                indices = order[start : min(start + config.batch_size, num_examples)]
                batch = [all_messages[i] for i in indices]
                predictions = generator.generate_many(batch, gen_params)

                for index, prediction in zip(indices, predictions):
                    example = examples[index]
                    correct = score_prediction(prediction, example, config)
                    num_correct += int(correct)
                    total_words += len(prediction.split())
                    # Re-tokenizes the DECODED string: the pipeline discards the raw
                    # generated ids, so an exact count would mean bypassing it. Within a
                    # token or two of the true length, which is all this probe needs.
                    num_tokens = len(
                        generator.tokenizer(prediction, add_special_tokens=False)["input_ids"]
                    )
                    total_tokens += num_tokens
                    num_truncated += int(num_tokens >= config.max_new_tokens)

                    record = {
                        "index": index,
                        "question": example[config.task_config.question_column],
                        "prediction": prediction,
                        "correct": correct,
                    }
                    predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.update(len(indices))

    _sort_predictions_file(predictions_path)

    accuracy = num_correct / num_examples if num_examples else 0.0
    # Mean answer length is a cheap degeneration probe: this task's prompt asks for
    # "the exact answer only" and is scored by exact match, so an arm whose mean runs
    # into the tens of tokens is ignoring the instruction — which shows up as a near-zero
    # accuracy AND as a multi-hour runtime, both from the same cause.
    #
    # Tokens are the primary unit because they are directly comparable to max_new_tokens,
    # the budget this probe exists to detect a model running into; truncation_rate names
    # that failure outright. Words are kept alongside for continuity with earlier runs.
    mean_prediction_words = total_words / num_examples if num_examples else 0.0
    mean_prediction_tokens = total_tokens / num_examples if num_examples else 0.0
    truncation_rate = num_truncated / num_examples if num_examples else 0.0
    summary = {
        "task": config.task,
        "model_id": config.model_id,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "mean_prediction_tokens": mean_prediction_tokens,
        "truncation_rate": truncation_rate,
        "mean_prediction_words": mean_prediction_words,
        "batch_size": config.batch_size,
        "sort_by_length": config.sort_by_length,
        "max_new_tokens": config.max_new_tokens,
        # Additive provenance (no metric changes): what a resumed or re-scored run is
        # checked against. Summaries written before these keys existed lack them.
        "num_samples": config.num_samples,
        "split": config.split,
        "strict_match": config.strict_match,
        "dtype": config.dtype,
    }

    summary_path = output_dir / f"{config.task}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Accuracy: %.4f (%d/%d)", accuracy, num_correct, num_examples)
    logger.info(
        "Mean prediction length: %.1f tokens (%.1f words); %.1f%% hit the %d-token budget",
        mean_prediction_tokens,
        mean_prediction_words,
        100 * truncation_rate,
        config.max_new_tokens,
    )
    logger.info("Predictions written to %s", predictions_path)
    logger.info("Summary written to %s", summary_path)
    return summary


def _run_choice_loglik(
    config: EvalConfig,
    generator: HFChatGenerator,
    examples: list[dict[str, Any]],
    all_messages: list[list[dict[str, str]]],
    order: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    """Score every answer option by log-likelihood: no generation, no answer parser.

    Everything but the scoring is shared with the generation path: the task's own prompt
    (`build_messages`), batches cut in the same length-sorted order, and predictions
    restored to dataset order. `accuracy` takes the option with the highest summed
    log-probability; `accuracy_norm` the highest per character (lm-eval's acc_norm), which
    removes the edge short options get from having fewer tokens to pay for.
    """
    task_config = config.task_config
    predictions_path = output_dir / f"{config.task}_predictions.jsonl"
    num_examples = len(examples)
    num_correct = 0
    num_correct_norm = 0
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        with tqdm(total=num_examples, desc=f"Scoring options [{config.task}]") as progress:
            for start in range(0, num_examples, config.batch_size):
                indices = order[start : min(start + config.batch_size, num_examples)]
                options = [parse_choices(examples[i][task_config.choices_column]) for i in indices]
                scores = generator.choice_logprobs(
                    [all_messages[i] for i in indices], [texts for _, texts in options]
                )
                for index, (labels, texts), option_scores in zip(indices, options, scores):
                    example = examples[index]
                    gold = resolve_gold(
                        labels,
                        example[task_config.answer_key_column],
                        texts=texts,
                        answer=example.get(task_config.answer_column),
                    )
                    predicted = argmax([logprob for logprob, _, _ in option_scores])
                    predicted_norm = argmax(
                        [logprob / max(n_chars, 1) for logprob, _, n_chars in option_scores]
                    )
                    num_correct += int(predicted == gold)
                    num_correct_norm += int(predicted_norm == gold)
                    record = {
                        "index": index,
                        "question": example[task_config.question_column],
                        "choice_scores": [
                            {"label": label, "logprob": logprob, "n_tokens": n_tokens, "n_chars": n_chars}
                            for label, (logprob, n_tokens, n_chars) in zip(labels, option_scores)
                        ],
                        "predicted": labels[predicted],
                        "predicted_norm": labels[predicted_norm],
                        "gold": labels[gold],
                        "correct": predicted == gold,
                        "correct_norm": predicted_norm == gold,
                    }
                    predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.update(len(indices))

    _sort_predictions_file(predictions_path)

    summary = {
        "task": config.task,
        "model_id": config.model_id,
        "scoring": CHOICE_LOGLIK,
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy": num_correct / num_examples if num_examples else 0.0,
        "num_correct_norm": num_correct_norm,
        "accuracy_norm": num_correct_norm / num_examples if num_examples else 0.0,
        "batch_size": config.batch_size,
        "sort_by_length": config.sort_by_length,
        "prompt_format": getattr(generator, "prompt_format", None),
        "num_samples": config.num_samples,
        "split": config.split,
        "dtype": config.dtype,
    }
    summary_path = output_dir / f"{config.task}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Accuracy: %.4f (%d/%d); per-character: %.4f",
        summary["accuracy"], num_correct, num_examples, summary["accuracy_norm"],
    )
    logger.info("Predictions written to %s", predictions_path)
    logger.info("Summary written to %s", summary_path)
    return summary


def _sort_predictions_file(path: Path) -> None:
    """Rewrite the streamed predictions file in dataset order.

    Generation runs in length-sorted order, so the streamed file comes out in that order
    too. Restoring dataset order here keeps the artifact diffable against a run made under
    a different `sort_by_length` — the ordering is an execution detail, not a result.
    """
    with path.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    records.sort(key=lambda r: r["index"])
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
