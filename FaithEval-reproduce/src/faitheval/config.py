"""Configuration schema and loading utilities for FaithEval evaluation runs.

Task-specific settings (dataset, prompt, scoring rule) live in the YAML files
under ``configs/`` and are loaded into :class:`TaskConfig`. Run-specific
settings (model, decoding, I/O) are supplied via CLI flags and assembled into
:class:`EvalConfig` by :mod:`faitheval.cli`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TASKS = ("unanswerable", "inconsistent", "counterfactual", "counterfactual_mc")
PHRASE_MATCH = "phrase_match"
ANSWER_MATCH = "answer_match"
# Not a rule over generated text: every answer option is scored by its log-likelihood as the
# continuation of the task prompt, and the likeliest option is the prediction. A modified
# protocol (configs/counterfactual_mc.yaml), reported beside the original tasks.
CHOICE_LOGLIK = "choice_loglik"
SCORING_MODES = (PHRASE_MATCH, ANSWER_MATCH, CHOICE_LOGLIK)
SORT_ORDERS = ("desc", "asc", "none")


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Dataset and scoring settings for a single FaithEval task."""

    dataset_name: str
    scoring: str
    task_specific_prompt: str = ""
    valid_phrases: list[str] = dataclasses.field(default_factory=list)
    strict_valid_phrases: list[str] = dataclasses.field(default_factory=list)
    context_column: str = "context"
    question_column: str = "question"
    answer_column: str = "answer"
    # choice_loglik only: the column holding the options ({"label": [...], "text": [...]},
    # or that dict as a JSON string) and the column naming the correct option's label.
    choices_column: str | None = None
    answer_key_column: str | None = None

    def __post_init__(self) -> None:
        if self.scoring not in SCORING_MODES:
            raise ValueError(f"Unknown scoring mode: {self.scoring!r}")
        if self.scoring == PHRASE_MATCH and not self.valid_phrases:
            raise ValueError("phrase_match scoring requires non-empty valid_phrases")
        if self.scoring == CHOICE_LOGLIK and not (self.choices_column and self.answer_key_column):
            raise ValueError("choice_loglik scoring requires choices_column and answer_key_column")


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Full configuration for a single evaluation run."""

    task: str
    task_config: TaskConfig
    model_id: str
    base_model_id: str | None = None
    tokenizer_id: str | None = None
    cache_dir: str | None = None
    split: str = "test"
    num_samples: int | None = None
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    strict_match: bool = False
    system_prompt: str | None = None
    output_dir: str = "outputs"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    batch_size: int = 1
    # Order in which examples are fed to the generator. "desc" groups similarly-long
    # prompts into the same padded forward pass — the pipeline pads each batch to its own
    # longest prompt, so file order (where length is effectively random) wastes most of the
    # compute on padding. Descending rather than ascending puts the peak-memory batch FIRST,
    # so a CUDA OOM surfaces in the first minute instead of at 90% completion.
    sort_by_length: str = "desc"

    def __post_init__(self) -> None:
        if self.sort_by_length not in SORT_ORDERS:
            raise ValueError(
                f"Unknown sort_by_length: {self.sort_by_length!r}; choose from {SORT_ORDERS}"
            )
        # `TaskConfig` checks `valid_phrases`, but the strict list is only reachable
        # through this flag, so it can only be checked here. Without this, --strict-match
        # on a task config that omits `strict_valid_phrases` scores every example False
        # and reports accuracy 0.0 as if the model had simply failed.
        if (
            self.strict_match
            and self.task_config.scoring == PHRASE_MATCH
            and not self.task_config.strict_valid_phrases
        ):
            raise ValueError(
                f"--strict-match needs non-empty 'strict_valid_phrases' for task "
                f"{self.task!r}, which uses {PHRASE_MATCH} scoring, but its config defines "
                "none. Every answer would score as wrong (accuracy 0.0) with no error. Add "
                "them to the task's YAML, or drop --strict-match to use 'valid_phrases'."
            )

    @property
    def active_valid_phrases(self) -> list[str]:
        """Valid phrases to use for phrase-match scoring, honoring `strict_match`."""
        if self.strict_match:
            return self.task_config.strict_valid_phrases
        return self.task_config.valid_phrases


def load_task_config(config_path: Path) -> TaskConfig:
    """Load a task's YAML config into a :class:`TaskConfig`."""
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    try:
        return TaskConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"Invalid task config at {config_path}: {exc}") from exc
