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

SUPPORTED_TASKS = ("unanswerable", "inconsistent", "counterfactual")
PHRASE_MATCH = "phrase_match"
ANSWER_MATCH = "answer_match"


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

    def __post_init__(self) -> None:
        if self.scoring not in (PHRASE_MATCH, ANSWER_MATCH):
            raise ValueError(f"Unknown scoring mode: {self.scoring!r}")
        if self.scoring == PHRASE_MATCH and not self.valid_phrases:
            raise ValueError("phrase_match scoring requires non-empty valid_phrases")


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
