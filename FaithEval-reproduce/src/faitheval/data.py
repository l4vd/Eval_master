"""Dataset loading helpers for FaithEval tasks."""

from __future__ import annotations

import logging

from datasets import Dataset, load_dataset

from faitheval.config import TaskConfig

logger = logging.getLogger(__name__)


def load_task_dataset(task_config: TaskConfig, split: str, num_samples: int | None = None) -> Dataset:
    """Load a FaithEval task split, optionally truncated to `num_samples` examples."""
    dataset = load_dataset(task_config.dataset_name, split=split)
    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
    logger.info("Loaded %d examples from %s [%s]", len(dataset), task_config.dataset_name, split)
    return dataset
