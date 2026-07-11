"""Prompt construction for FaithEval tasks."""

from __future__ import annotations

from typing import Any

from faitheval.config import TaskConfig

BASE_INSTRUCTION = (
    """You are an expert in retrieval question answering. 
Please respond with the exact answer only. Do not be verbose or provide extra information."""
)


def build_messages(
    example: dict[str, Any],
    task_config: TaskConfig,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build chat-formatted messages for a single FaithEval example."""
    context = example[task_config.context_column]
    question = example[task_config.question_column]
    prompt = (
        f"{BASE_INSTRUCTION}\n"
        f"{task_config.task_specific_prompt}\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages
