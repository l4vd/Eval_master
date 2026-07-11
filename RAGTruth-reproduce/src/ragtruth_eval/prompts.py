"""Prompt construction and detector-output parsing for RAGTruth.

* Stage 1 (generation): the generation prompt is simply the source item's own
  ``prompt`` field, sent as a chat user message.
* Stage 2 (detection): the detector prompt is the per-task ``TEMPLATES`` filled
  with the reference / response, then wrapped in ``[INST] ... [/INST]`` — ported
  from ``baseline/dataset.py`` (``TEMPLATES`` and
  ``process_dialog_to_single_turn(..., return_prompt=True)``), which is how the
  released detector (`CodingLL/RAGTruth_Eval`) was trained and served.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Ported verbatim from baseline/dataset.py (the detector's training/serving prompt).
TEMPLATES = {
    "QA": (
        "Below is a question:\n"
        "{question}\n\n"
        "Below are related passages:\n"
        "{reference}\n\n"
        "Below is an answer:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
    "Summary": (
        "Below is the original news:\n"
        "{reference}\n\n"
        "Below is a summary of the news:\n"
        "{response}\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
    "Data2txt": (
        "Below is a structured data in the JSON format:\n"
        "{reference}\n\n"
        "Below is an overview article written in accordance with the structured data:\n"
        "{response}\n\n"
        "Your task is to determine whether the summary contains either or both of the following two types of hallucinations:\n"
        "1. conflict: instances where the summary presents direct contraction or opposition to the original news;\n"
        "2. baseless info: instances where the generated summary includes information which is not substantiated by or inferred from the original news. \n"
        "Then, compile the labeled hallucinated spans into a JSON dict, with a key \"hallucination list\" and its value is a list of hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{\"hallucination list\": [hallucination span1, hallucination span2, ...]}}. Otherwise, leave the value as a empty list as following: {{\"hallucination list\": []}}.\n"
        "Output:"
    ),
}

B_INST, E_INST = "[INST]", "[/INST]"


def build_generation_messages(item: dict[str, Any], system_prompt: str | None = None) -> list[dict[str, str]]:
    """Chat messages for Stage 1: the source item's own RAG prompt as a user turn."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": item["prompt"]})
    return messages


def build_detector_prompt(item: dict[str, Any]) -> str:
    """Detector prompt for Stage 2 (the `[INST]`-wrapped per-task template).

    `item` must carry ``task_type``, ``reference``, ``response`` (and ``question``
    for QA). Equivalent to
    ``process_dialog_to_single_turn(item, tokenizer, return_prompt=True)`` +
    the ``[INST] ... [/INST]`` wrap the baseline client applied.
    """
    task_type = item["task_type"]
    if task_type == "QA":
        prompt = TEMPLATES[task_type].format(
            question=item.get("question", ""),
            reference=item["reference"],
            response=item["response"],
        )
    else:
        prompt = TEMPLATES[task_type].format(
            reference=item["reference"],
            response=item["response"],
        )
    return f"{B_INST} {prompt.strip()} {E_INST}"


def parse_hallucination_list(text: str) -> tuple[list[str], bool]:
    """Parse the detector output into a hallucination-span list.

    Returns ``(spans, ok)``. The detector is trained to emit
    ``{"hallucination list": [...]}``; the baseline TGI client simply retried
    generation on a JSON error. In-process greedy decoding can't benefit from a
    retry, so we instead repair the common failure modes: extract the outermost
    ``{...}`` object, and fall back to a regex over the ``"hallucination list"``
    array. On total failure we return ``([], False)`` (treated as "no
    hallucination flagged", and counted as a parse failure in the summary).
    """
    text = text.strip()

    def _coerce(obj: Any) -> list[str] | None:
        if isinstance(obj, dict):
            value = obj.get("hallucination list", [])
            if isinstance(value, list):
                return [str(v) for v in value]
        return None

    # 1. Direct JSON.
    try:
        spans = _coerce(json.loads(text))
        if spans is not None:
            return spans, True
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Outermost {...} object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            spans = _coerce(json.loads(text[start : end + 1]))
            if spans is not None:
                return spans, True
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Regex over the "hallucination list" array contents.
    match = re.search(r'"hallucination list"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if match:
        inner = match.group(1).strip()
        if not inner:
            return [], True
        spans = re.findall(r'"((?:[^"\\]|\\.)*)"', inner)
        if spans:
            return [s.encode().decode("unicode_escape") for s in spans], True

    return [], False
