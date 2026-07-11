"""Dataset loading for the RAGTruth serverless pipeline.

Two corpora files ship under ``dataset/``:

* ``source_info.jsonl`` — one row per RAG *source* item:
  ``{source_id, task_type, source, source_info, prompt}``. ``prompt`` is a
  ready-to-use generation prompt; ``source_info`` is a dict ``{question,
  passages}`` for QA and a string for Summary / Data2txt. This is all Stage 1
  (generation) needs.
* ``response.jsonl`` — one row per *original model response*:
  ``{id, source_id, model, temperature, labels, split, quality, response}``.
  ``labels`` are the gold hallucination spans. Needed only for the optional
  gold-F1 reproduction mode and to know which source ids belong to a split.

``task_type`` is one of ``QA`` / ``Summary`` / ``Data2txt``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TASK_TYPES = ("QA", "Summary", "Data2txt")


def _reference_and_question(task_type: str, source_info: Any) -> tuple[str, str]:
    """Derive the detector's ``reference`` (and ``question`` for QA) from ``source_info``.

    Mirrors ``baseline/prepare_dataset.get_json_data``.
    """
    if task_type == "QA":
        return source_info["passages"], source_info["question"]
    if task_type == "Summary":
        return source_info, ""
    # Data2txt: source_info is structured data; stringify it.
    return f"{source_info}", ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _split_source_ids(dataset_dir: Path, split: str) -> set[str]:
    """Set of source_ids (as strings) that appear in `split` per response.jsonl."""
    responses = _read_jsonl(dataset_dir / "response.jsonl")
    return {str(r["source_id"]) for r in responses if r.get("split") == split}


def load_source_items(
    dataset_dir: str | Path,
    split: str | None = None,
    num_samples: int | None = None,
    task_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Load source items for Stage 1 generation.

    Each returned dict carries ``source_id``, ``task_type``, ``prompt`` (the
    generation prompt), ``reference`` and ``question`` (for the Stage 2 detector).

    ``split`` (``train``/``dev``/``test``) filters to the source ids present in
    that split of ``response.jsonl``; ``None`` (or ``"all"``) keeps everything.
    """
    dataset_dir = Path(dataset_dir)
    rows = _read_jsonl(dataset_dir / "source_info.jsonl")

    keep_ids: set[str] | None = None
    if split and split != "all":
        keep_ids = _split_source_ids(dataset_dir, split)

    items: list[dict[str, Any]] = []
    for row in rows:
        task_type = row["task_type"]
        if task_types and task_type not in task_types:
            continue
        if keep_ids is not None and str(row["source_id"]) not in keep_ids:
            continue
        reference, question = _reference_and_question(task_type, row["source_info"])
        items.append(
            {
                "source_id": row["source_id"],
                "task_type": task_type,
                "prompt": row["prompt"],
                "reference": reference,
                "question": question,
            }
        )

    if num_samples is not None:
        items = items[:num_samples]
    logger.info("Loaded %d source items from %s (split=%s)", len(items), dataset_dir, split)
    return items


def load_gold_responses(
    dataset_dir: str | Path,
    split: str = "test",
    num_samples: int | None = None,
    task_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Load original responses + gold labels for the gold-F1 reproduction mode.

    Merges ``response.jsonl`` (``quality == 'good'`` in `split`) with
    ``source_info.jsonl`` and derives the detector's ``reference``/``question``,
    reproducing ``baseline/prepare_dataset``. Each returned dict has
    ``source_id``, ``task_type``, ``reference``, ``question``, ``response`` and
    ``labels`` (gold spans).
    """
    dataset_dir = Path(dataset_dir)
    responses = _read_jsonl(dataset_dir / "response.jsonl")
    sources = {str(r["source_id"]): r for r in _read_jsonl(dataset_dir / "source_info.jsonl")}

    items: list[dict[str, Any]] = []
    for r in responses:
        if r.get("split") != split or r.get("quality") != "good":
            continue
        src = sources.get(str(r["source_id"]))
        if src is None:
            continue
        task_type = src["task_type"]
        if task_types and task_type not in task_types:
            continue
        reference, question = _reference_and_question(task_type, src["source_info"])
        items.append(
            {
                "source_id": r["source_id"],
                "task_type": task_type,
                "reference": reference,
                "question": question,
                "response": r["response"],
                "labels": r.get("labels", []),
            }
        )

    if num_samples is not None:
        items = items[:num_samples]
    logger.info("Loaded %d gold responses from %s (split=%s)", len(items), dataset_dir, split)
    return items
