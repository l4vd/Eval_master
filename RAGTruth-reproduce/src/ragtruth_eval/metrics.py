"""Metrics for the RAGTruth serverless pipeline.

Primary metric — **hallucination rate**: the fraction of generations the detector
flags (non-empty ``hallucination list``), overall and per task type. This is the
number that characterizes *your* generation model: how often the detector judges
its RAG responses to hallucinate.

Optional secondary metric — **gold F1** (reproduction): detector predictions
vs. the corpus gold labels over the original responses, giving precision / recall
/ F1 per task, reproducing the paper's detector evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def hallucination_rate(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall + per-task hallucination rate over Stage 2 detections.

    Each detection must carry ``task_type`` and ``pred_halu`` (bool: did the
    detector flag any span). ``parse_failed`` (bool), if present, is tallied but
    does not change the rate (a failed parse is treated as "nothing flagged").
    """
    overall_total = 0
    overall_flagged = 0
    overall_failed = 0
    per_task_total: dict[str, int] = defaultdict(int)
    per_task_flagged: dict[str, int] = defaultdict(int)

    for d in detections:
        task = d.get("task_type", "unknown")
        flagged = bool(d.get("pred_halu"))
        overall_total += 1
        per_task_total[task] += 1
        if flagged:
            overall_flagged += 1
            per_task_flagged[task] += 1
        if d.get("parse_failed"):
            overall_failed += 1

    def _rate(flagged: int, total: int) -> float:
        return flagged / total if total else 0.0

    per_task = {
        task: {
            "total": per_task_total[task],
            "flagged": per_task_flagged[task],
            "hallucination_rate": _rate(per_task_flagged[task], per_task_total[task]),
        }
        for task in sorted(per_task_total)
    }

    return {
        "total": overall_total,
        "flagged": overall_flagged,
        "hallucination_rate": _rate(overall_flagged, overall_total),
        "parse_failures": overall_failed,
        "per_task": per_task,
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def gold_f1(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Detector-vs-gold precision/recall/F1 (example-level, "is this response hallucinated").

    Each detection must carry ``task_type``, ``pred_halu`` (predicted) and
    ``gold_halu`` (bool derived from the gold ``labels`` being non-empty).
    Reproduces ``baseline/predict_and_evaluate.py``'s case-level scores.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # task -> [tp, fp, fn]
    overall = [0, 0, 0]

    for d in detections:
        task = d.get("task_type", "unknown")
        pred = bool(d.get("pred_halu"))
        gold = bool(d.get("gold_halu"))
        tp = int(pred and gold)
        fp = int(pred and not gold)
        fn = int((not pred) and gold)
        counts[task][0] += tp
        counts[task][1] += fp
        counts[task][2] += fn
        overall[0] += tp
        overall[1] += fp
        overall[2] += fn

    return {
        "overall": {"total": len(detections), **_prf(*overall)},
        "per_task": {task: _prf(*counts[task]) for task in sorted(counts)},
    }
