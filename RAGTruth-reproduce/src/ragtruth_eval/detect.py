"""Stage 2 — detection.

Load the detector model (the RAGTruth authors' released ``CodingLL/RAGTruth_Eval``
or your own), read Stage 1's ``generations.jsonl``, build the per-task detector
prompt, generate, parse ``{"hallucination list": [...]}``, and write
``detections.jsonl`` + ``summary.json`` (the hallucination-rate summary).

With ``gold_f1=True`` the stage instead reads the corpus's *original* responses
and gold labels (``ragtruth_eval.data.load_gold_responses``) and the summary adds
detector-vs-gold precision/recall/F1 — the paper's detector evaluation, no TGI
server required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ragtruth_eval.data import load_gold_responses
from ragtruth_eval.metrics import gold_f1, hallucination_rate
from ragtruth_eval.model import GenerationParams, HFGenerator
from ragtruth_eval.prompts import build_detector_prompt, parse_hallucination_list

logger = logging.getLogger(__name__)

# The released detector was trained/served with these decoding params
# (baseline/predict_and_evaluate.py). Kept as the default for faithfulness.
DEFAULT_DETECTOR_PARAMS = GenerationParams(
    max_new_tokens=512, do_sample=True, temperature=0.05, top_p=0.95, top_k=40
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_detection(
    output_dir: str | Path,
    detector_model_id: str,
    base_model_id: str | None = None,
    tokenizer_id: str | None = None,
    cache_dir: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    gen_params: GenerationParams | None = None,
    gold_f1_mode: bool = False,
    dataset_dir: str | Path | None = None,
    split: str = "test",
    num_samples: int | None = None,
    task_types: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run Stage 2 and return the summary dict (also written to ``summary.json``)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gold_f1_mode:
        if dataset_dir is None:
            raise ValueError("gold_f1 mode requires dataset_dir")
        items = load_gold_responses(dataset_dir, split=split, num_samples=num_samples, task_types=task_types)
    else:
        generations_path = output_dir / "generations.jsonl"
        if not generations_path.is_file():
            raise FileNotFoundError(
                f"{generations_path} not found — run Stage 1 (generate) first."
            )
        items = _read_jsonl(generations_path)

    detector = HFGenerator(
        model_id=detector_model_id,
        base_model_id=base_model_id,
        tokenizer_id=tokenizer_id,
        cache_dir=cache_dir,
        device_map=device_map,
        dtype=dtype,
    )
    gen_params = gen_params or DEFAULT_DETECTOR_PARAMS

    detections_path = output_dir / "detections.jsonl"
    detections: list[dict[str, Any]] = []
    with detections_path.open("w", encoding="utf-8") as out:
        for item in tqdm(items, desc="Stage 2 [detect]"):
            prompt = build_detector_prompt(item)
            raw = detector.complete(prompt, gen_params)
            spans, ok = parse_hallucination_list(raw)
            record: dict[str, Any] = {
                "source_id": item["source_id"],
                "task_type": item["task_type"],
                "response": item["response"],
                "hallucination_list": spans,
                "pred_halu": len(spans) > 0,
                "parse_failed": not ok,
                "detector_output": raw,
            }
            if gold_f1_mode:
                record["gold_halu"] = len(item.get("labels", [])) > 0
            detections.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "detector_model_id": detector_model_id,
        "gold_f1_mode": gold_f1_mode,
        "rate": hallucination_rate(detections),
    }
    if gold_f1_mode:
        summary["gold_f1"] = gold_f1(detections)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d detections to %s", len(detections), detections_path)
    logger.info("Summary written to %s", summary_path)
    return summary
