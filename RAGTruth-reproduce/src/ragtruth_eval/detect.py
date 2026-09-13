"""Stage 2 — detection.

Load the detector model (the RAGTruth authors' released ``CodingLL/RAGTruth_Eval``
or your own), read Stage 1's ``generations.jsonl``, build the per-task detector
prompt, generate, parse ``{"hallucination list": [...]}``, and write
``detections.jsonl`` + ``summary.json`` (the hallucination-rate summary).

With ``gold_f1=True`` the stage instead reads the corpus's *original* responses
and gold labels (``ragtruth_eval.data.load_gold_responses``) and the summary adds
detector-vs-gold precision/recall/F1 — the paper's detector evaluation, no TGI
server required.

Prompts go to the detector in batches of ``batch_size`` (left-padded, longest first);
``detections.jsonl`` keeps dataset order. The authors' decoding samples, so ``seed`` is
applied before each run directory: a directory's labels then do not depend on what ran
before it in the same process. :func:`run_detection_job` runs the stage over many run
directories with ONE detector load, skipping those already finished.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ragtruth_eval.data import load_gold_responses
from ragtruth_eval.metrics import gold_f1, hallucination_rate
from ragtruth_eval.model import GenerationParams, HFGenerator, generation_order
from ragtruth_eval.prompts import build_detector_prompt, parse_hallucination_list

logger = logging.getLogger(__name__)

# The released detector was trained/served with these decoding params
# (baseline/predict_and_evaluate.py). Kept as the default for faithfulness.
DEFAULT_DETECTOR_PARAMS = GenerationParams(
    max_new_tokens=512, do_sample=True, temperature=0.05, top_p=0.95, top_k=40
)
GENERATION_SUMMARY = "generation_summary.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _seed_everything(seed: int) -> None:
    from transformers import set_seed

    set_seed(seed)


def load_detector(
    detector_model_id: str,
    base_model_id: str | None = None,
    tokenizer_id: str | None = None,
    cache_dir: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
) -> HFGenerator:
    return HFGenerator(
        model_id=detector_model_id,
        base_model_id=base_model_id,
        tokenizer_id=tokenizer_id,
        cache_dir=cache_dir,
        device_map=device_map,
        dtype=dtype,
    )


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
    *,
    detector: HFGenerator | None = None,
    batch_size: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run Stage 2 and return the summary dict (also written to ``summary.json``).

    ``detector`` reuses an already loaded detector (the shared job); otherwise one is
    loaded here, as before.
    """
    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation: dict[str, Any] = {}
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
        generation_summary = output_dir / GENERATION_SUMMARY
        if generation_summary.is_file():
            generation = json.loads(generation_summary.read_text(encoding="utf-8"))

    if detector is None:
        detector = load_detector(detector_model_id, base_model_id, tokenizer_id, cache_dir, device_map, dtype)
    gen_params = gen_params or DEFAULT_DETECTOR_PARAMS
    if seed is not None:
        _seed_everything(seed)

    sequences = [detector.completion_ids(build_detector_prompt(item)) for item in items]
    order = generation_order([len(ids) for ids in sequences], batch_size)
    raw_outputs: list[str] = [""] * len(items)
    partial_path = output_dir / "detections.jsonl.partial"
    with partial_path.open("w", encoding="utf-8") as partial, \
            tqdm(total=len(items), desc="Stage 2 [detect]") as progress:
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            for index, raw in zip(indices, detector.generate_ids([sequences[i] for i in indices], gen_params)):
                raw_outputs[index] = raw
                partial.write(json.dumps({"index": index, "detector_output": raw}, ensure_ascii=False) + "\n")
            progress.update(len(indices))

    detections_path = output_dir / "detections.jsonl"
    detections: list[dict[str, Any]] = []
    with detections_path.open("w", encoding="utf-8") as out:
        for item, raw in zip(items, raw_outputs):
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
    partial_path.unlink()

    summary: dict[str, Any] = {
        "detector_model_id": detector_model_id,
        "gold_f1_mode": gold_f1_mode,
        "rate": hallucination_rate(detections),
    }
    if gold_f1_mode:
        summary["gold_f1"] = gold_f1(detections)

    # Provenance, beside the metrics above (which are unchanged): what the numbers were
    # computed on and with. `split` is Stage 1's; None when the generations predate
    # generation_summary.json.
    detect_elapsed = round(time.monotonic() - started, 1)
    summary.update({
        "split": split if gold_f1_mode else generation.get("split"),
        "generation": {k: generation[k] for k in ("model_id", "dtype", "decoding", "batch_size",
                                                   "num_generations", "elapsed_s") if k in generation} or None,
        "detector_decoding": dataclasses.asdict(gen_params),
        "detector_batch_size": batch_size,
        "detector_seed": seed,
        "detect_elapsed_s": detect_elapsed,
        # Both stages' wall clock (the detector load included when this call loaded it).
        "elapsed_s": round(detect_elapsed + float(generation.get("elapsed_s") or 0.0), 1),
    })

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d detections to %s", len(detections), detections_path)
    logger.info("Summary written to %s", summary_path)
    return summary


def detection_dirs(patterns: list[str]) -> tuple[list[Path], list[Path], list[Path]]:
    """``(pending, finished, without_generations)`` among run directories (globs accepted).

    Pending = has ``generations.jsonl`` and no ``summary.json``; finished = has a
    ``summary.json``, and is skipped.
    """
    seen: list[Path] = []
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)) or [pattern]:
            path = Path(match)
            if path.is_dir() and path not in seen:
                seen.append(path)
    finished = [d for d in seen if (d / "summary.json").is_file()]
    pending = [d for d in seen if d not in finished and (d / "generations.jsonl").is_file()]
    without = [d for d in seen if d not in finished and d not in pending]
    return pending, finished, without


def run_detection_job(
    output_dirs: list[str],
    detector_model_id: str,
    base_model_id: str | None = None,
    tokenizer_id: str | None = None,
    cache_dir: str | None = None,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    gen_params: GenerationParams | None = None,
    *,
    batch_size: int = 1,
    seed: int | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Stage 2 over every pending run directory with a single detector load.

    Resumable: a directory is done once its ``summary.json`` exists, so re-submitting the
    same job only processes what is left.
    """
    pending, finished, without = detection_dirs(output_dirs)
    logger.info("Detection job: %d pending, %d already finished (skipped), %d without generations.jsonl",
                len(pending), len(finished), len(without))
    for d in without:
        logger.warning("No generations.jsonl in %s: run its generate stage first", d)
    if not pending:
        return []
    detector = load_detector(detector_model_id, base_model_id, tokenizer_id, cache_dir, device_map, dtype)
    logger.info("Detector %s loaded once for %d run directories", detector_model_id, len(pending))
    results = []
    for i, out_dir in enumerate(pending, 1):
        logger.info("[%d/%d] detecting %s", i, len(pending), out_dir)
        results.append((out_dir, run_detection(
            out_dir, detector_model_id, gen_params=gen_params,
            detector=detector, batch_size=batch_size, seed=seed,
        )))
    return results
