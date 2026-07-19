#!/usr/bin/env python
"""Materialise the FaithEval task datasets into local, version-agnostic JSONL.

Run this **once, on a machine WITH internet access** (your laptop, or an HPC
login node that has connectivity). It downloads each FaithEval task split from
the Hugging Face Hub and writes it as plain JSON Lines under ``data/faitheval/``,
one file per task/split:

    data/faitheval/FaithEval-unanswerable-v1.0/test.jsonl
    data/faitheval/FaithEval-inconsistent-v1.0/test.jsonl
    data/faitheval/FaithEval-counterfactual-v1.0/test.jsonl

Why: the FaithEval Hub datasets were published with ``datasets>=4``, whose
parquet/cache metadata tags list columns with the newer ``List`` feature type.
Any *older* ``datasets`` (the cluster's pinned ``datasets<3``, but also e.g. a
``datasets`` 3.x on your laptop) blows up trying to read that metadata via
``load_dataset`` (``ValueError: Feature type 'List' not found`` /
``TypeError: must be called with a dataclass type or instance``). So this script
deliberately does **not** call ``load_dataset``: it fetches the raw parquet files
from the Hub with ``huggingface_hub`` and reads them with ``pyarrow``, which
ignores the HF feature metadata entirely. The rows are then written as plain
JSONL — which carries no version-stamped metadata and loads on **any**
``datasets`` version and any platform.

Copy the resulting ``data/faitheval/`` directory to the cluster (or commit it)
and the eval reads it directly via ``faitheval.data`` — no ``load_dataset``, no
Hub access, no version coupling.

The ``datasets`` version installed where you RUN this script genuinely does not
matter (it is never imported); you only need internet access and the
``huggingface_hub`` + ``pyarrow`` that ``datasets`` already pulls in.

Usage
-----
    # all tasks discovered from configs/*.yaml (the default)
    python scripts/prepare_datasets.py

    # a specific split / dataset / output dir
    python scripts/prepare_datasets.py --split test
    python scripts/prepare_datasets.py --dataset Salesforce/FaithEval-unanswerable-v1.0
    python scripts/prepare_datasets.py --output-dir /path/to/staging/faitheval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "faitheval"

logger = logging.getLogger("prepare_datasets")


def dataset_slug(dataset_name: str) -> str:
    """Local directory name for a Hub dataset id (its last path component)."""
    return dataset_name.rstrip("/").split("/")[-1]


def discover_dataset_names() -> list[str]:
    """Collect the ``dataset_name`` field from every ``configs/*.yaml`` task file."""
    names: list[str] = []
    for cfg in sorted(CONFIG_DIR.glob("*.yaml")):
        with cfg.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        name = raw.get("dataset_name")
        if name and name not in names:
            names.append(name)
    return names


def _find_split_parquet(root: Path, split: str) -> list[Path]:
    """Locate the parquet files belonging to `split` within a snapshot dir."""
    parquet = sorted(root.rglob("*.parquet"))
    if not parquet:
        return []

    def matches(path: Path, name: str) -> bool:
        posix = path.as_posix()
        return (
            path.stem == name
            or path.stem.startswith(f"{name}-")
            or f"/{name}/" in posix
            or f"/{name}-" in posix
        )

    selected = [p for p in parquet if matches(p, split)]
    if selected:
        return selected

    # No split-tagged files: if nothing looks like a *different* split, treat this
    # as a single-split dataset (FaithEval ships only `test`) — use everything.
    known_splits = ("train", "validation", "valid", "dev", "test")
    has_other_split = any(
        matches(p, other) for p in parquet for other in known_splits if other != split
    )
    return [] if has_other_split else parquet


def export_split(dataset_name: str, split: str, output_dir: Path) -> Path:
    """Fetch one split's parquet from the Hub and write it as JSONL under ``output_dir``.

    Reads the parquet with pyarrow directly (never ``load_dataset``) so the local
    ``datasets`` version — and the ``List`` feature metadata the files carry — is
    irrelevant.
    """
    logger.info("Downloading %s from the Hub ...", dataset_name)
    local_dir = snapshot_download(dataset_name, repo_type="dataset")

    files = _find_split_parquet(Path(local_dir), split)
    if not files:
        raise FileNotFoundError(
            f"No parquet files for split {split!r} found under {local_dir!r}."
        )

    # replace_schema_metadata(None) drops the HF feature metadata (the source of
    # the `List` incompatibility); pyarrow reads the raw columns regardless.
    table = pq.read_table([str(f) for f in files]).replace_schema_metadata(None)

    out_path = output_dir / dataset_slug(dataset_name) / f"{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for batch in table.to_batches():
            for record in batch.to_pylist():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples (%d file(s)) -> %s", table.num_rows, len(files), out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="HUB_ID",
        help="Hub dataset id to export (repeatable). Defaults to the dataset_name of every configs/*.yaml.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for the JSONL output (default: <repo>/data/faitheval).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    names = args.datasets or discover_dataset_names()
    if not names:
        parser.error(
            f"No datasets to export: none passed via --dataset and none found in {CONFIG_DIR}."
        )

    logger.info("Exporting %d dataset(s) to %s", len(names), args.output_dir)
    for name in names:
        export_split(name, args.split, args.output_dir)
    logger.info(
        "Done. Copy %s to the cluster (or commit it) so the offline eval can read it.",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
