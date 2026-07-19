"""Dataset loading helpers for FaithEval tasks.

Loading strategy (default: local, offline, version-agnostic)
------------------------------------------------------------
By default this module loads each task split from a **local, pre-materialised**
copy of the FaithEval datasets that lives under ``data/faitheval/`` in the repo
(one plain JSON Lines file per task/split, e.g.
``data/faitheval/FaithEval-unanswerable-v1.0/test.jsonl``).

This deliberately sidesteps ``datasets.load_dataset`` on the cluster. The HPC
stack pins ``datasets<3`` (see ``pyproject-HPC.toml``), but a Hugging Face arrow
cache written by ``datasets>=4`` encodes list-valued columns with the newer
``List``/``LargeList`` feature type. Under ``datasets<3`` those names resolve to
``typing.List`` (not a dataclass), so reading such a cache dies with
``TypeError: must be called with a dataclass type or instance``. Plain JSONL
carries none of that cache metadata, so these files load on *any* ``datasets``
version and any platform — no Hub access, no re-download, no version coupling.

Populate ``data/faitheval/`` **once, on a machine with internet access** using
``scripts/prepare_datasets.py`` (it downloads each split from the Hub and writes
the JSONL). Then copy the ``data/faitheval/`` directory to the cluster. See the
README ("Preparing the datasets") for the exact steps.

The location can be overridden with the ``FAITHEVAL_DATA_DIR`` environment
variable (useful if you stage the data outside the repo, e.g. on ``$PROJECT``).

Using the framework online instead
----------------------------------
If you have internet access and are **not** constrained to ``datasets<3`` (i.e.
you can run a modern ``datasets``), you can skip the prepare step entirely and
stream the splits straight from the Hub. The original Hub-loading implementation
is preserved, commented out, at the bottom of this module
(``_load_split_from_hub``); point ``_load_split`` at it to restore that
behaviour.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from datasets import Dataset

from faitheval.config import TaskConfig

logger = logging.getLogger(__name__)

# ``src/faitheval/data.py`` -> repo root is two levels up from ``src``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the pre-materialised JSONL splits live. Override with FAITHEVAL_DATA_DIR.
DEFAULT_DATA_DIR = Path(
    os.environ.get("FAITHEVAL_DATA_DIR", _REPO_ROOT / "data" / "faitheval")
)


def load_task_dataset(task_config: TaskConfig, split: str, num_samples: int | None = None) -> Dataset:
    """Load a FaithEval task split, optionally truncated to `num_samples` examples."""
    dataset = _load_split(task_config.dataset_name, split)
    if num_samples is not None:
        num_samples = min(num_samples, len(dataset))
        dataset = dataset.select(range(num_samples))
    logger.info("Loaded %d examples from %s [%s]", len(dataset), task_config.dataset_name, split)
    return dataset


def dataset_slug(dataset_name: str) -> str:
    """Local directory name for a Hub dataset id (its last path component).

    ``"Salesforce/FaithEval-unanswerable-v1.0"`` -> ``"FaithEval-unanswerable-v1.0"``.
    """
    return dataset_name.rstrip("/").split("/")[-1]


def local_split_path(dataset_name: str, split: str, data_dir: Path | None = None) -> Path:
    """Path of the local JSONL file for a given dataset id and split."""
    base = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    return base / dataset_slug(dataset_name) / f"{split}.jsonl"


def _load_split(dataset_name: str, split: str, data_dir: Path | None = None) -> Dataset:
    """Load a single split from its local JSONL file.

    Raises a clear, actionable error if the file has not been materialised yet.
    """
    path = local_split_path(dataset_name, split, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Local FaithEval data for {dataset_name!r} [{split}] not found at {path}. "
            "Materialise it on a machine with internet access via "
            "`python scripts/prepare_datasets.py` (see the README section "
            "'Preparing the datasets'), then copy data/faitheval/ to this machine. "
            "Alternatively set FAITHEVAL_DATA_DIR to where you staged it."
        )

    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"Local dataset file {path} is empty.")
    logger.info("Loaded %s [%s] from local file %s", dataset_name, split, path)
    # Building from a list of dicts re-infers features from the raw Python types,
    # so there is no HF cache metadata to be incompatible with.
    return Dataset.from_list(records)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file into a list of records (blank lines skipped)."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# ORIGINAL online (Hub) loading path — kept for reference.
#
# Use this only with internet access AND a modern, unpinned `datasets` (>=2.20).
# It streams each split straight from the Hugging Face Hub / its arrow cache.
# On the pinned HPC stack (`datasets<3`) reading a cache written by `datasets>=4`
# fails with `TypeError: must be called with a dataclass type or instance`, which
# is exactly why the default path above reads local JSONL instead.
#
# To restore Hub loading, replace the body of `_load_split` with:
#     return _load_split_from_hub(dataset_name, split)
# and add `load_dataset` back to the `datasets` import at the top.
#
#     from datasets import load_dataset
#
#     def _load_split_from_hub(dataset_name: str, split: str) -> Dataset:
#         """Load a split directly from the Hugging Face Hub / local HF cache."""
#         return load_dataset(dataset_name, split=split)
# ---------------------------------------------------------------------------
