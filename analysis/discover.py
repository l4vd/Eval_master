"""Discover run dirs and infer their seeds for an arm.

An *arm* is a set of run dirs (an ensemble of seeds); a *seed* is one run dir. A run
dir is a leaf that contains per-benchmark subfolders (``faitheval/``, ``harness/``, ...)
and/or a ``run_metadata.json``. Seeds are recovered exactly as the training side does
(``run_metadata.json`` ``"seed"`` -> ``seed_<N>`` dir name), extended to also walk:

* an Eval_master ``--multirun`` root: ``outputs/multirun/<date>/<time>/{0,1,2}/``, and
* a training-side group dir: ``<...>_<METHOD>_ensemble/seed_<SEED>/`` (+ an ``ensemble/``
  sibling that is correctly ignored, having no benchmark subfolders).

Unlike the training-side ``_infer_seed``, there is **no** ``hash(name) % 100000``
fallback: when a seed cannot be determined we return ``None`` (a fixed point), and
paired-mode code refuses to fabricate a seed identity (see :mod:`analysis.compare`).
"""

from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

from analysis.parse import PARSERS

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def infer_seed(run_dir: Path) -> int | None:
    """Seed from ``run_metadata.json`` ``"seed"``, else a ``seed_<N>`` dir name, else None."""
    run_dir = Path(run_dir)
    meta = run_dir / "run_metadata.json"
    if meta.exists():
        try:
            seed = json.loads(meta.read_text(encoding="utf-8")).get("seed")
        except (OSError, ValueError):
            seed = None
        if seed is not None:
            return int(seed)
    m = _SEED_DIR_RE.match(run_dir.name)
    if m:
        return int(m.group(1))
    return None


def is_run_dir(path: Path) -> bool:
    """A leaf run dir carries run_metadata.json or at least one benchmark subfolder."""
    if (path / "run_metadata.json").exists():
        return True
    return any((path / bench).is_dir() for bench in PARSERS)


def expand_spec(spec: str) -> list[Path]:
    """Resolve one arm spec (a dir, a group/multirun dir, or a glob) to run dirs.

    A container dir (group dir, multirun root) is expanded to its run-dir children;
    a leaf run dir resolves to itself. Non-run children (e.g. ``ensemble/``) drop out.
    """
    matches = [Path(m) for m in glob(str(spec))]
    if not matches:
        p = Path(spec)
        matches = [p] if p.exists() else []

    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for m in matches:
        if not m.is_dir():
            continue
        if is_run_dir(m):
            _add(run_dirs, seen, m)
            continue
        # Container: pick up run-dir children (seed_*, multirun numeric, or benchmark-bearing).
        for child in sorted(m.iterdir()):
            if not child.is_dir():
                continue
            if is_run_dir(child) or _SEED_DIR_RE.match(child.name) or child.name.isdigit():
                _add(run_dirs, seen, child)
    return run_dirs


def discover_arm(spec: str) -> list[tuple[Path, int | None]]:
    """Resolve an arm spec to ``[(run_dir, seed), ...]`` (seed None = fixed point)."""
    return [(rd, infer_seed(rd)) for rd in expand_spec(spec)]


def _add(run_dirs: list[Path], seen: set[Path], path: Path) -> None:
    rp = path.resolve()
    if rp not in seen:
        seen.add(rp)
        run_dirs.append(path)
