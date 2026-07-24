"""Shared test fixtures. Offline only: no network, no model loads, no heavy imports.

scipy/matplotlib are optional; tests that need them use ``importorskip``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the `analysis` package importable when pytest is invoked from anywhere.
_EVAL_MASTER_ROOT = Path(__file__).resolve().parents[2]
if str(_EVAL_MASTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_MASTER_ROOT))

from analysis import fixtures  # noqa: E402


@pytest.fixture
def make_run(tmp_path):
    """Factory: write a synthetic run dir under tmp_path and return its Path."""
    def _make(name: str, *, seed, **benchmarks) -> Path:
        return fixtures.write_full_run(tmp_path / name, seed=seed, **benchmarks)

    return _make
