"""Read-only aggregation + comparison + plotting layer over Eval_master results.

This package post-processes the per-benchmark ``summary.json`` / ``summary.csv``
files that ``run_benchmarks.py`` fans out across the five benchmarks, into a single
long-form data model (:mod:`analysis.model`), aggregates it across seeds
(:mod:`analysis.aggregate`), compares arms under the statistically correct regime
(:mod:`analysis.compare`), and plots (:mod:`analysis.plot`).

It is additive and read-only: it never touches the model-loading seam or the
per-benchmark tokenizer/comparability contract (see ``harness-eval/ARCHITECTURE.md``).

Import-light by design: only :mod:`numpy` is imported eagerly; ``scipy`` and
``matplotlib`` are imported lazily inside the functions that need them, so the
parsing / data layer stays unit-testable offline.
"""

from __future__ import annotations

from analysis.model import MetricRecord, RecordSet, signed_value

__all__ = ["MetricRecord", "RecordSet", "signed_value"]
