"""Vendored statistics helpers for the eval-analysis layer.

Why vendored rather than imported: the training repo ``SP-DPO-Base`` and this
``Eval_master`` package are separate projects installed in separate virtualenvs,
so there is no clean shared import path. ``bootstrap_ci`` below is copied verbatim
from ``SP-DPO-Base/src/hallu_mitigate/evaluation/metrics.py`` (pure-numpy percentile
bootstrap) so behaviour matches the training-side ensemble numbers exactly.

``wilcoxon_matched`` is a **corrected** rewrite of that repo's
``ensemble.aggregation.paired_comparison``, which paired two arms by *list position*
(``a[:n], b[:n]`` after ``n = min(sizes)``) with no seed-identity check — silently
mispairing when the two arms' seed sets differ or are ordered differently. Here we
pair by seed **value** and, by default, fail loudly on any seed-set mismatch.

``one_sample_summary`` is new: the one-sample regime (trained ensemble vs. a single
fixed base point) that the training side never needed.

Only :mod:`numpy` is imported eagerly; :mod:`scipy` is imported lazily inside
:func:`wilcoxon_matched` so the module stays importable without the stats extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    Verbatim from ``hallu_mitigate.evaluation.metrics.bootstrap_ci`` so the CIs
    reproduce the training-side ensemble figures.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


@dataclass
class OneSampleSummary:
    """One arm's across-seed summary, reported against a single fixed reference point."""

    n: int
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    base_point: float | None = None
    delta_vs_base: float | None = None  # mean - base_point (raw, not direction-signed)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def one_sample_summary(
    values, base_point: float | None = None, *, seed: int = 0
) -> OneSampleSummary:
    """Mean +/- std + bootstrap CI over per-seed ``values``, plus delta vs a fixed point.

    This is the correct regime when comparing a multi-seed ensemble against a
    *constant* (e.g. an untrained base evaluated once): the base has zero seed
    variance, so a paired test is degenerate. We instead characterise the arm and
    report its distance from the base's single point.
    """
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return OneSampleSummary(0, float("nan"), float("nan"), float("nan"),
                                float("nan"), base_point, None)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0
    lo, hi = bootstrap_ci(arr, seed=seed)
    delta = (mean - base_point) if base_point is not None else None
    return OneSampleSummary(n, mean, std, lo, hi, base_point, delta)


@dataclass
class PairedResult:
    """Seed-aligned paired comparison of two arms (regime: arm-vs-arm)."""

    n_pairs: int
    seeds: list[int]
    mean_diff: float  # mean(a - b) over matched seeds (raw values)
    statistic: float
    p_value: float
    ci_95_lower: float = field(default=float("nan"))  # paired bootstrap CI of the diffs
    ci_95_upper: float = field(default=float("nan"))
    #: Why ``statistic``/``p_value`` are NaN, when they are. ``None`` means the test ran.
    #: Without it a declined test and an uncomputed one both render as an empty cell.
    note: str | None = None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


class SeedMismatchError(ValueError):
    """Raised when two arms do not share an identical seed set in paired mode."""


def wilcoxon_matched(
    a_by_seed: dict[int, float],
    b_by_seed: dict[int, float],
    *,
    require_matched: bool = True,
    seed: int = 0,
) -> PairedResult:
    """Wilcoxon signed-rank + paired bootstrap over **seed-aligned** values.

    Args:
        a_by_seed / b_by_seed: seed -> scalar value for each arm.
        require_matched: if True (default), raise :class:`SeedMismatchError` unless
            the two seed sets are identical. If False, silently intersect and test
            over the shared seeds (still aligned by value, never by position).

    Pairs are formed by seed **value**, fixing the training-side positional-pairing
    bug. Returns statistic/p-value (lazy scipy; degenerate all-equal -> stat 0, p 1),
    the mean paired difference ``a - b``, and a paired bootstrap CI of the differences.
    """
    seeds_a, seeds_b = set(a_by_seed), set(b_by_seed)
    if require_matched and seeds_a != seeds_b:
        only_a = sorted(seeds_a - seeds_b)
        only_b = sorted(seeds_b - seeds_a)
        raise SeedMismatchError(
            "Paired comparison requires identical seed sets. "
            f"Only in A: {only_a}; only in B: {only_b}. "
            "Re-run the missing seeds, or pass require_matched=False to intersect."
        )

    shared = sorted(seeds_a & seeds_b)
    a = np.array([a_by_seed[s] for s in shared], dtype=float)
    b = np.array([b_by_seed[s] for s in shared], dtype=float)
    n = len(shared)
    diffs = a - b
    mean_diff = float(np.mean(diffs)) if n else float("nan")

    result = PairedResult(
        n_pairs=n, seeds=shared, mean_diff=mean_diff,
        statistic=float("nan"), p_value=float("nan"),
    )
    if n < 1:
        result.note = "no shared seeds"
        return result

    lo, hi = bootstrap_ci(diffs, seed=seed)
    result.ci_95_lower, result.ci_95_upper = lo, hi

    # Degenerate: no differences to rank. Return the conventional (0, 1) before the
    # scipy import so this stays correct without scipy installed.
    if np.allclose(a, b):
        result.statistic, result.p_value = 0.0, 1.0
        return result

    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(a, b)
        result.statistic, result.p_value = float(stat), float(p)
    except ImportError:
        # Recorded rather than swallowed: an absent p-value must not read as a
        # non-significant one, and "scipy missing" and "scipy declined this sample" call
        # for different fixes. `significance_stars(nan)` renders both as an empty cell.
        result.note = "scipy not installed (pip install 'eval-master[stats]')"
    except ValueError as exc:
        result.note = f"wilcoxon declined this sample: {exc}"
    return result


# --- multiplicity ---------------------------------------------------------------

#: Adjustment procedures :func:`adjust_pvalues` knows. ``"none"`` is offered explicitly so a
#: caller can turn correction off by configuration rather than by skipping the call, which
#: keeps the recorded provenance honest ("corrected with: none" beats a silently absent key).
MC_METHODS = ("holm", "bh", "none")

#: Default procedure. Holm rather than Bonferroni (uniformly more powerful, same
#: assumptions) and rather than BH, because the headline claim is "this method beats the
#: baseline", where one spurious win costs more than one missed one. Use "bh" for a
#: screening table where a known share of false positives is acceptable.
DEFAULT_MC_METHOD = "holm"


def adjust_pvalues(p_values, method: str = "holm") -> list[float]:
    """Multiplicity-adjusted p-values, returned in the input order.

    A comparison grid of ``(arms - 1) x primary keys`` runs tens of simultaneous tests
    against one reference, so an uncorrected 0.05 threshold does not control anything at the
    family level. Both standard procedures are offered because they control different
    things and the right choice depends on the claim:

    * ``"holm"`` — Holm (1979) step-down; controls the **family-wise error rate**, i.e. the
      probability of *any* false positive. Use when a single spurious win would undermine
      the conclusion. Uniformly more powerful than plain Bonferroni and just as assumption-
      free, so there is no reason to prefer Bonferroni here.
    * ``"bh"`` — Benjamini-Hochberg (1995) step-up; controls the **false discovery rate**,
      i.e. the expected *proportion* of false positives among the rejections. Use for a
      screening table where a known fraction of false positives is tolerable.
    * ``"none"`` — returned unchanged.

    ``NaN`` entries (a test that did not run — no shared seeds, SciPy absent, SciPy
    declined the sample) are **excluded from the family** and returned as ``NaN``. Counting
    them would inflate ``m`` and penalise the tests that *did* run for the existence of
    tests that did not.

    Args:
        p_values: Raw two-sided p-values, in any order; ``NaN`` for "not tested".
        method: One of :data:`MC_METHODS`.

    Returns:
        Adjusted p-values, aligned with the input. ``"holm"`` and ``"bh"`` clip to
        ``[0, 1]``; ``"none"`` returns the inputs unchanged.

    Raises:
        ValueError: On an unknown ``method`` — a typo must not silently mean "no correction".
    """
    if method not in MC_METHODS:
        raise ValueError(f"Unknown multiplicity method {method!r}; expected one of {MC_METHODS}.")
    raw = [float(p) for p in p_values]
    if method == "none":
        return raw

    idx = [i for i, p in enumerate(raw) if p == p]  # drops NaN
    m = len(idx)
    out = list(raw)
    if m == 0:
        return out

    order = sorted(idx, key=lambda i: raw[i])
    if method == "holm":
        # Step-down: adj_(k) = max_{j<=k} (m - j + 1) * p_(j), enforced monotone by the
        # running max -- without it an adjusted p could fall below an earlier, smaller one.
        running = 0.0
        for rank, i in enumerate(order):  # rank is 0-based
            running = max(running, (m - rank) * raw[i])
            out[i] = min(1.0, running)
    else:  # "bh"
        # Step-up: adj_(k) = min_{j>=k} (m / j) * p_(j), so the sweep runs from the largest
        # p downwards and the running min supplies the monotonicity.
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, m / (rank + 1) * raw[i])
            out[i] = min(1.0, running)
    return out


def significance_stars(p_value: float) -> str:
    """Conventional significance annotation used in the paired-delta plot."""
    if p_value != p_value:  # NaN
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"
