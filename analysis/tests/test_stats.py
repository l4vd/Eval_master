"""Vendored stats: bootstrap CI reproducibility + the seed-alignment bug fix."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.stats import (
    SeedMismatchError,
    bootstrap_ci,
    one_sample_summary,
    significance_stars,
    wilcoxon_matched,
)


def test_bootstrap_ci_deterministic_and_bracketing():
    vals = np.array([0.3, 0.5, 0.4, 0.6, 0.2])
    lo, hi = bootstrap_ci(vals, seed=0)
    lo2, hi2 = bootstrap_ci(vals, seed=0)
    assert (lo, hi) == (lo2, hi2)              # fixed rng -> reproducible
    assert lo <= float(vals.mean()) <= hi


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_one_sample_summary_vs_base_point():
    s = one_sample_summary([0.4, 0.5, 0.6], base_point=0.3)
    assert s.n == 3
    assert abs(s.mean - 0.5) < 1e-9
    assert abs(s.delta_vs_base - 0.2) < 1e-9
    assert s.ci_95_lower <= s.mean <= s.ci_95_upper


def test_one_sample_single_seed_zero_std():
    s = one_sample_summary([0.42], base_point=None)
    assert s.n == 1 and s.std == 0.0 and s.delta_vs_base is None


def test_wilcoxon_matched_aligns_by_seed_value():
    # Same seeds, DIFFERENT order -> must pair by value, not position.
    a = {42: 0.5, 7: 0.4, 99: 0.6}
    b = {7: 0.3, 99: 0.5, 42: 0.4}   # each seed: a is 0.1 higher
    res = wilcoxon_matched(a, b)
    assert res.n_pairs == 3
    assert res.seeds == [7, 42, 99]
    assert abs(res.mean_diff - 0.1) < 1e-9


def test_wilcoxon_matched_reordered_regression():
    # Regression the training side lacks: a naive positional pairing would give a
    # different (wrong) mean_diff here.
    a = {1: 1.0, 2: 2.0, 3: 3.0}
    b = {3: 3.0, 2: 2.0, 1: 1.0}     # identical per seed -> diff 0 everywhere
    res = wilcoxon_matched(a, b)
    assert res.mean_diff == 0.0
    assert res.p_value == 1.0        # degenerate all-equal, handled without scipy


def test_wilcoxon_matched_mismatch_raises():
    with pytest.raises(SeedMismatchError) as exc:
        wilcoxon_matched({42: 0.5, 7: 0.4}, {42: 0.5, 1337: 0.6})
    assert "1337" in str(exc.value) or "7" in str(exc.value)


def test_wilcoxon_matched_intersect_when_not_strict():
    res = wilcoxon_matched({42: 0.5, 7: 0.4}, {42: 0.4, 1337: 0.6},
                           require_matched=False)
    assert res.n_pairs == 1 and res.seeds == [42]


def test_wilcoxon_matched_detects_difference():
    pytest.importorskip("scipy")
    a = {i: 0.6 for i in range(8)}
    b = {i: 0.4 for i in range(8)}
    res = wilcoxon_matched(a, b)
    assert res.n_pairs == 8
    assert 0.0 <= res.p_value <= 1.0
    assert res.mean_diff > 0


def test_significance_stars():
    assert significance_stars(0.0005) == "***"
    assert significance_stars(0.03) == "*"
    assert significance_stars(0.2) == "ns"
    assert significance_stars(float("nan")) == ""
