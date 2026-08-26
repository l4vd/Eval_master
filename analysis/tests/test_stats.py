"""Vendored stats: bootstrap CI reproducibility + the seed-alignment bug fix."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.stats import (
    SeedMismatchError,
    adjust_pvalues,
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


# --- multiplicity ----------------------------------------------------------------


def test_holm_matches_the_textbook_worked_example():
    """Step-down: adj_(k) = max_{j<=k} (m - j + 1) * p_(j), monotone in the raw order."""
    raw = [0.001, 0.008, 0.039, 0.041, 0.042]
    assert adjust_pvalues(raw, "holm") == pytest.approx([0.005, 0.032, 0.117, 0.117, 0.117])


def test_bh_step_up_uses_the_running_minimum():
    """The three near-0.04 p-values all collapse to 5/5 * 0.042, not to their own m/j."""
    raw = [0.001, 0.008, 0.039, 0.041, 0.042]
    assert adjust_pvalues(raw, "bh") == pytest.approx([0.005, 0.02, 0.042, 0.042, 0.042])


def test_bh_is_never_more_conservative_than_holm():
    raw = [0.001, 0.008, 0.039, 0.041, 0.042]
    holm = adjust_pvalues(raw, "holm")
    bh = adjust_pvalues(raw, "bh")
    assert all(b <= h + 1e-12 for b, h in zip(bh, holm))


def test_adjustment_preserves_input_order_and_is_monotone_in_the_raw_p():
    raw = [0.20, 0.001, 0.05, 0.30, 0.01]
    for method in ("holm", "bh"):
        adj = adjust_pvalues(raw, method)
        assert all(a >= r - 1e-12 for a, r in zip(adj, raw))   # never anti-conservative
        assert all(0.0 <= a <= 1.0 for a in adj)
        ranked = sorted(zip(raw, adj))
        assert all(ranked[i][1] <= ranked[i + 1][1] + 1e-12 for i in range(len(ranked) - 1))


def test_nan_is_excluded_from_the_family_not_counted_in_it():
    """A test that did not run must not inflate m and penalise the ones that did."""
    with_nan = adjust_pvalues([0.01, float("nan"), 0.04], "holm")
    assert np.isnan(with_nan[1])
    # Family size is 2, so the smallest p is multiplied by 2 -- not by 3.
    assert with_nan[0] == pytest.approx(0.02)
    assert with_nan[2] == pytest.approx(0.04)


def test_none_is_the_identity_and_an_unknown_method_raises():
    raw = [0.01, 0.5]
    assert adjust_pvalues(raw, "none") == raw
    with pytest.raises(ValueError, match="Unknown multiplicity method"):
        adjust_pvalues(raw, "bonferroni")


def test_all_nan_family_returns_all_nan():
    out = adjust_pvalues([float("nan"), float("nan")], "bh")
    assert all(np.isnan(x) for x in out)


def test_five_seeds_cannot_reach_significance_even_before_correction():
    """The power wall of analysis_theory.md 5.2: exact two-sided p_min = 2 / 2**n.

    Pinned as a test because it governs how every paired result in this package must be
    read: at n = 5 the Wilcoxon cannot return p < 0.05 for ANY effect size, so an absence
    of stars is a property of the sample size and never evidence of equivalence.
    """
    a = {s: 1.0 for s in range(5)}          # maximally separated: every difference positive
    b = {s: 0.0 for s in range(5)}
    result = wilcoxon_matched(a, b)
    assert result.n_pairs == 5
    assert result.p_value == pytest.approx(2 / 2**5)
    assert result.p_value > 0.05
    assert significance_stars(result.p_value) == "ns"
