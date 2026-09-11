"""Tests for the deterministic statistics toolkit (src/statistics.py).

All expected values are hand-computed exact cases or calibrated against
fixed seeds — no mocks, no statistical-framework imports.  The resampling
functions must be bit-reproducible for a fixed seed (the manuscript
reproduces these numbers verbatim in section 17).
"""
from __future__ import annotations

import numpy as np
import pytest

from statistics import (
    bootstrap_ci,
    cohens_d,
    p_adjust_bonferroni,
    permutation_test,
    scaling_fit,
    summarize,
)


# --------------- summarize ---------------


def test_summarize_exact_values():
    # Hand-computed: [1,2,3,4] has mean 2.5, median 2.5, sample std
    # sqrt(5/3) = 1.290994..., quartiles 1.75/3.25 (linear interpolation),
    # so IQR = 1.5 exactly.
    s = summarize([1, 2, 3, 4])
    assert s["n"] == 4
    assert s["mean"] == 2.5
    assert s["median"] == 2.5
    assert s["iqr"] == 1.5
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["std"] == pytest.approx(1.2909944, abs=1e-6)


def test_summarize_single_value():
    s = summarize([7.0])
    assert s["n"] == 1
    assert s["mean"] == 7.0
    assert s["median"] == 7.0
    assert np.isnan(s["std"])  # ddof=1 is undefined for one observation


def test_summarize_rejects_empty():
    with pytest.raises(ValueError, match="at least one value"):
        summarize([])


# --------------- bootstrap_ci ---------------


def test_bootstrap_ci_deterministic_for_fixed_seed():
    x = np.arange(1.0, 11.0)
    lo1, hi1 = bootstrap_ci(x, iters=200, seed=42)
    lo2, hi2 = bootstrap_ci(x, iters=200, seed=42)
    assert (lo1, hi1) == (lo2, hi2)


def test_bootstrap_ci_seed_changes_interval():
    x = np.arange(1.0, 11.0)
    assert bootstrap_ci(x, iters=200, seed=0) != bootstrap_ci(x, iters=200, seed=1)


def test_bootstrap_ci_constant_sample_is_degenerate():
    # A constant sample resamples to the same constant, so both interval
    # endpoints collapse to c regardless of seed or iteration count.
    ci = bootstrap_ci(np.full(8, 3.5), iters=50, seed=7)
    assert ci == (3.5, 3.5)


def test_bootstrap_ci_custom_statistic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = bootstrap_ci(x, stat=np.median, iters=100, seed=3)
    # The median of a fixed odd-size sample is the middle element for any
    # resample drawn with replacement... median of 5 draws is at most 5
    # and at least 1, and the interval must bracket the point estimate.
    assert 1.0 <= lo <= hi <= 5.0
    assert lo <= np.median(x) <= hi


def test_bootstrap_ci_rejects_empty_and_bad_iters():
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="iters"):
        bootstrap_ci([1.0, 2.0], iters=0)


# --------------- permutation_test ---------------


def test_permutation_test_separated_samples_significant():
    # Shifted copies of the same seeded sample: every permutation keeps
    # the groups far apart, so only the +1 pseudo-count survives and the
    # two-sided p-value is the floor 1/(iters+1) < 0.01.
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, 20)
    b = a + 10
    p = permutation_test(a, b, iters=200, seed=0)
    assert p < 0.01


def test_permutation_test_reproducible_and_seed_sensitive():
    # Overlapping samples: the exceedance count is mid-range, so it
    # actually depends on the seed (well-separated samples always floor
    # out at the add-one pseudo-count and would look seed-insensitive).
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20)
    b = rng.normal(0.5, 1, 20)
    p1 = permutation_test(a, b, iters=200, seed=5)
    p2 = permutation_test(a, b, iters=200, seed=5)
    p3 = permutation_test(a, b, iters=200, seed=6)
    assert p1 == p2
    assert p1 != p3


def test_permutation_test_alternatives_directional():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, 20)
    b = a + 10
    # The alternative refers to the observed difference mean(a) - mean(b).
    # Here that difference (~ -10) sits in the extreme left tail of the
    # permutation distribution, so "less" is significant while "greater"
    # cannot be rejected (p near 1).
    assert permutation_test(a, b, iters=200, seed=0, alternative="less") < 0.01
    assert permutation_test(a, b, iters=200, seed=0, alternative="greater") > 0.9

def test_permutation_test_bounds():
    # Overlapping identical distributions: p stays strictly inside (0, 1]
    # because of the add-one convention.
    x = np.arange(10.0)
    p = permutation_test(x, x, iters=100, seed=0)
    assert 0.0 < p <= 1.0


def test_permutation_test_rejects_empty_and_unknown_alternative():
    with pytest.raises(ValueError, match="at least one value"):
        permutation_test([], [1.0])
    with pytest.raises(ValueError, match="at least one value"):
        permutation_test([1.0], [])
    with pytest.raises(ValueError, match="unknown alternative"):
        permutation_test([1.0], [2.0], alternative="both")


# --------------- cohens_d ---------------


def test_cohens_d_identical_arrays_is_zero():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(x, x) == 0.0


def test_cohens_d_exact_value():
    # a = [1,2,3], b = [4,5,6]: variances are both 1 (ddof=1), pooled std
    # is 1, mean difference is -3, so d = -3 exactly.
    assert cohens_d(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])) == -3.0


def test_cohens_d_constant_samples_degenerate():
    # Zero pooled spread: equal constants give 0, different constants give
    # signed infinity rather than a nan from 0/0.
    assert cohens_d(np.full(3, 2.0), np.full(4, 2.0)) == 0.0
    assert cohens_d(np.full(3, 2.0), np.full(4, 5.0)) == -np.inf
    assert cohens_d(np.full(3, 5.0), np.full(4, 2.0)) == np.inf


def test_cohens_d_rejects_empty():
    with pytest.raises(ValueError, match="at least one value"):
        cohens_d([], [1.0])


# --------------- p_adjust_bonferroni ---------------


def test_bonferroni_scales_and_caps_at_one():
    p = np.array([0.01, 0.1, 0.9])
    adj = p_adjust_bonferroni(p)
    assert np.allclose(adj, [0.03, 0.3, 1.0])  # 0.9 * 3 caps at 1.0


def test_bonferroni_returns_float_array_and_rejects_empty():
    adj = p_adjust_bonferroni([0.5])
    assert adj.dtype == float and adj[0] == 0.5
    with pytest.raises(ValueError, match="at least one value"):
        p_adjust_bonferroni([])


# --------------- scaling_fit ---------------


def test_scaling_fit_recovers_square_root_exponent():
    sizes = np.array([1.0, 10.0, 100.0])
    times = sizes**0.5
    slope, intercept, r2 = scaling_fit(sizes, times)
    assert abs(slope - 0.5) < 1e-9
    assert abs(intercept) < 1e-9  # log(1) = 0 passes through the origin
    assert abs(r2 - 1.0) < 1e-12


def test_scaling_fit_linear_scaling_recovers_unit_slope():
    sizes = np.array([10.0, 100.0, 1000.0])
    times = 3.0 * sizes
    slope, _, r2 = scaling_fit(sizes, times)
    assert abs(slope - 1.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-12


def test_scaling_fit_rejects_empty_mismatched_and_nonpositive():
    with pytest.raises(ValueError, match="at least one value"):
        scaling_fit([], [])
    with pytest.raises(ValueError, match="same length"):
        scaling_fit([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        scaling_fit([0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        scaling_fit([1.0, 2.0], [-1.0, 2.0])