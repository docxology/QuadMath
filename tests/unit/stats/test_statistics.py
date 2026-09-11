"""Tests for the deterministic statistics toolkit (quadmath/stats/statistics.py).

All expected values are hand-computed exact cases or calibrated against
fixed seeds — no mocks, no statistical-framework imports.  The resampling
functions must be bit-reproducible for a fixed seed (the manuscript
reproduces these numbers verbatim in section 17).
"""
from __future__ import annotations

import numpy as np
import pytest

from quadmath.stats.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    cohens_d,
    jackknife_ci,
    p_adjust_bonferroni,
    permutation_test,
    rotation_stats,
    scaling_fit,
    summarize,
    welch_t_test,
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


# --------------- jackknife_ci ---------------


def test_jackknife_ci_mean_exact_interval():
    # [1,2,3]: leave-one-out means are 2.5, 2.0, 1.5, so mean(theta_i) = 2.0
    # equals theta_hat = 2.0 and the bias is exactly 0.  se^2 = (2/3) *
    # (0.25 + 0 + 0.25) = 1/3, se = 0.5773503, and z_(0.975) = 1.9599640,
    # giving 2 +/- 1.1315857.
    lo, hi, bias = jackknife_ci([1, 2, 3])
    assert lo == pytest.approx(0.8684143, abs=1e-6)
    assert hi == pytest.approx(3.1315857, abs=1e-6)
    assert bias == 0.0
    assert 0.5 * (lo + hi) == pytest.approx(2.0, abs=1e-12)


def test_jackknife_ci_bias_recovers_unbiased_variance():
    # stat = np.var (ddof=0) on [1,2,3]: theta_hat = 2/3, leave-one-out
    # values 0.25, 1.0, 0.25 with mean 0.5, so bias = 2*(0.5 - 2/3) = -1/3
    # and theta_hat - bias = 1 equals the ddof=1 variance.  se^2 =
    # (2/3)*0.375 = 0.25, so the interval is 2/3 +/- 0.9799820.
    lo, hi, bias = jackknife_ci([1, 2, 3], stat=np.var)
    assert bias == pytest.approx(-1.0 / 3.0, abs=1e-12)
    assert 2.0 / 3.0 - bias == pytest.approx(1.0, abs=1e-12)
    assert lo == pytest.approx(-0.3133153, abs=1e-6)
    assert hi == pytest.approx(1.6466487, abs=1e-6)


def test_jackknife_ci_deterministic_without_seed():
    # No randomness anywhere: two calls return bit-identical triples.
    x = np.arange(1.0, 6.0)
    assert jackknife_ci(x) == jackknife_ci(x)


def test_jackknife_ci_alpha_widens_interval():
    x = np.arange(1.0, 6.0)
    lo5, hi5, _ = jackknife_ci(x, alpha=0.05)
    lo2, hi2, _ = jackknife_ci(x, alpha=0.2)
    assert hi2 - lo2 < hi5 - lo5


def test_jackknife_ci_rejects_empty_nonfinite_short_and_bad_alpha():
    with pytest.raises(ValueError, match="at least one value"):
        jackknife_ci([])
    with pytest.raises(ValueError, match="only finite values"):
        jackknife_ci([1.0, np.nan])
    with pytest.raises(ValueError, match="only finite values"):
        jackknife_ci([1.0, np.inf])
    with pytest.raises(ValueError, match="at least two values"):
        jackknife_ci([1.0])
    with pytest.raises(ValueError, match="alpha must lie"):
        jackknife_ci([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError, match="alpha must lie"):
        jackknife_ci([1.0, 2.0], alpha=1.0)


# --------------- benjamini_hochberg ---------------


def test_bh_known_values():
    # m=3 with sorted p = 0.01, 0.03, 0.04: raw values 0.03, 0.045, 0.04;
    # the backward cumulative minimum makes them 0.03, 0.04, 0.04.
    adj = benjamini_hochberg([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.04, 0.04])


def test_bh_preserves_input_order_and_handles_ties():
    # Tied p-values receive the same adjusted value.
    assert np.allclose(benjamini_hochberg([0.02, 0.02]), [0.02, 0.02])
    # sorted p = 0.01, 0.05, 0.9 -> raw 0.03, 0.075, 0.9; the backward
    # cumulative minimum keeps all three, mapped back to the scrambled
    # input order.
    assert np.allclose(benjamini_hochberg([0.9, 0.01, 0.05]), [0.9, 0.03, 0.075])


def test_bh_monotone_in_sorted_p_and_bounded():
    p = np.array([0.5, 0.02, 0.9, 0.3, 0.02, 1.0])
    adj = benjamini_hochberg(p)
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(adj[order]) >= 0.0)
    assert np.all((adj >= 0.0) & (adj <= 1.0))
    assert adj[-1] == pytest.approx(1.0)  # p = 1.0 stays at exactly 1.0


def test_bh_single_value():
    assert np.allclose(benjamini_hochberg([0.3]), [0.3])
    assert np.allclose(benjamini_hochberg([0.0]), [0.0])


def test_bh_reproducible():
    # Pure function of the input, no seed involved.
    p = np.array([0.01, 0.04, 0.03])
    assert np.array_equal(benjamini_hochberg(p), benjamini_hochberg(p))


def test_bh_rejects_empty_and_out_of_range():
    with pytest.raises(ValueError, match="at least one value"):
        benjamini_hochberg([])
    with pytest.raises(ValueError, match="must all lie in"):
        benjamini_hochberg([-0.1, 0.5])
    with pytest.raises(ValueError, match="must all lie in"):
        benjamini_hochberg([0.5, 1.5])
    with pytest.raises(ValueError, match="must all lie in"):
        benjamini_hochberg([0.5, np.nan])


# --------------- welch_t_test ---------------


def test_welch_identical_arrays_is_exact_null():
    # Equal samples: the mean difference is exactly 0, so t = 0 and the
    # two-sided tail is exactly 1.0; one-sided alternatives split at 0.5.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert welch_t_test(a, a) == (0.0, 1.0)
    assert welch_t_test(a, a, alternative="greater") == (0.0, 0.5)
    assert welch_t_test(a, a, alternative="less") == (0.0, 0.5)


def test_welch_exact_pvalue_at_two_degrees_of_freedom():
    # a = [0, 2] and b = [-sqrt(2), 2 - sqrt(2)] both have variance 2, so
    # wa = wb = 1, t = (1 - (1 - sqrt(2)))/sqrt(2) = 1 exactly, and the
    # Welch-Satterthwaite df collapses to 2.  For df = 2 the two-sided
    # Student-t tail is 1 - t/sqrt(t**2 + 2), i.e. p = 1 - 1/sqrt(3).
    a = np.array([0.0, 2.0])
    b = np.array([-np.sqrt(2.0), 2.0 - np.sqrt(2.0)])
    t, p = welch_t_test(a, b)
    assert t == pytest.approx(1.0, abs=1e-12)
    assert p == pytest.approx(1.0 - 1.0 / np.sqrt(3.0), abs=1e-10)


def test_welch_separated_samples_significant_and_directional():
    # a = [1..4] vs b = [11..14]: equal variances give
    # t = -10/sqrt(5/6) ~ -10.954 with df = 6, deep in the lower tail.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([11.0, 12.0, 13.0, 14.0])
    t, p = welch_t_test(a, b)
    assert t == pytest.approx(-10.0 / np.sqrt(5.0 / 6.0), rel=1e-9)
    assert p < 0.05
    _, p_greater = welch_t_test(a, b, alternative="greater")
    _, p_less = welch_t_test(a, b, alternative="less")
    assert p_greater > 0.99
    assert p_less < 0.01
    assert p_greater + p_less == pytest.approx(1.0, abs=1e-12)
    # Swapping the groups flips the sign and the directional verdict.
    t2, p2 = welch_t_test(b, a, alternative="greater")
    assert t2 > 0.0
    assert p2 < 0.01


def test_welch_constant_samples_degenerate():
    # Zero variance in both samples: equal constants are an exact null;
    # different constants give a signed infinite t with exact 0/1 tails.
    assert welch_t_test(np.full(3, 2.0), np.full(4, 2.0)) == (0.0, 1.0)
    t, p = welch_t_test(np.full(3, 2.0), np.full(4, 5.0))
    assert t == -np.inf
    assert p == 0.0
    assert welch_t_test(np.full(3, 2.0), np.full(4, 5.0), alternative="greater") == (-np.inf, 1.0)
    assert welch_t_test(np.full(3, 2.0), np.full(4, 5.0), alternative="less") == (-np.inf, 0.0)
    assert welch_t_test(np.full(4, 5.0), np.full(3, 2.0), alternative="greater") == (np.inf, 0.0)


def test_welch_reproducible():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert welch_t_test(a, b) == welch_t_test(a, b)


def test_welch_rejects_empty_short_and_unknown_alternative():
    with pytest.raises(ValueError, match="at least one value"):
        welch_t_test([], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least two values per sample"):
        welch_t_test([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least two values per sample"):
        welch_t_test([1.0, 2.0], [3.0])
    with pytest.raises(ValueError, match="unknown alternative"):
        welch_t_test([1.0, 2.0], [3.0, 4.0], alternative="both")


# --------------- rotation_stats ---------------


def test_rotation_stats_constant_angles():
    # All angles equal: the unit vectors align, giving R = 1 and both
    # variance conventions collapse to 0, with the mean equal to the angle.
    s = rotation_stats([0.7, 0.7, 0.7])
    assert s["mean"] == pytest.approx(0.7, abs=1e-12)
    assert s["resultant_length"] == pytest.approx(1.0, abs=1e-12)
    assert s["variance"] == pytest.approx(0.0, abs=1e-12)
    assert s["variance_2pi"] == pytest.approx(0.0, abs=1e-12)


def test_rotation_stats_hand_computed_two_angles():
    # [0, pi/2]: the averaged unit vector is (1/2, 1/2), so R = 1/sqrt(2),
    # the circular mean is pi/4, variance = 1 - 1/sqrt(2), and the 2*pi
    # convention doubles it.
    s = rotation_stats([0.0, np.pi / 2.0])
    assert s["mean"] == pytest.approx(np.pi / 4.0, abs=1e-12)
    assert s["resultant_length"] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-12)
    assert s["variance"] == pytest.approx(1.0 - 1.0 / np.sqrt(2.0), abs=1e-12)
    assert s["variance_2pi"] == pytest.approx(2.0 * (1.0 - 1.0 / np.sqrt(2.0)), abs=1e-12)


def test_rotation_stats_zero_resultant_degenerate():
    # Antipodal angles cancel: R ~ 0, both variances hit their maxima and
    # the mean degenerates to the raw atan2 of the dusty components.
    s = rotation_stats([np.pi / 2.0, 3.0 * np.pi / 2.0])
    assert s["resultant_length"] == pytest.approx(0.0, abs=1e-15)
    assert s["variance"] == pytest.approx(1.0, abs=1e-12)
    assert s["variance_2pi"] == pytest.approx(2.0, abs=1e-12)


def test_rotation_stats_is_wrap_invariant():
    # A full extra turn cannot change any circular statistic.
    base = rotation_stats([1.0, 2.5])
    wrapped = rotation_stats([1.0, 2.5 + 2.0 * np.pi])
    for key in base:
        assert base[key] == pytest.approx(wrapped[key], abs=1e-9)


def test_rotation_stats_reproducible():
    assert rotation_stats([1.0, 2.5]) == rotation_stats([1.0, 2.5])


def test_rotation_stats_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError, match="at least one value"):
        rotation_stats([])
    with pytest.raises(ValueError, match="only finite values"):
        rotation_stats([0.5, np.inf])
    with pytest.raises(ValueError, match="only finite values"):
        rotation_stats([np.nan])