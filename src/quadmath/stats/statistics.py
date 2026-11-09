"""Deterministic descriptive statistics and resampling toolkit.

Small NumPy/standard-library methods shared by benchmarking and graphics.
Samples are finite one-dimensional float arrays. Means use compensated
summation with an exact fallback only when the intermediate sum overflows;
moments use local centered scales rather than squaring raw observations.
Resampling consumes only an explicitly seeded generator, never a 2-D batch
callback, and rejects non-finite statistic results. Identical arguments and
seed reproduce numerical results within the same toolchain, not image bytes
or results across dependency versions. Public results are floats or arrays.

- :func:`summarize` — n, mean, sample std (ddof=1), median, IQR, min, max.
- :func:`sample_mean` — range-safe mean of finite observations.
- :func:`bootstrap_ci` — seeded percentile bootstrap confidence interval
  for an arbitrary statistic.
- :func:`permutation_test` — seeded pooled permutation test on the mean
  difference between two samples (two-sided / greater / less).
- :func:`cohens_d` — pooled-standard-deviation effect size between two
  samples.
- :func:`p_adjust_bonferroni` — Bonferroni correction of p-values with a
  cap at 1.0.
- :func:`scaling_fit` — ordinary least squares on log-log data returning
  the power-law exponent (slope), intercept, and R^2.
- :func:`scaling_predict` — anchored power-law predictions without reconstructing
  an ill-conditioned global intercept.
- :func:`jackknife_ci` — leave-one-out jackknife confidence interval and
  bias estimate for an arbitrary statistic.
- :func:`benjamini_hochberg` — Benjamini-Hochberg FDR step-up adjusted
  p-values, monotone and capped at 1.0.
- :func:`welch_t_test` — Welch's unequal-variance t test with exact
  Student-t p-values from a continued-fraction incomplete beta function.
- :func:`rotation_stats` — circular mean, mean resultant length, and
  circular variance of angles in radians.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Callable, Dict, Tuple

import numpy as np

__all__ = [
    "benjamini_hochberg",
    "bootstrap_ci",
    "cohens_d",
    "jackknife_ci",
    "p_adjust_bonferroni",
    "permutation_test",
    "rotation_stats",
    "sample_mean",
    "scaling_fit",
    "scaling_predict",
    "summarize",
    "welch_t_test",
]


def _as_float_array(x, name: str) -> np.ndarray:
    """Convert an array-like to a float array, rejecting empty input."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def _exact_mean(values: np.ndarray) -> Fraction:
    """Mean of the represented floats, without an intermediate float sum."""
    return sum(
        (Fraction.from_float(float(value)) for value in values), Fraction()
    ) / values.size


def _mean_finite(values: np.ndarray) -> float:
    """Compensated finite mean; exact arithmetic is an overflow-only path."""
    try:
        return math.fsum(values) / values.size
    except OverflowError:
        return float(_exact_mean(values))


def sample_mean(x: np.ndarray) -> float:
    """Mean of a nonempty finite 1-D sample without overflowing its sum.

    A finite sample's mean is within its range. ``math.fsum`` retains small
    cancellation residuals; an exact represented-value sum handles the rare
    case where even compensated summation has an overflowing intermediate.
    Inputs are not modified. Invalid/non-vector samples raise ``ValueError``.
    """
    return _mean_finite(_as_finite_float_array(x, "x"))


def _linear_quantiles(values: np.ndarray, probabilities: tuple) -> tuple:
    """Linear order-statistic interpolation without overflowing endpoint gaps."""
    positions = [(values.size - 1) * float(probability) for probability in probabilities]
    lower = [math.floor(position) for position in positions]
    upper = [min(index + 1, values.size - 1) for index in lower]
    ordered = np.partition(values, sorted(set(lower + upper)))
    result = []
    for position, left, right in zip(positions, lower, upper):
        weight = Fraction.from_float(position - left)
        lo = Fraction.from_float(float(ordered[left]))
        hi = Fraction.from_float(float(ordered[right]))
        result.append(float(lo + weight * (hi - lo)))
    return tuple(result)


def _centered_values(values: np.ndarray) -> tuple:
    """Return origin, scale, and owned values in a bounded local frame."""
    lo, hi = float(np.min(values)), float(np.max(values))
    origin = lo if lo == hi else 0.5 * lo + 0.5 * hi
    normalized = values - origin
    scale = float(np.max(np.abs(normalized)))
    if scale:
        normalized /= scale
    return origin, scale, normalized


@dataclass(frozen=True, slots=True)
class _SampleMoments:
    """Mean/variance in ``(x-origin)/scale``; constant samples have scale zero."""

    origin: float
    scale: float
    mean: float
    variance: float


def _sample_moments(values: np.ndarray) -> _SampleMoments:
    origin, scale, normalized = _centered_values(values)
    mean = _mean_finite(normalized)
    centered = normalized - mean
    variance = (
        float(np.dot(centered, centered) / (values.size - 1))
        if values.size > 1 else float("nan")
    )
    return _SampleMoments(origin, scale, mean, variance)


def _two_sample_frame(a: np.ndarray, b: np.ndarray) -> tuple:
    """Mean difference and variances in a shared within-sample scale."""
    left, right = _sample_moments(a), _sample_moments(b)
    scale = max(left.scale, right.scale)
    if scale == 0.0:
        return left.origin - right.origin, 0.0, 0.0, scale
    # Same-sign origins can be subtracted without overflow, retaining offsets.
    # Opposite-sign origins are divided first, avoiding an overflowing gap.
    if (left.origin < 0.0) == (right.origin < 0.0):
        base = (left.origin - right.origin) / scale
    else:
        base = left.origin / scale - right.origin / scale
    ratio_a, ratio_b = left.scale / scale, right.scale / scale
    difference = math.fsum((base, ratio_a * left.mean, -ratio_b * right.mean))
    return (
        difference,
        left.variance * ratio_a * ratio_a,
        right.variance * ratio_b * ratio_b,
        scale,
    )


def summarize(x: np.ndarray) -> Dict[str, float]:
    """Descriptive summary of a sample.

    Returns a dict with integer ``n`` and float entries ``mean``,
    ``std`` (sample standard deviation, ``ddof=1``), ``median``, ``iqr``
    (75th minus 25th percentile), ``min``, and ``max``.  For a single
    observation ``std`` is NaN by definition, returned without a warning.

    Raises
    ------
    ValueError
        If ``x`` is empty, non-vector, or contains non-finite values.
    """
    x = _as_finite_float_array(x, "x")
    q25, median, q75 = _linear_quantiles(x, (0.25, 0.5, 0.75))
    moments = _sample_moments(x)
    std = moments.scale * math.sqrt(moments.variance) if x.size > 1 else float("nan")
    return {
        "n": int(x.size),
        "mean": _mean_finite(x),
        "std": std,
        "median": median,
        "iqr": q75 - q25,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable = sample_mean,
    *,
    iters: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple:
    """Percentile bootstrap confidence interval for ``stat`` on ``x``.

    Draws ``iters`` resamples of size ``len(x)`` with replacement from a
    single seeded generator (``np.random.default_rng(seed)``), applies
    ``stat`` to each resample, and returns the ``alpha``/2 and
    1-``alpha``/2 percentiles of the bootstrap distribution as a
    ``(low, high)`` tuple. ``stat`` receives one 1-D float array per call
    and must return a finite real scalar; it is never called on a 2-D batch.
    Resamples stream through one generator, using ``O(len(x) + iters)``
    memory rather than retaining an ``iters``-by-``len(x)`` index matrix.

    Raises
    ------
    ValueError
        If ``x`` is empty, non-vector or non-finite, ``iters`` is not
        positive, ``alpha`` is outside ``(0, 1)``, or ``stat`` is not finite
        and scalar.
    """
    x = _as_finite_float_array(x, "x")
    if iters <= 0:
        raise ValueError("iters must be a positive integer")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    n = len(x)
    rng = np.random.default_rng(seed)
    stats = np.empty(iters, dtype=float)
    for index in range(iters):
        stats[index] = _finite_statistic(stat, x[rng.integers(0, n, size=n)])
    return _linear_quantiles(stats, (alpha / 2, 1 - alpha / 2))


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    iters: int = 2000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> float:
    """Pooled permutation test on the difference of sample means.

    The samples are pooled, then for each of ``iters`` iterations a fresh
    permutation of the pool (drawn from one seeded generator created
    before the loop) is split back into groups of the original sizes and
    the mean difference is recorded.  The p-value follows the standard
    add-one convention ``(# exceedances + 1) / (iters + 1)``, which keeps
    the result strictly positive and attainable.  ``alternative`` selects
    the tail: ``"two-sided"`` counts permutations whose absolute mean
    difference reaches ``abs(obs)``, ``"greater"`` / ``"less"`` use the
    signed comparison against the observed difference. Comparisons use a
    centered, normalized pool and ``1e-12 * ptp(pool)`` as a rounding-level
    tie tolerance. A shared offset or positive change of units therefore
    cannot turn distinct partitions into ties; a constant pool is an exact null.

    Raises
    ------
    ValueError
        If either sample is empty, non-vector or non-finite, ``iters`` is
        not positive, or ``alternative`` is not one of ``"two-sided"``,
        ``"greater"``, ``"less"``.
    """
    a = _as_finite_float_array(a, "a")
    b = _as_finite_float_array(b, "b")
    na = len(a)
    if iters <= 0:
        raise ValueError("iters must be a positive integer")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    _, _, pool = _centered_values(np.concatenate([a, b]))
    obs = _mean_finite(pool[:na]) - _mean_finite(pool[na:])
    tie = 1e-12 * float(np.ptp(pool))
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(iters, dtype=float)
    for i in range(iters):
        perm = rng.permutation(pool)
        perm_diffs[i] = _mean_finite(perm[:na]) - _mean_finite(perm[na:])
    if alternative == "two-sided":
        count = np.count_nonzero(np.abs(perm_diffs) >= abs(obs) - tie)
    elif alternative == "greater":
        count = np.count_nonzero(perm_diffs >= obs - tie)
    else:
        count = np.count_nonzero(perm_diffs <= obs + tie)
    return float((count + 1) / (iters + 1))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-standard-deviation Cohen's d between two samples.

    ``d = (mean(a) - mean(b)) / s_pooled`` where ``s_pooled`` is the
    square root of the degrees-of-freedom-weighted average of the two
    sample variances (``ddof=1``).  If the pooled standard deviation is
    zero (both samples constant) the result is ``0.0`` for equal means
    and signed infinity otherwise.

    Raises
    ------
    ValueError
        If either sample is empty or contains non-finite values, or has
        fewer than two values.
    """
    a = _as_finite_float_array(a, "a")
    b = _as_finite_float_array(b, "b")
    if a.size < 2 or b.size < 2:
        raise ValueError("cohens_d requires at least two values per sample")
    na, nb = a.size, b.size
    diff, va, vb, _ = _two_sample_frame(a, b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0.0:
        if diff == 0.0:
            return 0.0
        return math.copysign(math.inf, diff)
    return diff / pooled


def p_adjust_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Bonferroni-adjusted p-values, elementwise ``min(1, p * m)``.

    Parameters
    ----------
    pvals:
        Raw p-values; the family size ``m`` is ``pvals.size``.

    Raises
    ------
    ValueError
        If ``pvals`` is empty or any value lies outside ``[0, 1]`` (NaN
        fails the range check).
    """
    p = _as_float_array(pvals, "pvals")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("pvals must all lie in [0, 1]")
    m = p.size
    return np.minimum(1.0, p * m)


def _relative_logs(values: np.ndarray, reference: float) -> np.ndarray:
    """Log ratios, retaining distinct adjacent floats near the reference."""
    differences = values - reference
    near = (values >= 0.5 * reference) & (differences <= reference)
    result = np.empty(values.size, dtype=float)
    result[near] = np.log1p(differences[near] / reference)
    result[~near] = np.log(values[~near]) - math.log(reference)
    return result


@dataclass(frozen=True, slots=True)
class _ScalingModel:
    """Anchored log regression; no references to mutable caller arrays."""

    slope: float
    r2: float
    size_reference: float
    time_reference: float
    offset: float

    @property
    def intercept(self) -> float:
        return (
            math.log(self.time_reference) + self.offset
            - self.slope * math.log(self.size_reference)
        )

    def predict(self, sizes: np.ndarray) -> np.ndarray:
        sizes = _as_finite_float_array(sizes, "query_sizes")
        if np.any(sizes <= 0):
            raise ValueError("query_sizes must be positive")
        factors = self.slope * _relative_logs(sizes, self.size_reference) + self.offset
        mantissa, exponent = math.frexp(self.time_reference)
        result = np.empty(sizes.size, dtype=float)
        for index, factor in enumerate(factors):
            factor = float(factor)
            try:
                power = math.trunc(factor / _LN2)
                residual = factor - power * _LN2
                value = math.ldexp(
                    mantissa + mantissa * math.expm1(residual), exponent + power
                )
            except OverflowError as error:
                raise ValueError("Scaling predictions are outside the floating range") from error
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("Scaling predictions are outside the floating range")
            result[index] = value
        return result


def _fit_scaling(sizes: np.ndarray, times: np.ndarray) -> _ScalingModel:
    sizes = _as_finite_float_array(sizes, "sizes")
    times = _as_finite_float_array(times, "times")
    if sizes.size != times.size:
        raise ValueError("sizes and times must have the same length")
    if sizes.size < 2:
        raise ValueError("scaling_fit requires at least two observations")
    if np.any(sizes <= 0) or np.any(times <= 0):
        raise ValueError("sizes and times must be positive for a log-log fit")
    anchor = int(np.argmax(times))
    size_reference, time_reference = float(sizes[anchor]), float(times[anchor])
    log_sizes = _relative_logs(sizes, size_reference)
    log_times = _relative_logs(times, time_reference)
    size_center, time_center = _mean_finite(log_sizes), _mean_finite(log_times)
    centered_sizes, centered_times = log_sizes - size_center, log_times - time_center
    scale = float(np.max(np.abs(centered_sizes)))
    if scale == 0.0:
        raise ValueError("scaling_fit requires at least two distinct sizes")
    normalized_sizes = centered_sizes / scale
    time_scale = float(np.max(np.abs(centered_times)))
    if time_scale:
        normalized_times = centered_times / time_scale
        slope = float(
            np.dot(normalized_sizes, normalized_times)
            / np.dot(normalized_sizes, normalized_sizes) * (time_scale / scale)
        )
        # Average pairwise residuals rather than subtracting two large means.
        # This does not force the fit through the reference observation.
        offset = _mean_finite(log_times - slope * log_sizes)
        residual = (log_times - slope * log_sizes - offset) / time_scale
        r2 = 1.0 - float(
            np.dot(residual, residual) / np.dot(normalized_times, normalized_times)
        )
    else:
        slope, offset, r2 = 0.0, 0.0, float("nan")
    return _ScalingModel(slope, r2, size_reference, time_reference, offset)


def scaling_fit(sizes: np.ndarray, times: np.ndarray) -> tuple:
    """Return ``(slope, intercept, r2)`` for a power-law log-space OLS fit.

    The kernel centers log ratios to represented-value references, using
    ``log1p`` near each reference so distinct adjacent sizes stay distinct.
    ``r2`` is NaN for constant times. The slope describes the supplied
    measurements, not an algorithmic complexity proof.

    The global intercept is a reporting coefficient, not a stable prediction
    representation for narrowly spaced extreme sizes. Use :func:`scaling_predict`
    to evaluate the same anchored model without cancellation.

    Raises ``ValueError`` for empty/non-vector/non-finite arrays, unequal lengths,
    fewer than two observations, non-positive values, or no distinct sizes.
    """
    model = _fit_scaling(sizes, times)
    return model.slope, model.intercept, model.r2


def scaling_predict(
    sizes: np.ndarray, times: np.ndarray, query_sizes: np.ndarray
) -> np.ndarray:
    """Predict positive runtimes from the same anchored fit as ``scaling_fit``.

    The positive finite query vector is evaluated relative to the fit's size
    and time references. Binary exponent decomposition avoids overflowing an
    intermediate exponential or losing adjacent-float variation in a global
    log intercept. Returns an independently owned float array.

    Invalid fit/query data or unrepresentable positive predictions raise
    ``ValueError``; extrapolation does not imply a complexity guarantee.
    """
    return _fit_scaling(sizes, times).predict(query_sizes)


_SQRT2 = math.sqrt(2.0)
_LN2 = math.log(2.0)
_FPMIN = 1e-300
_BETA_EPS = 1e-14
_BETA_MAX_ITERS = 200


def _as_finite_float_array(x, name: str) -> np.ndarray:
    """Convert a nonempty finite 1-D sample without modifying caller data."""
    arr = _as_float_array(x, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _finite_statistic(stat: Callable, values: np.ndarray) -> float:
    """Apply a scalar statistic without accepting undefined resampling results."""
    result = np.asarray(stat(values))
    if result.ndim != 0 or np.iscomplexobj(result):
        raise ValueError("stat must return a finite real scalar")
    value = float(result)
    if not math.isfinite(value):
        raise ValueError("stat must return a finite real scalar")
    return value


def _normal_z_from_two_sided_alpha(alpha: float) -> float:
    """Normal quantile ``z`` with ``P(|Z| > z) = alpha``.

    ``math.erfc(z / sqrt(2))`` decreases monotonically from 2 to 0 as
    ``z`` runs from ``-inf`` to ``inf``, so bisecting that single
    expression pins ``z`` to double precision deterministically, with no
    seed and no lookup table.
    """
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid / _SQRT2) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _reg_inc_beta(
    x: float, a: float, b: float, *, log_x: float
) -> float:
    """Regularized beta evaluated from the Student-t argument and its log.

    Lentz's modified continued fraction in its convergent argument region.
    The Student-t caller selects the complement without cancelling tiny tails;
    the supplied logarithm retains arguments below the float range.
    """
    log_front = (
        a * log_x
        + b * math.log1p(-x)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    front = math.exp(log_front)
    d = 1.0 - (a + b) * x / (a + 1.0)
    d = math.copysign(max(abs(d), _FPMIN), d)
    d = 1.0 / d
    f = d
    c = 1.0
    delta = 0.0
    m = 0
    while m < _BETA_MAX_ITERS and abs(delta - 1.0) >= _BETA_EPS:
        m += 1
        num = m * (b - m) * x / ((a - 1.0 + 2.0 * m) * (a + 2.0 * m))
        val = 1.0 + num * d
        d = math.copysign(max(abs(val), _FPMIN), val)
        val = 1.0 + num / c
        c = math.copysign(max(abs(val), _FPMIN), val)
        d = 1.0 / d
        f *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 1.0 + 2.0 * m))
        val = 1.0 + num * d
        d = math.copysign(max(abs(val), _FPMIN), val)
        val = 1.0 + num / c
        c = math.copysign(max(abs(val), _FPMIN), val)
        d = 1.0 / d
        delta = d * c
        f *= delta
    return front * f / a


def _t_two_sided_tail(
    t: float, df: float, *, log_abs_t: float | None = None
) -> float:
    """Student-t tail with convergence-aware beta/complement selection.

    Squaring a large t can underflow the beta argument even when its square
    root (and hence a low-df probability) is representable. Near zero, use
    the complementary argument so rounding ``1/(1+t*t/df)`` to one does not
    discard a representable departure from the exact null.
    """
    if t == 0.0:
        return 1.0
    if math.isinf(t) and log_abs_t is None:
        return 0.0
    log_ratio = (
        math.log(abs(t)) if log_abs_t is None else log_abs_t
    ) - 0.5 * math.log(df)
    if log_ratio >= 0.0:
        log_x = -2.0 * log_ratio - math.log1p(math.exp(-2.0 * log_ratio))
        return _reg_inc_beta(math.exp(log_x), 0.5 * df, 0.5, log_x=log_x)
    ratio_squared = math.exp(2.0 * log_ratio)
    log_y = 2.0 * log_ratio - math.log1p(ratio_squared)
    # I_y is safe to complement only in the near-null convergence region,
    # t² < 3*df/(df+2). The weaker |t| < sqrt(df) condition loses tiny tails.
    if abs(t) < math.sqrt(3.0 / (1.0 + 2.0 / df)):
        return 1.0 - _reg_inc_beta(
            math.exp(log_y), 0.5, 0.5 * df, log_x=log_y
        )
    log_x = -math.log1p(ratio_squared)
    return _reg_inc_beta(math.exp(log_x), 0.5 * df, 0.5, log_x=log_x)


def _t_sf(t: float, df: float, *, log_abs_t: float | None = None) -> float:
    """Survival function, retaining representable small positive tails."""
    if t > 0.0:
        return 0.5 * _t_two_sided_tail(t, df, log_abs_t=log_abs_t)
    if t < 0.0:
        return 1.0 - 0.5 * _t_two_sided_tail(t, df, log_abs_t=log_abs_t)
    return 0.5


def jackknife_ci(
    x: np.ndarray,
    stat: Callable = sample_mean,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Leave-one-out jackknife interval and bias estimate for ``stat``.

    Each of the ``n`` leave-one-out samples is scored with ``stat``; the
    returned triple is ``(low, high, bias)``.  The interval is the normal
    approximation ``theta_hat +/- z_(1 - alpha/2) * se_jack`` centered on
    the full-sample statistic, where ``se_jack`` is the jackknife
    standard error ``sqrt((n - 1) / n * sum((theta_i -
    mean(theta_i))**2))``.  ``bias`` is the jackknife bias estimate
    ``(n - 1) * (mean(theta_i) - theta_hat)``; subtracting it from
    ``theta_hat`` gives the bias-corrected jackknife estimate.  The
    quantile ``z`` comes from a deterministic erfc bisection, so the
    routine consumes no randomness and repeated calls return
    bit-identical results.  ``stat`` must accept a 1-D float array and
    return a finite real scalar.

    Parameters
    ----------
    x:
        Sample values; at least two finite observations are required.
    stat:
        Applied to one 1-D float array per call (each leave-one-out sample)
        and must return a scalar; the same contract as :func:`bootstrap_ci`.
    alpha:
        Tail probability in ``(0, 1)``; the interval covers ``1 - alpha``.

    Returns
    -------
    Tuple[float, float, float]
        ``(low, high, bias)`` as plain Python floats.

    Raises
    ------
    ValueError
        If ``x`` is empty, non-vector, non-finite, has fewer than two
        observations, ``alpha`` is outside ``(0, 1)``, ``stat`` is not finite
        and scalar, or the resulting interval/bias is outside the float range.
    """
    x = _as_finite_float_array(x, "x")
    if x.size < 2:
        raise ValueError("jackknife requires at least two values")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    n = x.size
    theta_hat = _finite_statistic(stat, x)
    loo = np.array([_finite_statistic(stat, np.delete(x, i)) for i in range(n)])
    try:
        bias = float((n - 1) * (_exact_mean(loo) - Fraction.from_float(theta_hat)))
    except OverflowError as error:
        raise ValueError("Jackknife bias is outside the floating range") from error
    moments = _sample_moments(loo)
    se = moments.scale * ((n - 1) / math.sqrt(n) * math.sqrt(moments.variance))
    radius = _normal_z_from_two_sided_alpha(alpha) * se
    low, high = theta_hat - radius, theta_hat + radius
    if not all(math.isfinite(value) for value in (low, high)):
        raise ValueError("Jackknife interval is outside the floating range")
    return low, high, bias


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (step-up procedure).

    Ranks the ``m`` p-values in ascending order, forms the raw values
    ``m * p_(i) / i`` for ranks ``i = 1..m``, enforces monotonicity with
    a running minimum taken from the largest rank backwards, maps the
    adjusted values back to the input order, and caps at 1.0.  The
    result is monotone non-decreasing in the sorted p-values and never
    exceeds the largest raw p-value.

    Parameters
    ----------
    pvals:
        Raw p-values, every element in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the original input order.

    Raises
    ------
    ValueError
        If ``pvals`` is empty or any value lies outside ``[0, 1]`` (NaN
        and infinity fail the range check).
    """
    p = _as_float_array(pvals, "pvals")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("pvals must all lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = np.minimum(1.0, m * p[order] / np.arange(1, m + 1))
    stepped = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m, dtype=float)
    adj[order] = stepped
    return adj


def welch_t_test(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Welch's unequal-variance two-sample t test.

    Returns the Welch t statistic and its p-value under Student's t with
    the Welch-Satterthwaite degrees of freedom
    ``(va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))``.
    The p-value is evaluated from the exact Student-t survival function
    via a continued-fraction incomplete beta function (``math`` only, no
    ``scipy``): ``"two-sided"`` reports ``P(|T| >= |t|)``, ``"greater"``
    reports ``P(T > t)``, and ``"less"`` reports ``P(T <= t)``.  When
    both samples are constant the denominator vanishes: equal means give
    ``t = 0.0`` (p-value 1.0 two-sided) and different means give a
    signed infinite ``t`` whose tail probabilities are exactly 0 or 1,
    evaluated against a finite pooled-df fallback so the tails stay
    well-defined.

    Local moment frames avoid raw squared-variance overflow/underflow.
    Tail evaluation retains its log statistic even if a non-degenerate t
    exceeds the float range: a reported infinite t can then carry a small
    positive representable p-value, unlike a true zero-variance degeneracy.

    Parameters
    ----------
    a, b:
        Samples, each with at least two values.
    alternative:
        ``"two-sided"``, ``"greater"``, or ``"less"``.

    Returns
    -------
    Tuple[float, float]
        The t statistic and its p-value as plain Python floats.

    Raises
    ------
    ValueError
        If either sample is empty, contains non-finite values, or has
        fewer than two values, or ``alternative`` is not one of the three
        supported strings.
    """
    a = _as_finite_float_array(a, "a")
    b = _as_finite_float_array(b, "b")
    if a.size < 2 or b.size < 2:
        raise ValueError("welch_t_test requires at least two values per sample")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    num, va, vb, scale = _two_sample_frame(a, b)
    na, nb = a.size, b.size
    wa, wb = va / na, vb / nb
    variance = wa + wb
    denom = math.sqrt(variance)
    log_abs_t = None
    if denom == 0.0:
        t = 0.0 if num == 0.0 else math.copysign(math.inf, num)
        df = float(na + nb - 2)
    else:
        t = num / denom
        fraction = wa / variance
        df = 1.0 / (fraction * fraction / (na - 1) + (1.0 - fraction) ** 2 / (nb - 1))
        if math.isinf(t):
            if math.isfinite(num):
                log_num = math.log(abs(num))
            else:
                exact = (_exact_mean(a) - _exact_mean(b)) / Fraction.from_float(scale)
                log_num = math.log(abs(exact.numerator)) - math.log(exact.denominator)
            log_abs_t = log_num - math.log(denom)
    if alternative == "two-sided":
        return t, _t_two_sided_tail(t, df, log_abs_t=log_abs_t)
    if alternative == "greater":
        return t, _t_sf(t, df, log_abs_t=log_abs_t)
    return t, _t_sf(-t, df, log_abs_t=log_abs_t)


def rotation_stats(angles: np.ndarray) -> Dict[str, float]:
    """Circular statistics for angles given in radians.

    Returns four plain floats: ``mean`` is the circular mean, the
    ``atan2`` of the averaged sine and cosine wrapped into ``(-pi, pi]``;
    ``resultant_length`` is the mean resultant length
    ``R = |n**-1 * sum(exp(1j * theta))|`` in ``[0, 1]``; ``variance`` is
    the unitless circular variance ``1 - R`` in ``[0, 1]``; and
    ``variance_2pi`` is ``2 * (1 - R)``, the convention for angles on the
    full ``[0, 2*pi)`` circle, which ranges over ``[0, 2]`` and
    approaches the linear variance for tightly clustered angles.  Because
    sine and cosine are ``2*pi``-periodic, wrapping the angles into any
    full circle leaves every value unchanged; when the resultant
    vanishes the mean degenerates to whatever ``atan2`` returns for the
    cancelled components.

    Parameters
    ----------
    angles:
        Angles in radians; at least one finite value is required.

    Returns
    -------
    Dict[str, float]
        Keys ``mean``, ``resultant_length``, ``variance``,
        ``variance_2pi``.

    Raises
    ------
    ValueError
        If ``angles`` is empty or contains non-finite values.
    """
    angles = _as_finite_float_array(angles, "angles")
    cos_mean = float(np.mean(np.cos(angles)))
    sin_mean = float(np.mean(np.sin(angles)))
    r = math.hypot(cos_mean, sin_mean)
    return {
        "mean": math.atan2(sin_mean, cos_mean),
        "resultant_length": r,
        "variance": 1.0 - r,
        "variance_2pi": 2.0 * (1.0 - r),
    }