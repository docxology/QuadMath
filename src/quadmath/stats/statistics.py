"""Deterministic descriptive statistics and resampling toolkit.

Small, dependency-free (``numpy`` plus the standard ``math`` module)
statistical utilities used by the benchmarking and visualization layers
of QuadMath.  Every function is a
pure computation over array-likes: inputs are converted with
``np.asarray(..., dtype=float)``, validated to be non-empty (an empty
sample would silently produce ``nan``), and no randomness is consumed
beyond an explicitly seeded generator, so repeated calls with the same
arguments and seed return bit-identical results.  There are no prints and
no module state; results are plain Python floats / numpy arrays.

- :func:`summarize` — n, mean, sample std (ddof=1), median, IQR, min, max.
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
    "scaling_fit",
    "summarize",
    "welch_t_test",
]


def _as_float_array(x, name: str) -> np.ndarray:
    """Convert an array-like to a float array, rejecting empty input."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def summarize(x: np.ndarray) -> Dict[str, float]:
    """Descriptive summary of a sample.

    Returns a dict with integer ``n`` and float entries ``mean``,
    ``std`` (sample standard deviation, ``ddof=1``), ``median``, ``iqr``
    (75th minus 25th percentile), ``min``, and ``max``.

    Raises
    ------
    ValueError
        If ``x`` is empty.
    """
    x = _as_float_array(x, "x")
    q25, q75 = np.percentile(x, [25.0, 75.0])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def bootstrap_ci(
    x: np.ndarray,
    stat: Callable = np.mean,
    *,
    iters: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple:
    """Percentile bootstrap confidence interval for ``stat`` on ``x``.

    Draws ``iters`` resamples of size ``len(x)`` with replacement from a
    single seeded generator (``np.random.default_rng(seed)``), evaluates
    ``stat`` on each row (along ``axis=1``), and returns the
    ``alpha``/2 and 1-``alpha``/2 percentiles of the bootstrap
    distribution as a ``(low, high)`` tuple.

    Raises
    ------
    ValueError
        If ``x`` is empty or ``iters`` is not positive.
    """
    x = _as_float_array(x, "x")
    if iters <= 0:
        raise ValueError("iters must be a positive integer")
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    stats = stat(x[idx], axis=1)
    return tuple(np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


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
    signed comparison against the observed difference.

    Raises
    ------
    ValueError
        If either sample is empty or ``alternative`` is not one of
        ``"two-sided"``, ``"greater"``, ``"less"``.
    """
    a = _as_float_array(a, "a")
    b = _as_float_array(b, "b")
    na = len(a)
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    pool = np.concatenate([a, b])
    obs = np.mean(a) - np.mean(b)
    rng = np.random.default_rng(seed)
    perm_diffs = np.empty(iters, dtype=float)
    for i in range(iters):
        perm = rng.permutation(pool)
        perm_diffs[i] = perm[:na].mean() - perm[na:].mean()
    if alternative == "two-sided":
        count = np.count_nonzero(np.abs(perm_diffs) >= abs(obs))
    elif alternative == "greater":
        count = np.count_nonzero(perm_diffs >= obs)
    else:
        count = np.count_nonzero(perm_diffs <= obs)
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
        If either sample is empty.
    """
    a = _as_float_array(a, "a")
    b = _as_float_array(b, "b")
    na, nb = len(a), len(b)
    diff = float(np.mean(a) - np.mean(b))
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0.0:
        if diff == 0.0:
            return 0.0
        return float(np.sign(diff) * np.inf)
    return float(diff / pooled)


def p_adjust_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Bonferroni-adjusted p-values, elementwise ``min(1, p * m)``.

    Parameters
    ----------
    pvals:
        Raw p-values; the family size ``m`` is ``pvals.size``.

    Raises
    ------
    ValueError
        If ``pvals`` is empty.
    """
    p = _as_float_array(pvals, "pvals")
    m = p.size
    return np.minimum(1.0, p * m)


def scaling_fit(sizes: np.ndarray, times: np.ndarray) -> tuple:
    """Power-law (log-log linear) fit of runtimes against input sizes.

    Fits ``log(times) = slope * log(sizes) + intercept`` by ordinary
    least squares (``np.polyfit(..., 1)``) and returns
    ``(slope, intercept, r2)`` where ``r2 = 1 - ss_res / ss_tot`` is
    computed on the log-log fit, so ``slope`` approximates the empirical
    complexity exponent (e.g. ~0.5 for linear scaling measured against
    the raw size axis, ~1.0 for quadratic).

    Raises
    ------
    ValueError
        If either array is empty, the arrays differ in length, or any
        value is non-positive (logs would be undefined).
    """
    sizes = _as_float_array(sizes, "sizes")
    times = _as_float_array(times, "times")
    if sizes.size != times.size:
        raise ValueError("sizes and times must have the same length")
    if np.any(sizes <= 0) or np.any(times <= 0):
        raise ValueError("sizes and times must be positive for a log-log fit")
    slope, intercept = np.polyfit(np.log(sizes), np.log(times), 1)
    fitted = slope * np.log(sizes) + intercept
    observed = np.log(times)
    ss_res = float(np.sum((observed - fitted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


_SQRT2 = math.sqrt(2.0)
_FPMIN = 1e-300
_BETA_EPS = 1e-14
_BETA_MAX_ITERS = 200


def _as_finite_float_array(x, name: str) -> np.ndarray:
    """Convert an array-like to a float array, rejecting empty or non-finite input."""
    arr = _as_float_array(x, name)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


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


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)`` on ``[0, 1]``.

    Lentz's modified continued fraction (Numerical Recipes) with the
    standard symmetry swap above ``x = (a + 1) / (a + b + 2)``; pure
    ``math``, so p-values never require ``scipy``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _reg_inc_beta(1.0 - x, b, a)
    log_front = (
        a * math.log(x)
        + b * math.log(1.0 - x)
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


def _t_two_sided_tail(t: float, df: float) -> float:
    """Two-sided tail probability ``P(|T| >= |t|)`` of Student's t.

    Uses the exact identity ``P(|T| >= t) = I_x(df/2, 1/2)`` with
    ``x = df / (df + t**2)``.
    """
    return _reg_inc_beta(df / (df + t * t), 0.5 * df, 0.5)


def _t_sf(t: float, df: float) -> float:
    """Survival function ``P(T > t)`` of Student's t with ``df`` dof."""
    if t > 0.0:
        return 0.5 * _t_two_sided_tail(t, df)
    if t < 0.0:
        return 1.0 - 0.5 * _t_two_sided_tail(t, df)
    return 0.5


def jackknife_ci(
    x: np.ndarray,
    stat: Callable = np.mean,
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
    return a scalar.

    Parameters
    ----------
    x:
        Sample values; at least two finite observations are required.
    stat:
        Scalar-valued statistic applied to each leave-one-out sample.
    alpha:
        Tail probability in ``(0, 1)``; the interval covers ``1 - alpha``.

    Returns
    -------
    Tuple[float, float, float]
        ``(low, high, bias)`` as plain Python floats.

    Raises
    ------
    ValueError
        If ``x`` is empty, contains non-finite values, has fewer than
        two observations, or ``alpha`` is outside ``(0, 1)``.
    """
    x = _as_finite_float_array(x, "x")
    if x.size < 2:
        raise ValueError("jackknife requires at least two values")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha!r}")
    n = x.size
    theta_hat = float(stat(x))
    loo = np.array([float(stat(np.delete(x, i))) for i in range(n)], dtype=float)
    loo_mean = float(np.mean(loo))
    bias = (n - 1) * (loo_mean - theta_hat)
    se = math.sqrt((n - 1) / n * float(np.sum((loo - loo_mean) ** 2)))
    z = _normal_z_from_two_sided_alpha(alpha)
    return theta_hat - z * se, theta_hat + z * se, bias


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
        If either sample is empty or has fewer than two values, or
        ``alternative`` is not one of the three supported strings.
    """
    a = _as_float_array(a, "a")
    b = _as_float_array(b, "b")
    if a.size < 2 or b.size < 2:
        raise ValueError("welch_t_test requires at least two values per sample")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative: {alternative!r}")
    na, nb = a.size, b.size
    wa = float(np.var(a, ddof=1)) / na
    wb = float(np.var(b, ddof=1)) / nb
    num = float(np.mean(a) - np.mean(b))
    denom = math.sqrt(wa + wb)
    if denom == 0.0:
        t = 0.0 if num == 0.0 else math.copysign(math.inf, num)
        df = float(na + nb - 2)
    else:
        t = num / denom
        df = (wa + wb) ** 2 / (wa * wa / (na - 1) + wb * wb / (nb - 1))
    if alternative == "two-sided":
        return t, _t_two_sided_tail(t, df)
    if alternative == "greater":
        return t, _t_sf(t, df)
    return t, 1.0 - _t_sf(t, df)


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