"""Deterministic descriptive statistics and resampling toolkit.

Small, dependency-free (``numpy`` only) statistical utilities used by the
benchmarking and visualization layers of QuadMath.  Every function is a
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
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

__all__ = [
    "bootstrap_ci",
    "cohens_d",
    "p_adjust_bonferroni",
    "permutation_test",
    "scaling_fit",
    "summarize",
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