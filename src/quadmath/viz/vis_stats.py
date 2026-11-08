"""Statistical graphics with explicit data, numerical and ownership boundaries.

The four axes primitives accept nonempty, real, finite one-dimensional arrays.
They preserve caller axes and leave newly created figures open on success,
transferring ownership to the caller. Drawing errors close only owned figures;
invalid data is rejected before any figure is created.

Scaling uses the shared :func:`quadmath.stats.statistics.scaling_fit` and
evaluates predictions in log space. Interval bounds are independent of the
point estimate: a valid confidence interval need not contain that estimate.
ECDFs include the initial jump, including tied and singleton samples.

:func:`gallery` owns four saved figures and closes them on success or failure.
Its seeded examples are synthetic, not benchmark measurements; its displayed
confidence bounds are calculated by the shared percentile-bootstrap method.
Seeds reproduce numerical inputs, not encoded image bytes across renderers.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from quadmath.paths import get_figure_dir
from quadmath.stats.statistics import bootstrap_ci, scaling_fit
from quadmath.viz._common import figure_scope, resolve_output_path, save_figure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = [
    "GALLERY_FILES",
    "gallery",
    "plot_ci_bars",
    "plot_ecdf",
    "plot_latency_hist",
    "plot_scaling_loglog",
]

#: Output file names of the four statistics gallery figures, in
#: composition order.
GALLERY_FILES: Tuple[str, str, str, str] = (
    "stats_gallery_latency.png",
    "stats_gallery_scaling.png",
    "stats_gallery_ci.png",
    "stats_gallery_ecdf.png",
)


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Reject missing, complex or non-vector display data without making axes."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a nonempty real one-dimensional array")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@contextmanager
def _axes_scope(ax: Optional["Axes"], title: str) -> Iterator["Axes"]:
    """Transfer a new figure on success; close only owned figures on errors."""
    owned = ax is None
    if ax is None:
        _, ax = plt.subplots()
    try:
        ax.set_title(title)
        yield ax
    except BaseException:
        if owned:
            plt.close(ax.figure)
        raise


def plot_latency_hist(
    times: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    bins: int = 20,
    title: str = "Latency distribution",
) -> "Axes":
    """Histogram of a latency sample with a dashed vertical mean line.

    Parameters
    - times: Nonempty real finite 1-D sample, in one caller-declared unit.
    - ax: Caller-owned axes, or ``None`` to transfer a new figure on success.
    - bins: Number of histogram bins (passed straight to
      :meth:`~matplotlib.axes.Axes.hist`).
    - title: Axes title.

    Returns
    - Axes: The axes holding the histogram and the mean line.

    Raises
    - ValueError: If the sample is empty, non-finite, complex or not 1-D.
    """
    values = _finite_vector(times, "times")
    scale = float(np.max(np.abs(values)))
    mean = float(np.mean(values / scale) * scale) if scale else 0.0
    with _axes_scope(ax, title) as ax:
        ax.hist(values, bins=bins)
        ax.axvline(mean, color="C1", linestyle="--", label=f"mean = {mean:.3g}")
        ax.set_xlabel("latency")
        ax.set_ylabel("count")
        ax.legend()
        return ax


def plot_scaling_loglog(
    sizes: np.ndarray,
    times: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    title: str = "Scaling (log-log)",
) -> "Axes":
    """Log-log scatter of times versus sizes with the fitted power law.

    The shared ``scaling_fit`` performs ordinary least squares on log data.
    Predictions are evaluated as ``exp(intercept + slope * log(size))``,
    rather than multiplying an underflowed coefficient by an overflowed power.
    The fitted line is sorted by size; the legend reports the log-space R².

    Parameters
    - sizes: Nonempty real finite 1-D positive sizes; two distinct sizes needed.
    - times: Real finite positive times, aligned with ``sizes``.
    - ax: Caller-owned axes, or ``None`` to transfer a new figure on success.
    - title: Axes title.

    Returns
    - Axes: The axes holding the scatter, the fitted line, and the legend.

    Raises
    - ValueError: For invalid/alignment data, an unidentified slope, or a fit
      whose positive predictions cannot be represented as finite floats.
    """
    sizes_f = _finite_vector(sizes, "sizes")
    times_f = _finite_vector(times, "times")
    slope, intercept, r2 = scaling_fit(sizes_f, times_f)
    try:
        with np.errstate(over="raise", invalid="raise"):
            fitted = np.exp(intercept + slope * np.log(sizes_f))
    except FloatingPointError as error:
        raise ValueError("Fitted scaling values are outside the floating display range") from error
    if np.any(fitted == 0):
        raise ValueError("Fitted scaling values are outside the floating display range")
    order = np.argsort(sizes_f)
    r2_text = f"{r2:.3f}" if np.isfinite(r2) else "undefined"
    with _axes_scope(ax, title) as ax:
        ax.loglog(sizes_f, times_f, "o", label="data")
        ax.loglog(
            sizes_f[order], fitted[order], "--",
            label=f"log-space fit: slope = {slope:.3f}, R² = {r2_text}",
        )
        ax.set_xlabel("size")
        ax.set_ylabel("time")
        ax.legend()
        return ax


def plot_ci_bars(
    labels: Sequence[str],
    means: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    title: str = "Estimates with CI",
) -> "Axes":
    """Render finite interval endpoints independently of their point estimates.

    Parameters
    - labels: One label per bar (drawn as rotated x tick labels).
    - means: Nonempty real finite 1-D point estimates; need not lie in the intervals.
    - lows: Real finite 1-D lower interval bounds.
    - highs: Real finite 1-D upper bounds, at least the corresponding lower bounds.
    - ax: Caller-owned axes, or ``None`` to transfer a new figure on success.
    - title: Axes title.

    Returns
    - Axes: The axes holding the error-bar plot.

    Raises
    - ValueError: For non-vector/empty/non-finite/complex data, mismatched
      lengths, or a lower bound greater than its upper bound.
    """
    means_f = _finite_vector(means, "means")
    lows_f = _finite_vector(lows, "lows")
    highs_f = _finite_vector(highs, "highs")
    n = means_f.size
    if len(labels) != n or lows_f.size != n or highs_f.size != n:
        raise ValueError("labels, means, lows, and highs must have equal length")
    if np.any(lows_f > highs_f):
        raise ValueError("Each lower interval bound must not exceed its upper bound")
    x = np.arange(n)
    with _axes_scope(ax, title) as ax:
        ax.vlines(x, lows_f, highs_f, color="C0")
        ax.plot(x, lows_f, linestyle="none", marker="_", markersize=10, color="C0")
        ax.plot(x, highs_f, linestyle="none", marker="_", markersize=10, color="C0")
        ax.plot(x, means_f, linestyle="none", marker="o", color="C0")
        ax.set_xticks(x, list(labels), rotation=30, ha="right")
        ax.set_ylabel("estimate")
        return ax


def plot_ecdf(
    values: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    title: str = "Empirical CDF",
) -> "Axes":
    """Render a right-continuous ECDF over the observed sample range.

    Equal values contribute one jump with their combined empirical mass.
    The path starts at zero at the minimum, so the first jump remains visible
    even when every observation is equal. No continuous density is inferred.

    Parameters
    - values: Nonempty real finite 1-D observations.
    - ax: Caller-owned axes, or ``None`` to transfer a new figure on success.
    - title: Axes title.

    Returns
    - Axes: The axes holding the step plot.

    Raises
    - ValueError: If observations are empty, non-finite, complex or not 1-D.
    """
    sample = _finite_vector(values, "values")
    ordered, counts = np.unique(sample, return_counts=True)
    cumulative = np.cumsum(counts) / sample.size
    with _axes_scope(ax, title) as ax:
        ax.step(
            np.concatenate(([ordered[0]], ordered)),
            np.concatenate(([0.0], cumulative)), where="post",
        )
        ax.set_xlabel("value")
        ax.set_ylabel("P(X <= x)")
        ax.set_ylim(0.0, 1.05)
        return ax


def gallery(paths_out_dir: str, seed: int = 32) -> List[str]:
    """Compose the four statistics-gallery figures deterministically.

    Panels:

    1. ``stats_gallery_latency.png`` — seeded clipped-lognormal synthetic
       latencies and their sample mean, not measured function timings.
    2. ``stats_gallery_scaling.png`` — a synthetic noisy power law with
       planted exponent 1.35 and a shared log-space regression.
    3. ``stats_gallery_ci.png`` — computed 95% percentile-bootstrap mean
       intervals for independent synthetic normal samples of sizes 16/64/256,
       with the known generating mean shown separately.
    4. ``stats_gallery_ecdf.png`` — the ECDF of the histogram's same sample.

    A fixed seed reproduces the numerical examples, not cross-version image
    bytes. These demonstrations establish neither benchmark performance nor
    guaranteed coverage of an individual confidence interval.

    Parameters
    - paths_out_dir: Directory the four PNGs are written to (created when
      missing).  A bare directory name is resolved under
      ``quadmath/output/figures/``; a path with a directory component (or an
      absolute path) is used as given.
    - seed: Seed for synthetic samples, scaling noise and bootstrap resampling.

    Returns
    - List[str]: The four written paths, in :data:`GALLERY_FILES` order.
    """
    paths_out_dir = resolve_output_path(paths_out_dir, get_figure_dir)
    os.makedirs(paths_out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Synthetic numerical examples, not measurements of named package functions.
    latency = np.clip(rng.lognormal(mean=0.0, sigma=0.6, size=256), 0.05, 5.0)
    sizes = np.array([8, 16, 32, 64, 128, 256], dtype=float)
    times = 3.0 * sizes**1.35 * (1.0 + rng.normal(0.0, 0.02, sizes.size))
    sample_sizes = (16, 64, 256)
    samples = [rng.normal(1.5, 0.4, size=n) for n in sample_sizes]
    labels = [f"n={n}" for n in sample_sizes]
    means = np.array([np.mean(sample) for sample in samples])
    intervals = np.array([
        bootstrap_ci(sample, np.mean, iters=2000, seed=int(seed) + index, alpha=0.05)
        for index, sample in enumerate(samples)
    ])
    lows, highs = intervals.T

    # Figure 1: latency histogram with dashed mean line.
    with figure_scope(figsize=(6.4, 4.8)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plot_latency_hist(latency, ax, title="Seeded synthetic latency sample")
        ax.set_xlabel("synthetic latency (s)")
        fig.tight_layout()
        latency_path = os.path.join(paths_out_dir, GALLERY_FILES[0])
        save_figure(fig, latency_path, dpi=160)

    # Figure 2: log-log scaling with fitted power-law line.
    with figure_scope(figsize=(6.4, 4.8)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plot_scaling_loglog(sizes, times, ax, title="Synthetic power law: planted exponent 1.35")
        ax.set_xlabel("synthetic workload size")
        ax.set_ylabel("synthetic duration (arbitrary units)")
        fig.tight_layout()
        scaling_path = os.path.join(paths_out_dir, GALLERY_FILES[1])
        save_figure(fig, scaling_path, dpi=160)

    # Figure 3: point estimates with confidence intervals.
    with figure_scope(figsize=(6.4, 4.8)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plot_ci_bars(
            labels, means, lows, highs, ax,
            title="Synthetic sample means\n95% percentile-bootstrap intervals",
        )
        ax.axhline(1.5, color="gray", linestyle=":", label="generating mean")
        ax.set_ylabel("mean (arbitrary units)")
        ax.legend()
        fig.tight_layout()
        ci_path = os.path.join(paths_out_dir, GALLERY_FILES[2])
        save_figure(fig, ci_path, dpi=160)

    # Figure 4: empirical CDF of the latency sample.
    with figure_scope(figsize=(6.4, 4.8)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plot_ecdf(latency, ax, title="ECDF of the same seeded synthetic latency sample")
        ax.set_xlabel("synthetic latency (s)")
        fig.tight_layout()
        ecdf_path = os.path.join(paths_out_dir, GALLERY_FILES[3])
        save_figure(fig, ecdf_path, dpi=160)

    return [latency_path, scaling_path, ci_path, ecdf_path]
