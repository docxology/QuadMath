"""Statistics visualization primitives for benchmark and inference results.

This module is the figure surface for the statistics layer (latency
distributions, log-log scaling fits, confidence intervals, empirical CDFs).
It is deliberately input-agnostic: every primitive consumes plain numpy
arrays and never imports ``benchmarks`` or ``statistics``, so panels
compose inside caller-owned figures for any data source.  Only
:func:`gallery` creates figures (four of them, written as PNG files); no
figure is created at import time.

Pieces:

- :func:`plot_latency_hist` — histogram of a latency sample with a
  dashed mean line.
- :func:`plot_scaling_loglog` — log-log scatter of measured times versus
  input size together with the fitted power law
  ``t ~ c * n**slope`` obtained from a degree-1 fit in log space.
- :func:`plot_ci_bars` — point estimates with symmetric confidence
  intervals rendered as error bars with caps.
- :func:`plot_ecdf` — empirical cumulative distribution function of a
  sample as a sorted step plot.
- :func:`gallery` — composes the four statistics-gallery figures
  deterministically (fixed file order :data:`GALLERY_FILES`) and returns
  their paths.

All randomness used by :func:`gallery` comes from a seeded
``numpy.random.default_rng(seed)``; the rendered PNGs carry no timestamp
metadata, so a re-run with the same seed reproduces the figures
byte-for-byte.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


def _prepare_axes(ax: Optional["Axes"], title: str) -> "Axes":
    """Return the axes to draw on, creating a fresh figure when needed.

    Parameters
    - ax: Caller-owned axes, or ``None`` to create a new figure/axes via
      :func:`matplotlib.pyplot.subplots`.
    - title: Title set on the returned axes.

    Returns
    - Axes: The axes to draw on, titled ``title``.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(title)
    return ax


def plot_latency_hist(
    times: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    bins: int = 20,
    title: str = "Latency distribution",
) -> "Axes":
    """Histogram of a latency sample with a dashed vertical mean line.

    Parameters
    - times: 1-D array of latency samples (any positive scale).
    - ax: Axes to draw on; ``None`` creates a fresh figure and axes.
    - bins: Number of histogram bins (passed straight to
      :meth:`~matplotlib.axes.Axes.hist`).
    - title: Axes title.

    Returns
    - Axes: The axes holding the histogram and the mean line.
    """
    ax = _prepare_axes(ax, title)
    values = np.asarray(times, dtype=float)
    mean = float(np.mean(values))
    ax.hist(values, bins=bins)
    ax.axvline(mean, color="C1", linestyle="--", label=f"mean = {mean:.3f}")
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

    The fitted line solves ``np.polyfit(np.log(sizes), np.log(times), 1)``,
    i.e. ``log t = intercept + slope * log n``, and is drawn as
    ``t = exp(intercept) * n**slope``.  The slope is annotated in the fit
    legend label.

    Parameters
    - sizes: 1-D array of strictly positive input sizes.
    - times: 1-D array of strictly positive measured times, aligned with
      ``sizes``.
    - ax: Axes to draw on; ``None`` creates a fresh figure and axes.
    - title: Axes title.

    Returns
    - Axes: The axes holding the scatter, the fitted line, and the legend.
    """
    ax = _prepare_axes(ax, title)
    sizes_f = np.asarray(sizes, dtype=float)
    times_f = np.asarray(times, dtype=float)
    (slope, intercept) = np.polyfit(np.log(sizes_f), np.log(times_f), 1)
    fitted = np.exp(intercept) * sizes_f**slope
    ax.loglog(sizes_f, times_f, "o", label="data")
    ax.loglog(
        sizes_f,
        fitted,
        "--",
        label=f"fit: slope = {slope:.3f}",
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
    """Point estimates with symmetric confidence intervals as error bars.

    Parameters
    - labels: One label per bar (drawn as rotated x tick labels).
    - means: 1-D array of point estimates.
    - lows: 1-D array of lower confidence-interval bounds.
    - highs: 1-D array of upper confidence-interval bounds.
    - ax: Axes to draw on; ``None`` creates a fresh figure and axes.
    - title: Axes title.

    Returns
    - Axes: The axes holding the error-bar plot.

    Raises
    - ValueError: If ``labels``, ``means``, ``lows``, and ``highs`` do not
      all have the same length.
    """
    ax = _prepare_axes(ax, title)
    means_f = np.asarray(means, dtype=float)
    n = means_f.shape[0]
    if len(labels) != n or len(lows) != n or len(highs) != n:
        raise ValueError(
            "labels, means, lows, and highs must have equal length; got "
            f"{len(labels)}, {n}, {len(lows)}, {len(highs)}"
        )
    lows_f = np.asarray(lows, dtype=float)
    highs_f = np.asarray(highs, dtype=float)
    x = np.arange(n)
    yerr = np.vstack((means_f - lows_f, highs_f - means_f))
    ax.errorbar(x, means_f, yerr=yerr, fmt="o", capsize=5)
    ax.set_xticks(x, list(labels), rotation=30, ha="right")
    ax.set_ylabel("estimate")
    return ax


def plot_ecdf(
    values: np.ndarray,
    ax: Optional["Axes"] = None,
    *,
    title: str = "Empirical CDF",
) -> "Axes":
    """Empirical cumulative distribution function as a sorted step plot.

    The step uses ``where='post'`` so each step starts at its data value;
    the vertical axis is the fraction ``i / n`` of the sample at or below
    each sorted value.

    Parameters
    - values: 1-D array of samples.
    - ax: Axes to draw on; ``None`` creates a fresh figure and axes.
    - title: Axes title.

    Returns
    - Axes: The axes holding the step plot.
    """
    ax = _prepare_axes(ax, title)
    ordered = np.sort(np.asarray(values, dtype=float))
    n = ordered.shape[0]
    cumulative = np.arange(1, n + 1, dtype=float) / n
    ax.step(ordered, cumulative, where="post")
    ax.set_xlabel("value")
    ax.set_ylabel("P(X <= x)")
    ax.set_ylim(0.0, 1.05)
    return ax


def gallery(paths_out_dir: str, seed: int = 32) -> List[str]:
    """Compose the four statistics-gallery figures deterministically.

    Panels:

    1. ``stats_gallery_latency.png`` — histogram of a clipped lognormal
       latency sample with its dashed mean line
       (:func:`plot_latency_hist`).
    2. ``stats_gallery_scaling.png`` — log-log scaling of times roughly
       ``3 * n**1.35`` over sizes 8..256 with the fitted slope
       (:func:`plot_scaling_loglog`; the times carry small float noise
       rounded to four decimals for byte stability).
    3. ``stats_gallery_ci.png`` — three benchmark estimates
       (``to_xyz``, ``quadray_from_xyz``, ``shell_enum``) with symmetric
       confidence intervals (:func:`plot_ci_bars`).
    4. ``stats_gallery_ecdf.png`` — empirical CDF of the same clipped
       lognormal latency sample (:func:`plot_ecdf`).

    All randomness is drawn from one seeded generator
    (``numpy.random.default_rng(seed)``), so the figures are reproducible
    byte-for-byte for a fixed ``seed``.

    Parameters
    - paths_out_dir: Directory the four PNGs are written to (created when
      missing).
    - seed: Seed for the latency sample, the scaling noise, and the
      confidence half-widths.

    Returns
    - List[str]: The four written paths, in :data:`GALLERY_FILES` order.
    """
    if not os.path.isdir(paths_out_dir):
        os.makedirs(paths_out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Shared deterministic data: one clipped lognormal latency sample,
    # one noisy power-law scaling sweep, three benchmark estimates.
    latency = np.clip(rng.lognormal(mean=0.0, sigma=0.6, size=256), 0.05, 5.0)
    sizes = np.array([8, 16, 32, 64, 128, 256], dtype=float)
    times = np.round(3.0 * sizes**1.35 * (1.0 + rng.normal(0.0, 0.02, sizes.size)), 4)
    labels = ["to_xyz", "quadray_from_xyz", "shell_enum"]
    means = np.array([1.1, 3.2, 9.8], dtype=float)
    half_widths = np.round(rng.uniform(0.05, 0.25, size=len(labels)), 4)
    lows = means - half_widths
    highs = means + half_widths

    # Figure 1: latency histogram with dashed mean line.
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    plot_latency_hist(latency, ax)
    fig.tight_layout()
    latency_path = os.path.join(paths_out_dir, GALLERY_FILES[0])
    fig.savefig(latency_path, dpi=160)
    plt.close(fig)

    # Figure 2: log-log scaling with fitted power-law line.
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    plot_scaling_loglog(sizes, times, ax)
    fig.tight_layout()
    scaling_path = os.path.join(paths_out_dir, GALLERY_FILES[1])
    fig.savefig(scaling_path, dpi=160)
    plt.close(fig)

    # Figure 3: point estimates with symmetric confidence intervals.
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    plot_ci_bars(labels, means, lows, highs, ax)
    fig.tight_layout()
    ci_path = os.path.join(paths_out_dir, GALLERY_FILES[2])
    fig.savefig(ci_path, dpi=160)
    plt.close(fig)

    # Figure 4: empirical CDF of the latency sample.
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    plot_ecdf(latency, ax)
    fig.tight_layout()
    ecdf_path = os.path.join(paths_out_dir, GALLERY_FILES[3])
    fig.savefig(ecdf_path, dpi=160)
    plt.close(fig)

    return [latency_path, scaling_path, ci_path, ecdf_path]
