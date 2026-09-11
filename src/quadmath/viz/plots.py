"""Standalone figure builders for training curves, shell growth, and errors.

Unlike the galleries in ``vis_lattice``/``vis_stats`` (which draw onto
caller-supplied axes), each function here owns a fixed-size figure, applies a
deterministic style, and (by default) writes a PNG into
``quadmath/output/figures/`` via :func:`quadmath.paths.get_figure_dir`,
returning the output path.

All functions are deterministic: inputs are plain sequences, styling is
fixed, and no randomness, wall-clock time, or ``plt.show()`` is used.  Tests
run headless via ``MPLBACKEND=Agg`` (set in ``tests/conftest.py``).
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from quadmath.core.quadray import DEFAULT_EMBEDDING, to_xyz
from quadmath.lattice.ivm_field import shell_cardinalities, shell_sites
from quadmath.paths import get_figure_dir
from quadmath.viz.visualize import _set_axes_equal

__all__ = [
    "plot_loss_history",
    "plot_shell_growth",
    "plot_error_histogram",
    "plot_lattice_shell_3d",
]

_FIGSIZE = (6.4, 4.8)
_DPI = 160


def plot_loss_history(losses: Sequence[float], save: bool = True) -> str:
    """Plot a training loss sequence as a line with markers.

    Parameters
    - losses: Any finite sequence of loss values (e.g.
      ``GradientDescentTrainer.loss_history`` or a plain list), one per
      iteration in training order.
    - save: If True, write PNG to ``quadmath/output/figures/loss_history.png``.

    Returns
    - str: Output file path when ``save`` is True, else "" (the open figure
      remains the caller's responsibility).

    Raises
    - ValueError: If ``losses`` is empty.
    """
    values = np.asarray(list(losses), dtype=float)
    if values.size == 0:
        raise ValueError("loss history must contain at least one value")

    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(values.size), values, color="tab:blue", marker="o", markersize=3, linewidth=1.2)
    ax.set_title("Training loss history")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)

    if save:
        outpath = f"{get_figure_dir()}/loss_history.png"
        plt.savefig(outpath, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return outpath
    return ""


def plot_shell_growth(k_max: int = 6, save: bool = True) -> str:
    """Plot IVM shell cardinalities (cuboctahedral numbers) versus shell index.

    Parameters
    - k_max: Largest shell index to include; shells ``0 .. k_max`` are drawn.
    - save: If True, write PNG to ``quadmath/output/figures/shell_growth.png``.

    Returns
    - str: Output file path when ``save`` is True, else "" (the open figure
      remains the caller's responsibility).

    Raises
    - ValueError: If ``k_max`` is negative.
    """
    if k_max < 0:
        raise ValueError(f"k_max must be non-negative, got {k_max}")

    cardinalities = shell_cardinalities(k_max)
    ks = np.arange(len(cardinalities))

    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.plot(ks, np.asarray(cardinalities, dtype=float), color="tab:purple", marker="o", linewidth=1.2)
    ax.set_title(f"IVM shell cardinalities, k = 0..{k_max}")
    ax.set_xlabel("shell index k")
    ax.set_ylabel("sites in shell")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    if save:
        outpath = f"{get_figure_dir()}/shell_growth.png"
        plt.savefig(outpath, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return outpath
    return ""


def plot_error_histogram(errors: Sequence[float], bins: int = 20, save: bool = True) -> str:
    """Plot a histogram of error values with a dashed vertical mean line.

    Parameters
    - errors: Non-empty sequence of finite error values.
    - bins: Number of histogram bins (positive integer).
    - save: If True, write PNG to
      ``quadmath/output/figures/error_histogram.png``.

    Returns
    - str: Output file path when ``save`` is True, else "" (the open figure
      remains the caller's responsibility).

    Raises
    - ValueError: If ``errors`` is empty, contains a non-finite value
      (NaN or infinity), or ``bins`` is not a positive integer.
    """
    values = np.asarray(list(errors), dtype=float)
    if values.size == 0:
        raise ValueError("errors must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError("errors must all be finite (no NaN or infinity)")
    if bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}")

    mean = float(np.mean(values))

    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.hist(values, bins=bins, color="tab:blue", edgecolor="white", alpha=0.8)
    ax.axvline(mean, color="tab:red", linestyle="--", linewidth=1.5, label=f"mean = {mean:.6g}")
    ax.set_title("Error histogram")
    ax.set_xlabel("error")
    ax.set_ylabel("count")
    ax.legend()

    if save:
        outpath = f"{get_figure_dir()}/error_histogram.png"
        plt.savefig(outpath, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return outpath
    return ""


def plot_lattice_shell_3d(k: int = 2, save: bool = True) -> str:
    """Scatter the sites of one IVM lattice shell in 3D (equal-aspect axes).

    Parameters
    - k: Non-negative shell index; sites with quadray shell norm ``2k`` are
      embedded via :data:`quadmath.core.quadray.DEFAULT_EMBEDDING`.
    - save: If True, write PNG to
      ``quadmath/output/figures/lattice_shell_3d.png``.

    Returns
    - str: Output file path when ``save`` is True, else "" (the open figure
      remains the caller's responsibility).

    Raises
    - ValueError: If ``k`` is negative.
    """
    if k < 0:
        raise ValueError(f"shell index must be non-negative, got {k}")

    sites = shell_sites(k)
    xyz = np.asarray([to_xyz(q, DEFAULT_EMBEDDING) for q in sites], dtype=float)

    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="tab:blue", s=24)
    ax.set_title(f"IVM lattice shell k={k} ({len(sites)} sites)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)

    if save:
        outpath = f"{get_figure_dir()}/lattice_shell_3d.png"
        plt.savefig(outpath, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return outpath
    return ""
