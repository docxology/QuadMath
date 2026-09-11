"""Standalone figure builders for training curves, shell growth, quaternion slerp paths, and errors.

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

import math
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, qrotate, slerp, to_xyz
from quadmath.lattice.ivm_field import shell_cardinalities, shell_sites
from quadmath.paths import get_figure_dir
from quadmath.viz.visualize import _set_axes_equal

__all__ = [
    "plot_loss_history",
    "plot_shell_growth",
    "plot_error_histogram",
    "plot_lattice_shell_3d",
    "plot_slerp_path",
]

_FIGSIZE = (6.4, 4.8)
_DPI = 160

#: Numerical tolerance for the unit-norm check on quaternion inputs.
_UNIT_QUAT_TOL = 1e-9

#: Default site rotated by :func:`plot_slerp_path`: the canonical (2, 0, 0, 0)
#: point on the A quadray axis, embedded at (2, 2, 2) under DEFAULT_EMBEDDING.
_DEFAULT_SLERP_SITE = Quadray(2, 0, 0, 0)


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


def _encoded_angle(quat: Tuple[float, float, float, float]) -> float:
    """Return the rotation magnitude a unit quaternion encodes.

    The magnitude is ``2*atan2(||(x, y, z)||, w)``; passing it to
    :func:`quadmath.core.quadray.qrotate` together with the quaternion
    satisfies that function's encoded-angle validation exactly.
    """
    return 2.0 * math.atan2(math.sqrt(quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2), quat[0])


def plot_slerp_path(
    q0: Sequence[float],
    q1: Sequence[float],
    n_frames: int = 16,
    site: Optional[Quadray] = None,
    save: bool = True,
) -> str:
    """Trace the 3D path of a fixed lattice site under shortest-arc slerp.

    Interpolates between the unit quaternions ``q0`` and ``q1`` with
    :func:`quadmath.core.quadray.slerp` over ``n_frames`` samples of ``t``
    in ``[0, 1]`` (endpoints included), rotates one fixed Quadray site by
    each interpolated quaternion via :func:`quadmath.core.quadray.qrotate`,
    and draws the traced 3D path as a line with start/end markers on
    equal-aspect axes.

    Parameters
    - q0, q1: Unit quaternions ``(w, x, y, z)`` (norm 1 within 1e-9) to
      interpolate between.
    - n_frames: Number of interpolation samples along the path (at least 2).
    - site: Quadray point to rotate; defaults to the canonical ``(2, 0, 0, 0)``
      point on the A quadray axis (embedded at ``(2, 2, 2)`` under
      :data:`quadmath.core.quadray.DEFAULT_EMBEDDING`).
    - save: If True, write PNG to
      ``quadmath/output/figures/quaternion_slerp_path.png``.

    Returns
    - str: Output file path when ``save`` is True, else "" (the open figure
      remains the caller's responsibility).

    Raises
    - ValueError: If ``q0`` or ``q1`` is not a unit quaternion (norm 1
      within 1e-9), or ``n_frames`` is smaller than 2.
    """
    if n_frames < 2:
        raise ValueError(f"n_frames must be at least 2, got {n_frames}")
    q0_vec = tuple(float(c) for c in q0)
    q1_vec = tuple(float(c) for c in q1)
    n0 = math.sqrt(sum(c * c for c in q0_vec))
    n1 = math.sqrt(sum(c * c for c in q1_vec))
    if abs(n0 - 1.0) > _UNIT_QUAT_TOL or abs(n1 - 1.0) > _UNIT_QUAT_TOL:
        raise ValueError(
            "q0 and q1 must be unit quaternions (norm 1 within 1e-9), "
            f"got norms {n0!r} and {n1!r}"
        )
    if site is None:
        site = _DEFAULT_SLERP_SITE
    base = to_xyz(site, DEFAULT_EMBEDDING)
    quats = [slerp(q0_vec, q1_vec, t) for t in np.linspace(0.0, 1.0, n_frames)]
    path = np.asarray([qrotate(q, base, _encoded_angle(q)) for q in quats], dtype=float)

    fig = plt.figure(figsize=_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="tab:blue", linewidth=1.2)
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], c="tab:green", s=48, label="start (t = 0)")
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c="tab:red", s=48, label="end (t = 1)")
    ax.legend()
    ax.set_title(f"Slerp rotation path of site {site.as_tuple()}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)

    if save:
        outpath = f"{get_figure_dir()}/quaternion_slerp_path.png"
        plt.savefig(outpath, dpi=_DPI, bbox_inches="tight")
        plt.close(fig)
        return outpath
    return ""
