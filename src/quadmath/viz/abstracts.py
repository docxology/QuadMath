"""Quadray basis and canonical FCC contact geometry in scale-one XYZ units.

Four native components describe three physical dimensions: adding the same
constant to every component leaves the default embedding unchanged. The FCC
panel uses the origin and the twelve canonical permutations of (2, 1, 1, 0),
not the four vertices of the basis tetrahedron as a substitute neighborhood.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
from matplotlib.axes import Axes

from quadmath.core.quadray import DEFAULT_EMBEDDING, IVM_NEIGHBOR_MOVES
from quadmath.paths import get_figure_dir
from quadmath.viz._common import figure_scope, resolve_output_path, save_figure, set_axes_equal

__all__ = ["render_quadray_abstract"]


def _physical_axes(ax: Axes, bound: float, ticks: tuple[int, ...]) -> None:
    """Use equal physical lengths and a common orthographic viewing direction."""
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
           xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(labelsize=9, pad=0)
    ax.grid(False)
    set_axes_equal(ax)
    ax.set_proj_type("ortho")
    ax.view_init(elev=19, azim=32)


def _draw_basis(ax: Axes, embedding: np.ndarray) -> None:
    """Draw the unnormalized native basis columns, including their endpoints."""
    vertices = embedding.T
    colors = ("#0072b2", "#d55e00", "#009e73", "#a4428b")
    for name, vector, color in zip("ABCD", vertices, colors):
        arrow = ax.quiver(
            0, 0, 0, *vector, color=color, linewidth=2.4,
            arrow_length_ratio=0.12, length=1.0, normalize=False,
        )
        arrow.set_gid(f"quadray-basis-{name}")
        ax.text(*(1.16 * vector), name, color=color, fontsize=13, fontweight="bold")
    for i, j in combinations(range(4), 2):
        ax.plot(*vertices[[i, j]].T, color="0.6", linewidth=1.0, alpha=0.65)
    ax.scatter(*vertices.T, c=colors, s=30, depthshade=False)
    ax.scatter([0], [0], [0], color="0.2", s=22, depthshade=False)
    _physical_axes(ax, 1.55, (-1, 0, 1))
    ax.set_title("A  |  Four native basis rays in 3D", fontsize=14, fontweight="bold", pad=22)
    ax.text2D(0.5, 1.01, r"Ray length $\sqrt{3}$; tetrahedron edge $2\sqrt{2}$",
              transform=ax.transAxes, ha="center", fontsize=11)
    ax.text2D(0.5, -0.10, "Common-offset gauge: " + r"$q \sim q+t(1,1,1,1)$" + "\n"
              + r"$E(1,1,1,1)^T=0$; $\operatorname{rank}(E)=3$",
              transform=ax.transAxes, ha="center", va="top", fontsize=11, linespacing=1.6)


def _draw_fcc_contacts(ax: Axes, embedding: np.ndarray) -> None:
    """Show thirteen actual sphere centers and the twelve central contacts."""
    neighbors = np.asarray(IVM_NEIGHBOR_MOVES, dtype=float) @ embedding.T
    centers = np.vstack((np.zeros(3), neighbors))
    radius = float(np.linalg.norm(neighbors[0])) / 2
    longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, 33), np.linspace(0, np.pi, 21))
    sphere = radius * np.asarray((
        np.cos(longitude) * np.sin(latitude),
        np.sin(longitude) * np.sin(latitude),
        np.cos(latitude),
    ))
    for index, center in enumerate(centers):
        surface = ax.plot_surface(
            *(sphere + center[:, None, None]),
            color="#e69f00" if index == 0 else "#56b4e9",
            alpha=0.38 if index == 0 else 0.10, linewidth=0,
            shade=True, antialiased=True,
        )
        surface.set_gid(f"fcc-sphere-{index}")
    for index, center in enumerate(neighbors):
        line, = ax.plot(*np.stack((np.zeros(3), center)).T,
                        color="#a65e00", linewidth=1.15, alpha=0.85)
        line.set_gid(f"fcc-central-contact-{index}")
    ax.scatter(*neighbors.T, color="#0072b2", s=24, depthshade=False)
    ax.scatter(*(neighbors / 2).T, color="#a65e00", s=17, depthshade=False)
    ax.scatter([0], [0], [0], color="#a65e00", s=52, depthshade=False)
    _physical_axes(ax, 3.85, (-2, 0, 2))
    ax.set_title("B  |  FCC packing: twelve around one", fontsize=14, fontweight="bold", pad=22)
    ax.text2D(0.5, 1.01, r"13 sites; 12 central contacts; sphere radius $\sqrt{2}$",
              transform=ax.transAxes, ha="center", fontsize=11)
    ax.text2D(0.5, -0.10, r"Center $q=(0,0,0,0)$; neighbors $\mathrm{perm}(2,1,1,0)$" + "\n"
              + r"All 12 center distances $2\sqrt{2}$ ($d^2=8$ XYZ units$^2$)",
              transform=ax.transAxes, ha="center", va="top", fontsize=11, linespacing=1.6)
    ax.text2D(0.5, -0.045, "Blue: neighbor centers   ·   Amber: central contacts",
              transform=ax.transAxes, ha="center", fontsize=9, color="0.25")


def render_quadray_abstract(out_path: str | None = None) -> tuple[str, str]:
    """Save the native basis and the canonical thirteen-site FCC neighborhood.

    Both panels use ``DEFAULT_EMBEDDING`` without normalizing vectors or
    rescaling coordinates. The basis columns have length sqrt(3). The twelve
    permutations of (2, 1, 1, 0) lie at distance 2*sqrt(2) from the origin;
    spheres of radius sqrt(2) therefore have twelve central contacts. Equal
    physical aspect is enforced within each panel; panel viewing limits differ.

    Parameters
    ----------
    out_path : str or None
        PNG destination, with a ``.png`` suffix. Defaults to
        ``graphical_abstract_quadray.png`` under the configured figure directory
        (``QUADMATH_OUTPUT_DIR/figures`` when set). Bare names use that directory;
        explicit parent directories must already exist. The SVG shares its stem.

    Returns
    -------
    tuple[str, str]
        PNG and SVG output paths, in that order. Each file is saved atomically;
        a failure on the second save leaves the completed PNG in place. Only the
        renderer's own figure is closed, including when drawing or saving fails.

    Raises
    ------
    ValueError
        If the destination does not have a ``.png`` suffix.
    OSError
        If an output cannot be written.
    """
    path = resolve_output_path(out_path or "graphical_abstract_quadray.png", get_figure_dir)
    stem, extension = os.path.splitext(path)
    if extension.lower() != ".png":
        raise ValueError("out_path must have a .png suffix; an SVG companion is also written")
    svg_path = f"{stem}.svg"
    embedding = np.asarray(DEFAULT_EMBEDDING, dtype=float)
    with figure_scope(figsize=(13.2, 7.0)) as fig:
        _draw_basis(fig.add_subplot(121, projection="3d"), embedding)
        _draw_fcc_contacts(fig.add_subplot(122, projection="3d"), embedding)
        fig.suptitle("Quadrays: a 3D quotient and its FCC contact geometry",
                     fontsize=18, fontweight="bold", y=0.97)
        fig.text(0.5, 0.905, r"Scale-one XYZ embedding: $(X,Y,Z)=(a-b-c+d,\ a+b-c-d,\ a-b+c-d)$",
                 ha="center", fontsize=12)
        fig.text(0.5, 0.025, "Physical coordinates are in XYZ units; each panel has equal X, Y and Z scale.",
                 ha="center", fontsize=10, color="0.3")
        fig.subplots_adjust(left=0.035, right=0.965, top=0.79, bottom=0.21, wspace=0.15)
        save_figure(fig, path, dpi=240)
        save_figure(fig, svg_path)
    return path, svg_path
