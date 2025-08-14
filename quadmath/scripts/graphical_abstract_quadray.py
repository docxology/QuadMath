#!/usr/bin/env python3
"""Graphical abstract: Quadray coordinate system overview.

Panel A: Four Quadray axes (A,B,C,D) rendered as colored arrows from the
origin to the vertices of a regular tetrahedron under a symmetric embedding.
Panel B: Close-packed spheres placed at the four tetrahedron vertices to
emphasize the IVM/CCP/FCC correspondence.
"""
from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    import numpy as np  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433
    from paths import get_output_dir  # noqa: WPS433
    from quadray import DEFAULT_EMBEDDING  # noqa: WPS433

    # Axes in Quadray units (one-step spokes); embed to XYZ
    A = np.array([1, 0, 0, 0], dtype=float)
    B = np.array([0, 1, 0, 0], dtype=float)
    C = np.array([0, 0, 1, 0], dtype=float)
    D = np.array([0, 0, 0, 1], dtype=float)
    E = np.array(DEFAULT_EMBEDDING, dtype=float)

    def to_xyz(q: np.ndarray) -> np.ndarray:
        return E @ q

    axes = {
        "A": to_xyz(A),
        "B": to_xyz(B),
        "C": to_xyz(C),
        "D": to_xyz(D),
    }

    # Create 2-panel figure (A: axes; B: spheres at vertices)
    fig = plt.figure(figsize=(12.5, 5.4))
    axA = fig.add_subplot(121, projection="3d")
    axB = fig.add_subplot(122, projection="3d")

    # Panel A — Axes as arrows + light tetrahedron wireframe
    origin = np.zeros(3)
    color_map = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    for name, vec in axes.items():
        axA.quiver(
            origin[0], origin[1], origin[2],
            vec[0], vec[1], vec[2],
            color=color_map[name],
            arrow_length_ratio=0.12,
            linewidth=2.2,
            length=1.0,
            normalize=True,
        )
        axA.text(vec[0] * 1.05, vec[1] * 1.05, vec[2] * 1.05, name, fontsize=12, fontweight="bold", color=color_map[name])

    P = np.stack([axes["A"], axes["B"], axes["C"], axes["D"]], axis=0)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        axA.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], c="gray", alpha=0.5, linewidth=1.2)
    axA.set_title("A — Quadray axes (A,B,C,D)")
    axA.set_xlabel("X")
    axA.set_ylabel("Y")
    axA.set_zlabel("Z")
    try:
        axA.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
    except Exception:
        pass
    axA.view_init(elev=18, azim=35)
    axA.text2D(0.02, 0.98, "A", transform=axA.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

    # Panel B — Close-packed spheres at tetrahedron vertices
    # Sphere mesh (unit sphere scaled by r, translated to each vertex)
    u = np.linspace(0, np.pi, 40)
    v = np.linspace(0, 2 * np.pi, 80)
    uu, vv = np.meshgrid(u, v)
    # Choose radius so neighboring vertex-centered spheres kiss along edges
    edge_lengths = [np.linalg.norm(P[i] - P[j]) for (i, j) in edges]
    r = 0.5 * float(min(edge_lengths))
    Xs = np.sin(uu) * np.cos(vv)
    Ys = np.sin(uu) * np.sin(vv)
    Zs = np.cos(uu)
    for (name, center) in zip(["A", "B", "C", "D"], P):
        axB.plot_surface(
            r * Xs + center[0], r * Ys + center[1], r * Zs + center[2],
            color=color_map[name], alpha=0.85, linewidth=0.0, shade=True, antialiased=True
        )
        axB.text(center[0] + 0.55 * r, center[1] + 0.55 * r, center[2] + 0.55 * r, name,
                 fontsize=12, fontweight="bold", color="black")
    # Wireframe edges for context
    for i, j in edges:
        axB.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], [P[i, 2], P[j, 2]], c="k", alpha=0.25, linewidth=1.0)
    axB.set_title("B — Tetrahedron as close-packed spheres (IVM/CCP/FCC)")
    axB.set_xlabel("X")
    axB.set_ylabel("Y")
    axB.set_zlabel("Z")
    try:
        axB.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
    except Exception:
        pass
    axB.view_init(elev=18, azim=35)
    axB.text2D(0.02, 0.98, "B", transform=axB.transAxes, va="top", ha="left", fontsize=14, fontweight="bold")

    # Reduce clutter
    for ax in (axA, axB):
        ax.grid(False)
        # Keep ticks sparse for legibility
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_tick_params(labelsize=9)

    outdir = get_output_dir()
    outpath = os.path.join(outdir, "graphical_abstract_quadray.png")
    # Avoid tight_layout on mixed text/3D; manual margins
    fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.06, wspace=0.08)
    fig.savefig(outpath, dpi=240)
    plt.close(fig)
    print(outpath)


if __name__ == "__main__":
    main()



