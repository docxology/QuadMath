#!/usr/bin/env python3
"""Vector equilibrium panels: close-packed spheres and tensegrity stylization.

Panel A: Twelve-around-one close-packed spheres (FCC/CCP motif) at the
         IVM neighbor positions (permutations of {2,1,1,0}) plus a central sphere.
Panel B: Vector equilibrium (cuboctahedron) adjacency drawn as struts (edges)
         and light radial cables to the origin (stylized tensegrity view).
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
    from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING  # noqa: WPS433
    from paths import get_output_dir, get_data_dir, get_figure_dir  # noqa: WPS433
    import itertools  # noqa: WPS433

    # Neighbor positions (12 around origin) in Quadray integers
    neighbors = [Quadray(*p) for p in sorted({p for p in itertools.permutations((2, 1, 1, 0))})]
    xyz = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in neighbors], dtype=float)

    # Figure
    fig = plt.figure(figsize=(10.5, 5.0))
    axA = fig.add_subplot(121, projection="3d")
    axB = fig.add_subplot(122, projection="3d")

    # Helper: equal aspect
    def _set_axes_equal(ax) -> None:
        try:
            ax.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
            return
        except Exception:
            pass
        limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
        spans = [abs(l[1] - l[0]) for l in limits]
        centers = [(l[0] + l[1]) / 2.0 for l in limits]
        radius = max(spans) / 2.0
        ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
        ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
        ax.set_zlim3d(centers[2] - radius, centers[2] + radius)

    # Panel A: close-packed spheres
    dists = np.linalg.norm(xyz, axis=1)
    r = 0.5 * float(np.min(dists))  # kissing radius
    u = np.linspace(0, np.pi, 36)
    v = np.linspace(0, 2 * np.pi, 72)
    uu, vv = np.meshgrid(u, v)
    Xs = np.sin(uu) * np.cos(vv)
    Ys = np.sin(uu) * np.sin(vv)
    Zs = np.cos(uu)
    # Central sphere
    axA.plot_surface(r * Xs, r * Ys, r * Zs, color="#bbbbbb", alpha=0.92, linewidth=0)
    # 12 neighbors
    for (x, y, z) in xyz:
        axA.plot_surface(r * Xs + x, r * Ys + y, r * Zs + z, color="#1f77b4", alpha=0.88, linewidth=0)
    axA.set_title("A — Vector equilibrium as close-packed spheres")
    axA.set_xlabel("X")
    axA.set_ylabel("Y")
    axA.set_zlabel("Z")
    _set_axes_equal(axA)

    # Panel B: adjacency (struts) + radial cables to origin
    axB.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="tab:blue")
    # Struts between touching neighbors
    n = xyz.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(xyz[i] - xyz[j])
            if abs(dij - 2.0 * r) / (2.0 * r) < 0.05:
                axB.plot([xyz[i, 0], xyz[j, 0]], [xyz[i, 1], xyz[j, 1]], [xyz[i, 2], xyz[j, 2]],
                         c="#555555", alpha=0.7, linewidth=1.2)
    # Radial cables
    for x, y, z in xyz:
        axB.plot([0, x], [0, y], [0, z], c="#aaaaaa", alpha=0.5, linewidth=0.9)
    axB.set_title("B — Vector equilibrium (struts) with radial cables")
    axB.set_xlabel("X")
    axB.set_ylabel("Y")
    axB.set_zlabel("Z")
    _set_axes_equal(axB)

    figure_dir = get_figure_dir()
    outpath = os.path.join(figure_dir, "vector_equilibrium_panels.png")
    # Avoid tight_layout on 3D axes; set margins explicitly
    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.06, wspace=0.08)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(outpath)


if __name__ == "__main__":
    main()


