#!/usr/bin/env python3
"""Generate improved IVM neighbor visualizations (2x2 panel) and save outputs.

Panels:
- A: IVM neighbors as points
- B: Neighbors with radial edges from origin
- C: Twelve-around-one close-packed spheres (central + 12 neighbors)
- D: Vector equilibrium adjacency (edges where sphere centers touch)
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

    from visualize import plot_ivm_neighbors  # noqa: WPS433
    from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING  # noqa: WPS433
    from paths import get_output_dir, get_data_dir, get_figure_dir  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433
    import numpy as np  # noqa: WPS433

    outpath = plot_ivm_neighbors(save=True)
    print(outpath)

    # Helper for equal aspect on 3D axes
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

    # Compute neighbor points
    import itertools  # noqa: WPS433
    base = [2, 1, 1, 0]
    perms = sorted({p for p in itertools.permutations(base)})
    points = [Quadray(*p) for p in perms]
    xyz = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in points], dtype=float)

    # Build 2x2 panel figure
    fig = plt.figure(figsize=(10.5, 9.0))
    axA = fig.add_subplot(221, projection="3d")
    axB = fig.add_subplot(222, projection="3d")
    axC = fig.add_subplot(223, projection="3d")
    axD = fig.add_subplot(224, projection="3d")

    # Panel A: points only
    axA.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="tab:blue")
    axA.set_title("A — IVM neighbors (points)")
    axA.set_xlabel("X")
    axA.set_ylabel("Y")
    axA.set_zlabel("Z")
    _set_axes_equal(axA)

    # Panel B: radial edges
    axB.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="tab:blue")
    for x, y, z in xyz:
        axB.plot([0, x], [0, y], [0, z], c="gray", alpha=0.55, linewidth=1.3)
    axB.set_title("B — Neighbors with radial edges")
    axB.set_xlabel("X")
    axB.set_ylabel("Y")
    axB.set_zlabel("Z")
    _set_axes_equal(axB)

    # Panel C: twelve-around-one close-packed spheres
    # Determine kissing radius from central-to-neighbor distances
    dists = np.linalg.norm(xyz, axis=1)
    r = 0.5 * float(np.min(dists))
    u = np.linspace(0, np.pi, 40)
    v = np.linspace(0, 2 * np.pi, 80)
    uu, vv = np.meshgrid(u, v)
    Xs = np.sin(uu) * np.cos(vv)
    Ys = np.sin(uu) * np.sin(vv)
    Zs = np.cos(uu)
    # Central sphere
    axC.plot_surface(r * Xs, r * Ys, r * Zs, color="#bbbbbb", alpha=0.9, linewidth=0.0, shade=True)
    # 12 neighbors
    for (x, y, z) in xyz:
        axC.plot_surface(r * Xs + x, r * Ys + y, r * Zs + z, color="#1f77b4", alpha=0.85, linewidth=0.0, shade=True)
    axC.set_title("C — Twelve around one (close-packed)")
    axC.set_xlabel("X")
    axC.set_ylabel("Y")
    axC.set_zlabel("Z")
    _set_axes_equal(axC)

    # Panel D: vector equilibrium adjacency (touching centers)
    axD.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="tab:blue")
    # Connect neighbor centers whose separation is ~2r (touching)
    n = xyz.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(xyz[i] - xyz[j])
            if abs(dij - 2.0 * r) < 1e-6 or abs(dij - 2.0 * r) / (2.0 * r) < 0.05:
                axD.plot([xyz[i, 0], xyz[j, 0]], [xyz[i, 1], xyz[j, 1]], [xyz[i, 2], xyz[j, 2]], c="gray", alpha=0.6, linewidth=1.0)
    # Also draw central connections lightly
    for x, y, z in xyz:
        axD.plot([0, x], [0, y], [0, z], c="lightgray", alpha=0.4, linewidth=0.8)
    axD.set_title("D — Vector equilibrium (adjacency)")
    axD.set_xlabel("X")
    axD.set_ylabel("Y")
    axD.set_zlabel("Z")
    _set_axes_equal(axD)

    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    outpath2 = os.path.join(figure_dir, "ivm_neighbors_edges.png")
    # Avoid tight_layout with 3D subplots; set margins manually
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.06, wspace=0.08, hspace=0.12)
    fig.savefig(outpath2, dpi=220)
    plt.close(fig)
    print(outpath2)

    # Save raw data
    import csv  # noqa: WPS433
    q_arr = np.array([p.as_tuple() for p in points], dtype=int)
    xyz_arr = np.array(xyz, dtype=float)
    np.savez(os.path.join(data_dir, "ivm_neighbors_edges_data.npz"), quadrays=q_arr, xyz=xyz_arr, radius=r)
    # Also provide a simple CSV of neighbor coordinates (both Quadray and XYZ)
    csv_path = os.path.join(data_dir, "ivm_neighbors_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "c", "d", "x", "y", "z"])
        for (a, b, c, d), (x, y, z) in zip(q_arr.tolist(), xyz_arr.tolist()):
            writer.writerow([int(a), int(b), int(c), int(d), float(x), float(y), float(z)])
    print(csv_path)


if __name__ == "__main__":
    main()


