#!/usr/bin/env python3
"""Visualize random quadray points under different embeddings.

Saves a static PNG of two point clouds using the default and scaled embeddings.
"""
from __future__ import annotations

import os
import sys
import random
from typing import List


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    import matplotlib.pyplot as plt  # noqa: WPS433
    from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING  # noqa: WPS433
    from paths import get_output_dir  # noqa: WPS433
    from conversions import urner_embedding, quadray_to_xyz  # noqa: WPS433
    import numpy as np  # noqa: WPS433

    # Deterministic sampling
    random.seed(0)

    def sample_points(n: int) -> List[Quadray]:
        pts: List[Quadray] = []
        for _ in range(n):
            # Non-negative components with at least one zero enforced by normalize
            a, b, c, d = (random.randint(0, 5) for _ in range(4))
            pts.append(Quadray(a, b, c, d).normalize())
        return pts

    pts = sample_points(200)
    xyz1 = [to_xyz(q, DEFAULT_EMBEDDING) for q in pts]
    scaled = tuple(tuple(0.75 * v for v in row) for row in DEFAULT_EMBEDDING)
    xyz2 = [to_xyz(q, scaled) for q in pts]

    # Compare with Urner embedding on the same points
    M = urner_embedding(scale=1.0)
    xyz3 = [quadray_to_xyz(q, M) for q in pts]

    fig = plt.figure(figsize=(15, 4.5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    xs, ys, zs = zip(*xyz1)
    ax1.scatter(xs, ys, zs, s=10, c="tab:blue")
    ax1.set_title("Default embedding")

    xs, ys, zs = zip(*xyz2)
    ax2.scatter(xs, ys, zs, s=10, c="tab:orange")
    ax2.set_title("Scaled embedding (0.75x)")

    xs, ys, zs = zip(*xyz3)
    ax3.scatter(xs, ys, zs, s=10, c="tab:purple")
    ax3.set_title("Urner embedding")

    outdir = get_output_dir()
    outpath = os.path.join(outdir, "quadray_clouds.png")
    # Avoid tight_layout on 3D subplots; set margins manually
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.08)
    fig.savefig(outpath, dpi=160)
    print(outpath)
    plt.close(fig)

    # Save raw points and embeddings
    q_arr = np.array([q.as_tuple() for q in pts], dtype=int)
    xyz1_arr = np.array(xyz1, dtype=float)
    xyz2_arr = np.array(xyz2, dtype=float)
    xyz3_arr = np.array(xyz3, dtype=float)
    emb_default = np.array(DEFAULT_EMBEDDING, dtype=float)
    emb_scaled = np.array(scaled, dtype=float)
    emb_urner = np.array(urner_embedding(scale=1.0), dtype=float)
    np.savez(
        os.path.join(outdir, "quadray_clouds_data.npz"),
        quadrays=q_arr,
        xyz_default=xyz1_arr,
        xyz_scaled=xyz2_arr,
        xyz_urner=xyz3_arr,
        embedding_default=emb_default,
        embedding_scaled=emb_scaled,
        embedding_urner=emb_urner,
    )
    print(os.path.join(outdir, "quadray_clouds_data.npz"))


if __name__ == "__main__":
    main()


