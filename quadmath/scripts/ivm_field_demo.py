#!/usr/bin/env python3
"""Static IVM field learning demo.

Renders a three-panel figure of scalar field learning on the IVM lattice:

- Panel A: the IVM lattice ball (radius 3) as embedded points, colored by
  a synthetic smooth field.
- Panel B: noisy observations sampled at half the sites (fixed seed).
- Panel C: the learned field (Laplacian-regularized kernel-weighted least
  squares on the lattice graph) at every site.

The right column prints reconstruction MSE against the synthetic truth.
Saves ``quadmath/output/figures/ivm_field_demo.png`` and prints the path.
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

    import matplotlib.pyplot as plt  # noqa: WPS433
    import numpy as np  # noqa: WPS433
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,WPS433  (registers 3d projection)

    from ivm_field import IVMField, ball_sites  # noqa: WPS433
    from quadray import DEFAULT_EMBEDDING, to_xyz  # noqa: WPS433
    from paths import get_figure_dir  # noqa: WPS433

    rng = np.random.default_rng(12)
    radius = 3

    sites = ball_sites(radius)
    embedding = np.array(DEFAULT_EMBEDDING, dtype=float)
    xyz = np.array([to_xyz(q, embedding) for q in sites], dtype=float)

    # Synthetic smooth field: linear (harmonic) part plus weak quadratic bowl.
    truth = np.array(
        [2.0 + 0.3 * p[0] - 0.2 * p[1] + 0.1 * p[2] - 0.008 * float(p @ p) for p in xyz]
    )

    # Observe half the sites with N(0, 0.3) noise.
    mask = rng.random(len(sites)) < 0.5
    obs_sites = [q for q, m in zip(sites, mask) if m]

    noise = rng.normal(0.0, 0.3, size=int(mask.sum()))
    obs_values = [truth[i] + n for i, n in zip(np.flatnonzero(mask), noise)]
    raw_mse = float(np.mean(noise**2))

    field = IVMField.lattice_ball(radius)
    field.learn(obs_sites, obs_values, lam=0.01, kernel_width=0.5)
    learned = np.array([field.predict(q) for q in sites], dtype=float)
    recon_mse = float(np.mean((learned - truth) ** 2))

    fig = plt.figure(figsize=(15.0, 5.2))
    panels = [
        ("A — synthetic field", xyz, truth, "truth"),
        ("B — noisy observations", xyz[mask], np.asarray(obs_values), "observed"),
        ("C — learned field", xyz, learned, "learned"),
    ]
    for ax_i, (title, pts, vals, tag) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, ax_i, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap="viridis", s=26)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        try:
            ax.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
        except Exception:
            pass
        if tag == "learned":
            fig.colorbar(sc, ax=ax, shrink=0.75, label="field value")

    fig.suptitle(
        f"Static IVM field learning — ball radius {radius}, "
        f"{len(obs_sites)}/{len(sites)} sites observed (seed 12)\n"
        f"observation noise MSE {raw_mse:.4f}  ->  learned reconstruction MSE {recon_mse:.4f}",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.12)

    outpath = os.path.join(get_figure_dir(), "ivm_field_demo.png")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(outpath)


if __name__ == "__main__":
    main()
