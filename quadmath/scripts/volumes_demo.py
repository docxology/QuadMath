#!/usr/bin/env python3
"""Demonstrate integer volumes and Cayley–Menger on simple tetrahedra.

Saves no figures, but prints numerical results; useful for quick checks.
"""
from __future__ import annotations

import os
import sys
import numpy as np


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    from quadray import Quadray, integer_tetra_volume  # noqa: WPS433
    from cayley_menger import tetra_volume_cayley_menger, ivm_tetra_volume_cayley_menger  # noqa: WPS433
    from paths import get_output_dir  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433
    import csv  # noqa: WPS433

    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    v_ivm = integer_tetra_volume(p0, p1, p2, p3)
    print(f"IVM unit tetra volume (integer): {v_ivm}")

    d2 = np.ones((4, 4)) - np.eye(4)
    v_xyz = tetra_volume_cayley_menger(d2)
    print(f"Regular tetra volume (XYZ units): {v_xyz:.8f} (expected sqrt(2)/12 ≈ {np.sqrt(2)/12:.8f})")

    # Scale study: volumes vs edge scale (XYZ) and converted to IVM via S3
    # Use a broad range and include s=0 for reference; x-axis [0, 5]
    scales = np.linspace(0.0, 5.0, 21)
    v_xyz_list = []
    v_ivm_list = []
    S3 = np.sqrt(9.0 / 8.0)
    base_d2 = np.ones((4, 4)) - np.eye(4)
    for s in scales:
        d2_s = (s * s) * base_d2
        v_xyz_s = tetra_volume_cayley_menger(d2_s)
        v_xyz_list.append(v_xyz_s)
        v_ivm_list.append(ivm_tetra_volume_cayley_menger(d2_s))

    # Consistent styling with manuscript (DejaVu Serif) and improved legibility
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(scales, v_xyz_list, label=r"$V_{xyz}$ (Euclidean)", marker="o")
    ax.plot(scales, v_ivm_list, label=r"$V_{ivm}=S3\cdot V_{xyz}$", marker="s")
    ax.set_xlabel(r"Edge scale $s$")
    ax.set_ylabel(r"Volume")
    ax.set_title("Tetra volume vs edge scale")
    ax.set_xlim(0.0, 5.0)
    ax.set_xticks(np.arange(0.0, 5.0 + 1e-9, 1.0))
    ax.grid(True, which="both", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left")
    # Add small S3 reference consistent with glossary
    ax.text(0.02, 0.98, r"$S3=\sqrt{9/8}$", transform=ax.transAxes, va="top", ha="left")
    outdir = get_output_dir()
    outpath = os.path.join(outdir, "volumes_scale_plot.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    # Also save as vector graphics for print-quality
    fig.savefig(os.path.join(outdir, "volumes_scale_plot.svg"))
    print(outpath)

    # Save raw data alongside the figure for reproducibility
    np.savez(
        os.path.join(outdir, "volumes_scale_data.npz"),
        scales=np.array(scales, dtype=float),
        V_xyz=np.array(v_xyz_list, dtype=float),
        V_ivm=np.array(v_ivm_list, dtype=float),
    )
    with open(os.path.join(outdir, "volumes_scale_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "V_xyz", "V_ivm"])
        for s, vx, vi in zip(scales, v_xyz_list, v_ivm_list):
            writer.writerow([float(s), float(vx), float(vi)])
    print(os.path.join(outdir, "volumes_scale_data.csv"))


if __name__ == "__main__":
    main()


