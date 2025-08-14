#!/usr/bin/env python3
"""Polyhedra relationships panel (synergetics volumes and mappings).

Produces a compact 2D diagram summarizing unit relationships used in the text:
regular tetra (1), cube (3), octahedron (4), rhombic dodecahedron (6),
cuboctahedron (20). Arrows indicate simple volume relations in the synergetics
convention and mapping via Quadray integer-coordinate constructions.
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
    from paths import get_output_dir, get_data_dir, get_figure_dir  # noqa: WPS433

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_facecolor("#ffffff")
    ax.axis("off")
    # Manually manage margins to avoid layout warnings
    fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.1)

    # Node helper
    def node(x: float, y: float, text: str, fc: str = "#f5f5f5") -> None:
        rect = plt.Rectangle((x - 0.8, y - 0.35), 1.6, 0.7, fc=fc, ec="#444444", lw=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=11, zorder=3)

    # Arrows
    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"), zorder=1)
        if label:
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax.text(xm, ym + 0.12, label, fontsize=10, ha="center", va="bottom", zorder=3)

    # Layout
    node(-3.0, 0.8, "Tetra\nV=1")
    node(-0.5, 0.8, "Cube\nV=3")
    node(2.0, 0.8, "Octa\nV=4")
    node(2.0, -0.8, "Rhombic\nDodeca\nV=6")
    node(-0.5, -0.8, "Cubocta\n(Vector Eq.)\nV=20")

    # Relations (schematic)
    arrow(-2.3, 0.8, -1.2, 0.8, "x3")  # tetra -> cube
    arrow(0.2, 0.8, 1.3, 0.8, "+1")    # cube -> octa (heuristic mapping)
    arrow(2.0, 0.5, 2.0, -0.5, "edge-union")
    arrow(-0.5, -0.5, -0.5, 0.5, "shell of 12")
    arrow(-2.3, 0.65, -1.0, -0.65, "neighbors")

    # Note: schematic relations are descriptive; integer-coordinate
    # constructions are discussed in the text under Quadray methods.
    ax.text(-3.4, -1.5,
            "Synergetics tetravolumes in IVM units. Nodes show volumes (1,3,4,6,20).\n"
            "Arrows: cube ~ 3× tetra; octa as edge-mid union; rhombic dodeca as Voronoi cell;\n"
            "cubocta is shell of 12 nearest IVM neighbors (permutations of (2,1,1,0)).",
            fontsize=9, ha="left", va="top")

    # Ensure content is within view; explicit limits avoid blank renders
    ax.set_xlim(-4.0, 3.0)
    ax.set_ylim(-1.8, 1.6)

    figure_dir = get_figure_dir()
    outpath = os.path.join(figure_dir, "polyhedra_quadray_constructions.png")
    fig.savefig(outpath, dpi=240)
    print(outpath)


if __name__ == "__main__":
    main()


