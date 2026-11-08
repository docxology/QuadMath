#!/usr/bin/env python3
"""Render closed, measured polyhedra through the importable visualization API."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    source = str(Path(__file__).resolve().parents[2] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from quadmath.viz.vis_polyhedra import render_polyhedra_constructions

    for path in render_polyhedra_constructions():
        print(f"Closed polyhedra figure saved to: {path}")


if __name__ == "__main__":
    main()
