#!/usr/bin/env python3
"""Animation gallery (thin orchestrator).

Renders three deterministic animation GIFs over the IVM lattice via
``src/quadmath/viz/animations.py``:

- ``animation_lattice.gif``: pulsing IVM ball (radius 3), 12 frames.
- ``animation_simplex.gif``: quaternion-slerp rotation of the radius-1
  ball from the identity to a 45-degree rotation about z, 16 frames.
- ``animation_diffusion.gif``: heat diffusion on the radius-3 ball
  adjacency, 12 frames (seed 0).

Writes the GIFs under ``quadmath/output/figures/`` and prints each output
path on its own line.  The stdout path lines are the ``make_all_figures``
manifest contract (only path lines on stdout).
"""
from __future__ import annotations

import math
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

    from quadmath.paths import get_figure_dir  # noqa: WPS433

    from quadmath.viz.animations import (  # noqa: WPS433
        diffusion_frames,
        frames_to_gif,
        lattice_frames,
        simplex_frames,
    )

    quarter_root = math.sqrt(0.5)
    qa = (1.0, 0.0, 0.0, 0.0)
    qb = (quarter_root, quarter_root, 0.0, 0.0)

    galleries = (
        ("animation_lattice.gif", lattice_frames(shells=3, n=12)),
        ("animation_simplex.gif", simplex_frames(qa, qb, n=16)),
        ("animation_diffusion.gif", diffusion_frames(n_steps=12, seed=0)),
    )
    for name, frames in galleries:
        out_path = frames_to_gif(frames, os.path.join(get_figure_dir(), name), fps=8, scale=8)
        print(out_path)


if __name__ == "__main__":
    main()