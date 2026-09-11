#!/usr/bin/env python3
"""Animation stills strip (thin orchestrator).

Samples one evenly spaced still from each of the three deterministic
animation sequences of ``src/quadmath/viz/animations.py`` and composes them
into the single-row PNG ``animation_frames_strip.png`` via ``frames_strip``:

- left: pulsing IVM ball (``lattice_frames``, radius 2, 6 frames) sampled
  at the quarter-pulse fraction (grown ball),
- center: quaternion-slerp rotation of the radius-1 ball
  (``simplex_frames``, 6 frames) from the identity to a pi/2 rotation
  about +z, sampled in the midpoint region of the arc; the endpoint
  quaternion is derived from ``quadmath.core.quadray.rotate_about_axis``,
- right: heat diffusion on the radius-3 ball adjacency
  (``diffusion_frames``, 6 frames, seed 0) sampled at the final step.

Writes the strip under ``quadmath/output/figures/`` and prints the output
path on its own line.  The stdout path line is the ``make_all_figures``
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

    from quadmath.core.quadray import rotate_about_axis  # noqa: WPS433

    from quadmath.viz.animations import (  # noqa: WPS433
        diffusion_frames,
        frames_strip,
        lattice_frames,
        simplex_frames,
    )

    # q1: the unit quaternion of the pi/2 rotation about +z, rebuilt from
    # rotate_about_axis's axis-angle construction q = (cos(angle/2),
    # sin(angle/2) * axis_hat): rotate the +x basis vector and read the
    # rotation angle back off the rotated vector.
    rotated_x = rotate_about_axis((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2.0)
    half_angle = 0.5 * math.atan2(rotated_x[1], rotated_x[0])
    q1 = (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))

    lattice = lattice_frames(shells=2, n=6)
    simplex = simplex_frames((1.0, 0.0, 0.0, 0.0), q1, n=6)
    diffusion = diffusion_frames(n_steps=6, seed=0)

    # Evenly spaced stills: quarter-pulse lattice frame, midpoint-region
    # slerp frame, and the final diffusion frame.
    stills = (
        lattice[int(round(0.25 * (len(lattice) - 1)))],
        simplex[int(round(0.5 * (len(simplex) - 1)))],
        diffusion[int(round(1.0 * (len(diffusion) - 1)))],
    )
    out_path = frames_strip(
        stills,
        "animation_frames_strip.png",
        labels=("lattice pulse", "simplex slerp", "diffusion end"),
    )
    print(out_path)


if __name__ == "__main__":
    main()