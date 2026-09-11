#!/usr/bin/env python3
"""Quaternion slerp gallery (thin orchestrator).

Delegates to ``src/quadmath/viz/plots.py::plot_slerp_path`` per the
thin-orchestrator contract in ``quadmath/scripts/AGENTS.md``; sets a headless
backend and traces the shortest-arc slerp geodesic from the identity
quaternion to a pi/2 rotation about the z axis applied to the default
(2, 0, 0, 0) lattice site, writing ``quaternion_slerp_path.png`` under
``quadmath/output/figures/`` and printing the output path.  The stdout path
line is the ``make_all_figures`` manifest contract (only path lines on
stdout).

The target quaternion is the unit quaternion
``(cos(pi/4), 0, 0, sin(pi/4))`` built from the half-angle form
``q = (cos(theta/2), sin(theta/2) * axis_hat)`` that
``quadmath.core.quadray.rotate_about_axis`` documents for axis-angle input,
so ``slerp`` interpolates the rotation angle linearly from 0 to pi/2.
Deterministic by construction: no RNG and no wall-clock inputs.
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

    from quadmath.viz.plots import plot_slerp_path  # noqa: WPS433

    q0 = (1.0, 0.0, 0.0, 0.0)
    half = math.pi / 4.0
    q1 = (math.cos(half), 0.0, 0.0, math.sin(half))
    print(plot_slerp_path(q0, q1))


if __name__ == "__main__":
    main()