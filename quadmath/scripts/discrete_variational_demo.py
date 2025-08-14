#!/usr/bin/env python3
"""Generate discrete variational optimization artifacts only.

Saves to quadmath/output/ and prints saved paths (MP4/PNG/CSV/NPZ).
"""
from __future__ import annotations

import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_src_on_path() -> None:
    src_path = os.path.join(_repo_root(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING  # noqa: WPS433
    from discrete_variational import discrete_ivm_descent  # noqa: WPS433
    from visualize import animate_discrete_path  # noqa: WPS433
    from paths import get_output_dir, get_data_dir, get_figure_dir  # noqa: WPS433

    # Simple convex objective in embedded coordinates
    def f(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        return float((x - 0.5) ** 2 + (y + 0.2) ** 2 + (z - 0.1) ** 2)

    dpath = discrete_ivm_descent(f, Quadray(6, 0, 0, 0))
    mp4_path = animate_discrete_path(dpath, save=True)
    print(mp4_path)

    # Also report auxiliary artifacts for convenience
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    print(os.path.join(figure_dir, "discrete_path_final.png"))
    print(os.path.join(data_dir, "discrete_path.csv"))
    print(os.path.join(data_dir, "discrete_path.npz"))


if __name__ == "__main__":
    main()


