#!/usr/bin/env python3
"""Run a small quadray Nelder–Mead and save an MP4 animation.

Prints the saved file path on success.
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

    from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING  # noqa: WPS433
    from nelder_mead_quadray import nelder_mead_quadray  # noqa: WPS433
    from visualize import animate_simplex  # noqa: WPS433
    from visualize import plot_simplex_trace  # noqa: WPS433
    from visualize import plot_ivm_neighbors  # noqa: WPS433
    from paths import get_output_dir, get_data_dir, get_figure_dir  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433

    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [
        Quadray(5, 0, 0, 0),
        Quadray(4, 1, 0, 0),
        Quadray(0, 4, 1, 0),
        Quadray(1, 1, 1, 0),
    ]
    state = nelder_mead_quadray(f, initial, max_iter=12)

    path = animate_simplex(state.history, save=True)
    print(path)

    # Save per-iteration diagnostics trace plot and data
    trace_path = plot_simplex_trace(state, save=True)
    print(trace_path)

    # Keep static final simplex image for completeness, though manuscript will use trace
    final_vertices = state.vertices
    pts = [to_xyz(v, DEFAULT_EMBEDDING) for v in final_vertices]
    xs, ys, zs = zip(*pts)
    fig = plt.figure(figsize=(5.8, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c="tab:green", s=50)
    ax.plot(xs + (xs[0],), ys + (ys[0],), zs + (zs[0],), c="tab:green", alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Final simplex (static)")
    try:
        ax.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
    except Exception:
        pass
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    static_path = os.path.join(figure_dir, "simplex_final.png")
    fig.savefig(static_path, dpi=160)
    plt.close(fig)
    print(static_path)

    # Save a fresh IVM neighbors reference alongside
    ivm_path = plot_ivm_neighbors(save=True)
    print(ivm_path)


if __name__ == "__main__":
    main()


