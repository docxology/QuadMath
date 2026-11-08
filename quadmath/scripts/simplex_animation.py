#!/usr/bin/env python3
"""Render integer-Quadray Nelder–Mead history, diagnostics and final recorded best.

The piecewise quadratic objective includes penalties and a deterministic
regularizer. Neither a zero objective nor convergence to the origin is assumed.
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

    import numpy as np
    from quadmath.core.quadray import Quadray, to_xyz, DEFAULT_EMBEDDING
    from quadmath.optimize.nelder_mead_quadray import nelder_mead_quadray
    from quadmath.viz._common import figure_scope, save_figure
    from quadmath.viz.visualize import animate_simplex, plot_simplex_trace, plot_ivm_neighbors
    from quadmath.paths import get_figure_dir

    def objective(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        value = (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2
        if x < 0 and y < 0:
            value += 5.0
        if abs(x) > 4 or abs(y) > 4 or abs(z) > 4:
            value += 10.0
        return value + 0.1 * abs(x + y + z)

    initial = [
        Quadray(5, 0, 0, 0),
        Quadray(4, 1, 0, 0),
        Quadray(0, 4, 1, 0),
        Quadray(1, 1, 1, 0),
    ]
    state = nelder_mead_quadray(objective, initial, max_iter=20)
    print(animate_simplex(state.history, save=True))
    print(plot_simplex_trace(state, save=True))

    points = np.asarray([
        [to_xyz(vertex, DEFAULT_EMBEDDING) for vertex in vertices]
        for vertices in state.history
    ], dtype=float)
    final_best = min(state.history[-1], key=objective)
    best_xyz = np.asarray(to_xyz(final_best, DEFAULT_EMBEDDING), dtype=float)
    last = len(state.history) - 1
    snapshots = sorted({round(index * last / 3) for index in range(4)})
    lower, upper = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    center = (lower + upper) / 2
    radius = 1.1 * max(float(np.max(upper - lower)) / 2, 0.5)

    def label_axes(ax) -> None:
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

    figure_dir = get_figure_dir()
    static_path = os.path.join(figure_dir, "simplex_final.png")
    with figure_scope(figsize=(12, 10), layout="constrained") as fig:
        for panel, frame in enumerate(snapshots, 1):
            ax = fig.add_subplot(2, 2, panel, projection="3d")
            xyz = points[frame]
            ax.scatter(*xyz.T, c="tab:red", s=60)
            for first in range(4):
                for second in range(first + 1, 4):
                    ax.plot(*xyz[[first, second]].T, c="tab:blue", alpha=0.6)
            values = [objective(vertex) for vertex in state.history[frame]]
            ax.set_title(
                f"Recorded frame {frame} of {last}\n"
                f"Best objective {min(values):.3g}; range {max(values) - min(values):.3g}"
            )
            label_axes(ax)
        fig.suptitle("Integer-Quadray simplex snapshots: initial through final recorded state")
        save_figure(fig, static_path, dpi=160, bbox_inches="tight")
    print(static_path)

    trace_path = os.path.join(figure_dir, "simplex_trace_visualization.png")
    with figure_scope(figsize=(10, 8), layout="constrained") as fig:
        ax = fig.add_subplot(111, projection="3d")
        for slot, color in enumerate(("tab:red", "tab:blue", "tab:green", "tab:orange")):
            xyz = points[:, slot]
            ax.plot(*xyz.T, color=color, alpha=0.7, linewidth=2, label=f"Simplex slot {slot + 1}")
            ax.scatter(*xyz[snapshots].T, color=color, s=55, edgecolors="black")
        ax.scatter(
            *best_xyz, c="black", s=180, marker="*", edgecolors="white",
            label=f"Final recorded best: f = {objective(final_best):.3g}",
        )
        label_axes(ax)
        ax.set_title("Recorded simplex slots in scale-one XYZ\nSlot order is not persistent vertex identity")
        ax.legend()
        save_figure(fig, trace_path, dpi=160, bbox_inches="tight")
    print(trace_path)
    print(f"Final recorded best XYZ: {tuple(best_xyz)}; objective: {objective(final_best):.6g}")
    print(plot_ivm_neighbors(save=True))


if __name__ == "__main__":
    main()


