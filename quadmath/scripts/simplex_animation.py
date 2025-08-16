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
    import numpy as np  # noqa: WPS433

    def f(q: Quadray) -> float:
        # Create a more challenging objective function that takes longer to converge
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        
        # Multiple local minima with barriers
        obj = (x - 2)**2 + (y - 2)**2 + (z - 2)**2  # Main minimum at (2,2,2)
        
        # Add some complexity with multiple basins
        if x < 0 and y < 0:
            obj += 5.0  # Penalty for negative quadrant
        if abs(x) > 4 or abs(y) > 4 or abs(z) > 4:
            obj += 10.0  # Penalty for large coordinates
            
        # Add some noise-like variation to make convergence more interesting
        obj += 0.1 * abs(x + y + z)
        
        return obj

    initial = [
        Quadray(5, 0, 0, 0),
        Quadray(4, 1, 0, 0),
        Quadray(0, 4, 1, 0),
        Quadray(1, 1, 1, 0),
    ]
    # Extend optimization to ensure we get meaningful iterations
    state = nelder_mead_quadray(f, initial, max_iter=20)

    path = animate_simplex(state.history, save=True)
    print(path)

    # Save per-iteration diagnostics trace plot and data
    trace_path = plot_simplex_trace(state, save=True)
    print(trace_path)

    # Create improved 2x2 panel showing simplex evolution at key iterations
    # Select iterations that show the actual convergence process
    if len(state.history) >= 12:
        key_iterations = [0, 3, 6, 9]  # Show initial, early convergence, mid-convergence, and final state
    elif len(state.history) >= 8:
        key_iterations = [0, 2, 4, 8]   # Show initial, early, mid, and convergence
    else:
        key_iterations = [0, 1, 2, len(state.history) - 1]  # Show all available iterations
    
    fig = plt.figure(figsize=(12, 10))
    
    for i, iteration in enumerate(key_iterations):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        
        if iteration < len(state.history):
            vertices = state.history[iteration]
            pts = [to_xyz(v, DEFAULT_EMBEDDING) for v in vertices]
            xs, ys, zs = zip(*pts)
            
            # Plot vertices as scatter points
            ax.scatter(xs, ys, zs, c="tab:red", s=60, alpha=0.8)
            
            # Connect vertices to show tetrahedron edges
            # Create a cycle to connect all vertices
            for j in range(4):
                for k in range(j + 1, 4):
                    ax.plot([xs[j], xs[k]], [ys[j], ys[k]], [zs[j], zs[k]], 
                           c="tab:blue", alpha=0.6, linewidth=1.5)
            
            # Add iteration info and objective values
            obj_values = [f(v) for v in vertices]
            best_val = min(obj_values)
            worst_val = max(obj_values)
            spread = worst_val - best_val
            
            # Check if this iteration shows convergence
            if best_val == 0 and spread == 0:
                convergence_status = " (CONVERGED)"
            else:
                convergence_status = ""
            
            ax.set_title(f"Iteration {iteration}{convergence_status}\nBest: {best_val:.1f}, Spread: {spread:.1f}")
        else:
            # Handle case where we don't have enough iterations
            ax.text(0.5, 0.5, 0.5, f"Iteration {iteration}\nNot reached", 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Set consistent view limits across all subplots
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        
        # Try to set equal aspect ratio
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    
    fig.suptitle("Nelder-Mead Simplex Evolution on Integer Quadray Lattice", fontsize=16, y=0.98)
    fig.tight_layout()
    
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    static_path = os.path.join(figure_dir, "simplex_final.png")
    fig.savefig(static_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(static_path)

    # Create a comprehensive trace visualization showing all vertex paths
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection="3d")
    
    # Plot the complete trace of each vertex across all iterations
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    markers = ['o', 's', '^', 'D']
    
    for vertex_idx in range(4):  # For each of the 4 vertices
        x_trace = []
        y_trace = []
        z_trace = []
        
        for iteration in range(len(state.history)):
            if iteration < len(state.history):
                vertex = state.history[iteration][vertex_idx]
                x, y, z = to_xyz(vertex, DEFAULT_EMBEDDING)
                x_trace.append(x)
                y_trace.append(y)
                z_trace.append(z)
        
        # Plot the trace line
        ax.plot(x_trace, y_trace, z_trace, c=colors[vertex_idx], alpha=0.7, linewidth=2,
                label=f'Vertex {vertex_idx+1}')
        
        # Mark key points along the trace
        # Select meaningful iterations to mark based on actual convergence
        if len(state.history) >= 12:
            key_markers = [0, 4, 8, 12]
        elif len(state.history) >= 8:
            key_markers = [0, 2, 4, 8]
        else:
            key_markers = [0, len(state.history) - 1]
            
        for i, (x, y, z) in enumerate(zip(x_trace, y_trace, z_trace)):
            if i in key_markers:  # Mark key iterations
                ax.scatter(x, y, z, c=colors[vertex_idx], s=80, marker=markers[vertex_idx], 
                          edgecolors='black', linewidth=1)
            else:
                ax.scatter(x, y, z, c=colors[vertex_idx], s=30, alpha=0.6)
    
    # Mark the final converged point (origin) prominently
    ax.scatter(0, 0, 0, c='black', s=200, marker='*', edgecolors='white', linewidth=2,
               label='Converged (0,0,0)', zorder=10)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Complete Simplex Optimization Trace\nNelder-Mead on Integer Quadray Lattice")
    ax.legend()
    
    # Set consistent view limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    
    fig2.tight_layout()
    
    trace_vis_path = os.path.join(figure_dir, "simplex_trace_visualization.png")
    fig2.savefig(trace_vis_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(trace_vis_path)

    # Save a fresh IVM neighbors reference alongside
    ivm_path = plot_ivm_neighbors(save=True)
    print(ivm_path)


if __name__ == "__main__":
    main()


