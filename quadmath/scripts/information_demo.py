#!/usr/bin/env python3
"""Generate illustrative figures for information theory components.

This script demonstrates Fisher information matrices in the context of the three 4D frameworks:
- Coxeter.4D (Euclidean): Standard Cartesian parameter space
- Einstein.4D (Minkowski): Information geometry with Fisher metric
- Fuller.4D (Synergetics): Quadray coordinate transformations

Saves to quadmath/output/ and prints saved paths.
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

    from paths import get_output_dir, get_data_dir, get_figure_dir  # noqa: WPS433
    from information import fisher_information_matrix, natural_gradient_step, free_energy  # noqa: WPS433
    from discrete_variational import discrete_ivm_descent  # noqa: WPS433
    from quadray import Quadray, DEFAULT_EMBEDDING, to_xyz  # noqa: WPS433
    from visualize import animate_discrete_path  # noqa: WPS433
    from metrics import fisher_curvature_analysis  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433
    import matplotlib.patches as patches  # noqa: WPS433
    from mpl_toolkits.mplot3d import Axes3D  # noqa: WPS433

    # Set style for professional appearance
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    rng = np.random.default_rng(0)

    # Fisher Information example: sample gradients from a noisy quadratic model
    #
    # Create a synthetic linear regression dataset with additive noise and
    # compute per-sample gradients of the squared loss with respect to a
    # deliberately misspecified parameter vector. This avoids the degenerate
    # zero-gradient case and produces a meaningful empirical FIM.
    w_true = np.array([1.0, -2.0, 0.5])
    X = rng.normal(size=(200, 3))
    noise = 0.1 * rng.normal(size=X.shape[0])
    y = X @ w_true + noise
    # Evaluate gradients at an estimate w_est that differs from w_true
    w_est = np.array([0.3, -1.2, 0.0])
    residuals = X @ w_est - y  # shape (N,)
    # gradients of squared loss wrt params for each sample: 2 * x_i * r_i
    grads = 2.0 * (X.T * residuals).T  # shape (N, 3)
    F = fisher_information_matrix(grads)

    # Enhanced Figure 10: Fisher Information Matrix (FIM) with 4D Framework Context
    # Now with 3 panels including linear regression model visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Panel 1: Linear Regression Model with Diagnostics
    ax1 = fig.add_subplot(1, 3, 1)
    
    # Plot data points and fitted line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_true_line = w_true[0] + w_true[1] * x_range + w_true[2] * x_range**2
    y_est_line = w_est[0] + w_est[1] * x_range + w_est[2] * x_range**2
    
    # Scatter plot of actual data (using first feature for visualization)
    ax1.scatter(X[:, 0], y, alpha=0.6, color='blue', s=20, label='Data points')
    ax1.plot(x_range, y_true_line, 'g-', linewidth=2, label=f'True: y = {w_true[0]:.1f} + {w_true[1]:.1f}x + {w_true[2]:.1f}x²')
    ax1.plot(x_range, y_est_line, 'r--', linewidth=2, label=f'Estimate: y = {w_est[0]:.1f} + {w_est[1]:.1f}x + {w_est[2]:.1f}x²')
    
    ax1.set_xlabel("Feature $x_1$", fontsize=11)
    ax1.set_ylabel("Target $y$", fontsize=11)
    ax1.set_title("Linear Regression Model\n(Misspecified Quadratic)", fontsize=12, fontweight='bold', pad=15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add diagnostic text
    ax1.text(0.02, 0.98, f"True params: {w_true}\nEst. params: {w_est}\nMSE: {np.mean(residuals**2):.3f}", 
             transform=ax1.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 2: FIM heatmap with professional styling
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(F, cmap="viridis", aspect='equal', interpolation='nearest')
    ax2.set_title("Fisher Information Matrix $F_{ij}$\n(Information Content Heatmap)", 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel("Parameter index $j$", fontsize=11)
    ax2.set_ylabel("Parameter index $i$", fontsize=11)
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(['$w_0$', '$w_1$', '$w_2$'])
    ax2.set_yticklabels(['$w_0$', '$w_1$', '$w_2$'])
    
    # Add value annotations with better contrast
    for i in range(3):
        for j in range(3):
            color = "white" if F[i, j] > 0.5 else "black"
            text = ax2.text(j, i, f'{F[i, j]:.3f}', 
                           ha="center", va="center", color=color, fontweight='bold', fontsize=9)
    
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("$F_{ij}$ (information content)", fontsize=11)
    
    # Panel 3: 4D framework explanation with tetrahedral visualization
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Create tetrahedral visualization
    tetra_vertices = np.array([
        [0, 0, 0],      # Origin (Coxeter.4D)
        [1, 0, 0],      # Coxeter.4D (Euclidean)
        [0.5, 0.866, 0], # Einstein.4D (Minkowski)
        [0.5, 0.289, 0.816]  # Fuller.4D (Synergetics)
    ])
    
    # Define tetrahedron faces
    faces = [
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    ]
    
    # Plot tetrahedron edges
    for face in faces:
        for i in range(3):
            start = tetra_vertices[face[i]]
            end = tetra_vertices[face[(i+1)%3]]
            ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    'k-', linewidth=1, alpha=0.6)
    
    # Plot vertices with labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Origin', 'Coxeter.4D\n(Euclidean)', 'Einstein.4D\n(Minkowski)', 'Fuller.4D\n(Synergetics)']
    
    for i, (vertex, color, label) in enumerate(zip(tetra_vertices, colors, labels)):
        ax3.scatter(vertex[0], vertex[1], vertex[2], c=color, s=100, alpha=0.8)
        ax3.text(vertex[0], vertex[1], vertex[2], label, fontsize=9, ha='center')
    
    ax3.set_title("4D Framework Integration\n(Tetrahedral Structure)", fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel("X (Euclidean)", fontsize=10)
    ax3.set_ylabel("Y (Minkowski)", fontsize=10)
    ax3.set_zlabel("Z (Synergetics)", fontsize=10)
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    fim_path = os.path.join(figure_dir, "fisher_information_matrix.png")
    fig.savefig(fim_path, dpi=300, bbox_inches='tight')
    print(fim_path)
    plt.close(fig)

    # Save raw data alongside the figure for reproducibility and downstream use
    np.savetxt(os.path.join(data_dir, "fisher_information_matrix.csv"), F, delimiter=",")
    np.savez(
        os.path.join(data_dir, "fisher_information_matrix.npz"),
        F=F,
        grads=grads,
        w_true=w_true,
        w_est=w_est,
        X=X,
        y=y,
    )

    # Enhanced Figure 11: Comprehensive Fisher Information Eigenspectrum with Curvature Analysis
    # Now with 3 panels including tetrahedral parameter space visualization
    from metrics import fim_eigenspectrum  # noqa: WPS433
    evals, evecs = fim_eigenspectrum(F)
    
    # Comprehensive curvature analysis
    curvature_analysis = fisher_curvature_analysis(F)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Panel 1: Eigenspectrum with improved styling
    ax1 = fig.add_subplot(1, 3, 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(np.arange(evals.size), evals, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title("Fisher Information Eigenspectrum\n(Principal Curvature Directions)", 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Eigenvalue index", fontsize=11)
    ax1.set_ylabel("Eigenvalue magnitude $\\lambda_i$", fontsize=11)
    ax1.set_xticks(range(evals.size))
    ax1.set_xticklabels(['$\\lambda_0$', '$\\lambda_1$', '$\\lambda_2$'])
    
    # Add value annotations on bars with better positioning
    for i, (bar, val) in enumerate(zip(bars, evals)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(evals) * 1.15)  # Add some headroom for annotations
    
    # Panel 2: Comprehensive curvature analysis with better organization
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.axis('off')
    ax2.text(0.05, 0.95, "Curvature Analysis & 4D Framework", fontsize=14, fontweight='bold', 
             transform=ax2.transAxes)
    
    # Key metrics section
    ax2.text(0.05, 0.88, "Key Metrics:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    
    ax2.text(0.05, 0.82, "Condition Number:", fontsize=10, fontweight='bold',
             transform=ax2.transAxes, color='#1f77b4')
    ax2.text(0.6, 0.82, f"{curvature_analysis['condition_number']:.2f}", fontsize=10,
             transform=ax2.transAxes)
    
    ax2.text(0.05, 0.76, "Anisotropy Index:", fontsize=10, fontweight='bold',
             transform=ax2.transAxes, color='#ff7f0e')
    ax2.text(0.6, 0.76, f"{curvature_analysis['anisotropy_index']:.3f}", fontsize=10,
             transform=ax2.transAxes)
    
    ax2.text(0.05, 0.7, "Total Curvature:", fontsize=10, fontweight='bold',
             transform=ax2.transAxes, color='#2ca02c')
    ax2.text(0.6, 0.7, f"{curvature_analysis['trace']:.3f}", fontsize=10,
             transform=ax2.transAxes)
    
    # Eigenvalue interpretation
    ax2.text(0.05, 0.62, "Eigenvalue Interpretation:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.08, 0.56, "• $\\lambda_0 = {:.3f}$: Primary direction", fontsize=9,
             transform=ax2.transAxes, color='#1f77b4')
    ax2.text(0.08, 0.51, "• $\\lambda_1 = {:.3f}$: Secondary direction", fontsize=9,
             transform=ax2.transAxes, color='#ff7f0e')
    ax2.text(0.08, 0.46, "• $\\lambda_2 = {:.3f}$: Tertiary direction", fontsize=9,
             transform=ax2.transAxes, color='#2ca02c')
    
    # 4D framework connection
    ax2.text(0.05, 0.38, "4D Framework Connection:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.08, 0.32, "• Coxeter.4D: Euclidean parameter space", fontsize=9,
             transform=ax2.transAxes)
    ax2.text(0.08, 0.27, "• Einstein.4D: Fisher metric geometry", fontsize=9,
             transform=ax2.transAxes)
    ax2.text(0.08, 0.22, "• Fuller.4D: Tetrahedral structure", fontsize=9,
             transform=ax2.transAxes)
    
    # Optimization implications
    ax2.text(0.05, 0.14, "Optimization Implications:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.08, 0.08, "• Natural gradient scales by $F^{-1}$", fontsize=9,
             transform=ax2.transAxes)
    ax2.text(0.08, 0.03, "• Anisotropic scaling improves convergence", fontsize=9,
             transform=ax2.transAxes)
    
    # Panel 3: Tetrahedral parameter space visualization
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Create parameter space tetrahedron based on eigenvectors
    # Scale eigenvectors by eigenvalues to show curvature structure
    scaled_evecs = evecs * np.sqrt(evals[:, None])
    
    # Create tetrahedron vertices in parameter space
    param_vertices = np.array([
        [0, 0, 0],  # Origin
        scaled_evecs[0],  # Primary direction
        scaled_evecs[1],  # Secondary direction  
        scaled_evecs[2]   # Tertiary direction
    ])
    
    # Define tetrahedron faces
    faces = [
        [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]
    ]
    
    # Plot tetrahedron edges with curvature-based coloring
    for face in faces:
        for i in range(3):
            start = param_vertices[face[i]]
            end = param_vertices[face[(i+1)%3]]
            # Color based on curvature magnitude
            if face[i] == 0 or face[(i+1)%3] == 0:
                color = 'k'  # Black for edges to origin
            else:
                color = colors[face[i]-1]  # Color based on eigenvalue
            ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color=color, linewidth=2, alpha=0.8)
    
    # Plot vertices with labels
    vertex_labels = ['Origin', f'λ₀={evals[0]:.3f}', f'λ₁={evals[1]:.3f}', f'λ₂={evals[2]:.3f}']
    
    for i, (vertex, color, label) in enumerate(zip(param_vertices, ['black'] + colors, vertex_labels)):
        ax3.scatter(vertex[0], vertex[1], vertex[2], c=color, s=100, alpha=0.8)
        ax3.text(vertex[0], vertex[1], vertex[2], label, fontsize=9, ha='center')
    
    ax3.set_title("Parameter Space Tetrahedron\n(Curvature Structure)", fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel("$w_0$ parameter", fontsize=10)
    ax3.set_ylabel("$w_1$ parameter", fontsize=10)
    ax3.set_zlabel("$w_2$ parameter", fontsize=10)
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    eig_path = os.path.join(figure_dir, "fisher_information_eigenspectrum.png")
    fig.savefig(eig_path, dpi=300, bbox_inches='tight')
    print(eig_path)
    plt.close(fig)
    
    # Save eigen-data
    np.savetxt(os.path.join(data_dir, "fisher_information_eigenvalues.csv"), evals[None, :], delimiter=",")
    print(os.path.join(data_dir, "fisher_information_eigenvalues.csv"))
    np.savez(
        os.path.join(data_dir, "fisher_information_eigensystem.npz"),
        eigenvalues=evals,
        eigenvectors=evecs,
        F=F,
        curvature_analysis=curvature_analysis,
    )

    # Partition tetrahedron plot for appendix
    from visualize import plot_partition_tetrahedron  # noqa: WPS433
    mu = (2, 1, 1, 0)
    s = (1, 2, 1, 0)
    a = (1, 1, 2, 0)
    psi = (2, 2, 1, 1)
    part_path = plot_partition_tetrahedron(mu, s, a, psi)
    print(part_path)

    # Natural gradient step trajectory on a simple quadratic bowl
    A = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
    w = np.array([2.0, 2.0, 2.0])
    path = [w.copy()]
    for _ in range(20):
        g = A @ (w - w_true)
        step = natural_gradient_step(g, F + 1e-3 * np.eye(3), step_size=0.5)
        w = w + step
        path.append(w.copy())
    path = np.array(path)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(path[:, 0], path[:, 1], marker="o", linewidth=2, markersize=4)
    ax.set_xlabel("$w_0$", fontsize=11)
    ax.set_ylabel("$w_1$", fontsize=11)
    ax.set_title("Natural Gradient Trajectory\n(Geodesic Motion on Information Manifold)", 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add start and end markers
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    ax.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='End')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ng_path = os.path.join(figure_dir, "natural_gradient_path.png")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.14)
    fig.savefig(ng_path, dpi=300, bbox_inches='tight')
    print(ng_path)
    plt.close(fig)

    # Save raw trajectory data for reproducibility
    import csv  # noqa: WPS433
    ng_csv = os.path.join(data_dir, "natural_gradient_path.csv")
    with open(ng_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w0", "w1", "w2"])
        for row in path:
            writer.writerow([float(row[0]), float(row[1]), float(row[2])])
    print(ng_csv)
    np.savez(
        os.path.join(data_dir, "natural_gradient_path.npz"),
        path=path,
        A=A,
        F=F,
    )

    # Free energy surface for a 2-state toy model
    log_p = np.log(np.array([0.7, 0.3]))
    q_vals = np.linspace(1e-3, 0.999, 200)
    F_vals = [
        free_energy(log_p, np.array([q, 1 - q]), np.array([0.5, 0.5])) for q in q_vals
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(q_vals, F_vals, linewidth=2, color='#1f77b4')
    ax.set_xlabel("$q(\\text{state}=0)$", fontsize=11)
    ax.set_ylabel("Free Energy $\\mathcal{F}$", fontsize=11)
    ax.set_title("Variational Free Energy Landscape\n(2-State System)", 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Add minimum marker
    min_idx = np.argmin(F_vals)
    ax.plot(q_vals[min_idx], F_vals[min_idx], 'ro', markersize=8, label='Minimum')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fe_path = os.path.join(figure_dir, "free_energy_curve.png")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.18)
    fig.savefig(fe_path, dpi=300, bbox_inches='tight')
    print(fe_path)
    plt.close(fig)

    # Discrete lattice descent on a simple quadratic in embedded coordinates
    def f2(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        return float((x - 0.5) ** 2 + (y + 0.2) ** 2 + (z - 0.1) ** 2)

    dpath = discrete_ivm_descent(f2, Quadray(6, 0, 0, 0))
    dpath_mp4 = animate_discrete_path(dpath, save=True)
    print(dpath_mp4)


if __name__ == "__main__":
    main()


