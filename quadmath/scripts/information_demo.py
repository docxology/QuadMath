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

    # Enhanced FIM visualization with 4D context
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel 1: FIM heatmap with professional styling
    im1 = ax1.imshow(F, cmap="viridis", aspect='equal', interpolation='nearest')
    ax1.set_title("Fisher Information Matrix (Coxeter.4D → Einstein.4D)", 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Parameter index $i$", fontsize=11)
    ax1.set_ylabel("Parameter index $j$", fontsize=11)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(['$w_0$', '$w_1$', '$w_2$'])
    ax1.set_yticklabels(['$w_0$', '$w_1$', '$w_2$'])
    
    # Add value annotations
    for i in range(3):
        for j in range(3):
            text = ax1.text(j, i, f'{F[i, j]:.2f}', 
                           ha="center", va="center", color="white", fontweight='bold')
    
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("$F_{ij}$ (information content)", fontsize=11)
    
    # Panel 2: 4D framework explanation
    ax2.axis('off')
    ax2.text(0.1, 0.9, "4D Framework Context", fontsize=14, fontweight='bold', 
             transform=ax2.transAxes)
    
    # Coxeter.4D explanation
    ax2.text(0.1, 0.8, "• Coxeter.4D (Euclidean):", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, color='#1f77b4')
    ax2.text(0.15, 0.75, "  Standard 3D parameter space", fontsize=10,
             transform=ax2.transAxes)
    ax2.text(0.15, 0.7, "  with Euclidean metric", fontsize=10,
             transform=ax2.transAxes)
    
    # Einstein.4D explanation  
    ax2.text(0.1, 0.6, "• Einstein.4D (Minkowski):", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, color='#ff7f0e')
    ax2.text(0.15, 0.55, "  Fisher metric replaces spacetime", fontsize=10,
             transform=ax2.transAxes)
    ax2.text(0.15, 0.5, "  metric; geodesics follow F⁻¹∇L", fontsize=10,
             transform=ax2.transAxes)
    
    # Fuller.4D explanation
    ax2.text(0.1, 0.4, "• Fuller.4D (Synergetics):", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, color='#2ca02c')
    ax2.text(0.15, 0.35, "  Tetrahedral coordinate system", fontsize=10,
             transform=ax2.transAxes)
    ax2.text(0.15, 0.3, "  with IVM quantization", fontsize=10,
             transform=ax2.transAxes)
    
    # Mathematical details
    ax2.text(0.1, 0.2, "Mathematical Structure:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.15, 0.15, "$F_{ij} = \\frac{1}{N}\\sum_n \\frac{\\partial L}{\\partial w_i} \\frac{\\partial L}{\\partial w_j}$", 
             fontsize=10, transform=ax2.transAxes, style='italic')
    
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

    # Enhanced eigenspectrum visualization
    from metrics import fim_eigenspectrum  # noqa: WPS433
    evals, evecs = fim_eigenspectrum(F)
    
    # Comprehensive curvature analysis
    curvature_analysis = fisher_curvature_analysis(F)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel 1: Eigenspectrum with enhanced styling
    bars = ax1.bar(np.arange(evals.size), evals, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax1.set_title("Fisher Information Eigenspectrum\n(Principal Curvature Directions)", 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel("Eigenvalue index", fontsize=11)
    ax1.set_ylabel("Eigenvalue magnitude $\\lambda_i$", fontsize=11)
    ax1.set_xticks(range(evals.size))
    ax1.set_xticklabels(['$\\lambda_0$', '$\\lambda_1$', '$\\lambda_2$'])
    
    # Add value annotations on bars
    for i, (bar, val) in enumerate(zip(bars, evals)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Curvature analysis summary
    ax2.axis('off')
    ax2.text(0.1, 0.9, "Curvature Analysis Summary", fontsize=14, fontweight='bold', 
             transform=ax2.transAxes)
    
    # Key metrics
    ax2.text(0.1, 0.8, "Condition Number:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, color='#1f77b4')
    ax2.text(0.6, 0.8, f"{curvature_analysis['condition_number']:.2f}", fontsize=11,
             transform=ax2.transAxes)
    
    ax2.text(0.1, 0.7, "Anisotropy Index:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, color='#ff7f0e')
    ax2.text(0.6, 0.7, f"{curvature_analysis['anisotropy_index']:.3f}", fontsize=11,
             transform=ax2.transAxes)
    
    ax2.text(0.1, 0.6, "Total Curvature:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, color='#2ca02c')
    ax2.text(0.6, 0.6, f"{curvature_analysis['trace']:.2f}", fontsize=11,
             transform=ax2.transAxes)
    
    # Interpretation
    ax2.text(0.1, 0.5, "Interpretation:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.15, 0.4, "• High $\\lambda$: Tight constraints", fontsize=10,
             transform=ax2.transAxes)
    ax2.text(0.15, 0.35, "• Low $\\lambda$: Loose constraints", fontsize=10,
             transform=ax2.transAxes)
    ax2.text(0.15, 0.3, "• Natural gradient scales by $F^{-1}$", fontsize=10,
             transform=ax2.transAxes)
    
    # 4D connection
    ax2.text(0.1, 0.2, "4D Connection:", fontsize=11, fontweight='bold',
             transform=ax2.transAxes)
    ax2.text(0.15, 0.15, "Eigenvalues reveal anisotropic", fontsize=10,
             transform=ax2.transAxes)
    ax2.text(0.15, 0.1, "parameter space structure", fontsize=10,
             transform=ax2.transAxes)
    
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


