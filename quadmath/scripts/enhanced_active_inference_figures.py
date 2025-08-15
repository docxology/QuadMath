#!/usr/bin/env python3
"""Generate enhanced Figures 13 and 14 for Active Inference and Free Energy Principle.

This script creates comprehensive visualizations that demonstrate the integration of:
- Natural gradient descent with Active Inference principles
- Free energy landscapes in 4D frameworks
- Information-geometric optimization in biological contexts

Figures 13 and 14 are distinct from Figures 11 and 12, focusing on:
- 4D trajectory evolution over time
- Active Inference four-fold partition dynamics
- Free energy principle in neural dynamics
- Information-geometric flows in biological systems

Saves to quadmath/output/ and prints saved paths.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def create_4d_trajectory_visualization():
    """Create Figure 13: 4D Natural Gradient Trajectory with Active Inference Context."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()
    
    from paths import get_output_dir, get_data_dir, get_figure_dir
    from information import fisher_information_matrix, natural_gradient_step
    
    # Set style for professional appearance
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Create synthetic data for demonstration
    rng = np.random.default_rng(42)
    
    # Create a more complex objective function with multiple minima
    # This represents the free energy landscape in Active Inference
    def create_active_inference_objective():
        """Create an objective function that mimics Active Inference dynamics."""
        # Parameters: [perception_weight, action_weight, internal_state, external_state]
        # This represents the four-fold partition of Active Inference
        
        # Create a synthetic dataset that mimics sensory observations
        X = rng.normal(size=(150, 4))
        noise = 0.15 * rng.normal(size=X.shape[0])
        
        # True parameters representing optimal Active Inference configuration
        w_true = np.array([1.2, -0.8, 0.6, -0.4])
        y = X @ w_true + noise
        
        # Evaluate at a suboptimal estimate
        w_est = np.array([0.5, -0.3, 0.1, -0.1])
        residuals = X @ w_est - y
        
        # Gradients of squared loss with respect to parameters
        grads = 2.0 * (X.T * residuals).T
        F = fisher_information_matrix(grads)
        
        return F, w_est, w_true, X, y
    
    F, w_est, w_true, X, y = create_active_inference_objective()
    
    # Natural gradient descent with Active Inference interpretation
    w = w_est.copy()
    path = [w.copy()]
    free_energy_trace = []
    
    # Track the evolution of the four Active Inference components
    perception_trace = []
    action_trace = []
    internal_trace = []
    external_trace = []
    
    for step in range(25):
        # Compute gradient (prediction error in Active Inference)
        g = X.T @ (X @ w - y)
        
        # Natural gradient step (geodesic motion on information manifold)
        step_update = natural_gradient_step(g, F + 1e-3 * np.eye(4), step_size=0.3)
        w = w + step_update
        
        # Store trajectory
        path.append(w.copy())
        
        # Compute free energy proxy (squared loss)
        free_energy_val = float(np.mean((X @ w - y) ** 2))
        free_energy_trace.append(free_energy_val)
        
        # Store individual component evolution
        perception_trace.append(w[0])
        action_trace.append(w[1])
        internal_trace.append(w[2])
        external_trace.append(w[3])
    
    path = np.array(path)
    
    # Create comprehensive 4D visualization
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: 3D trajectory with time as color
    ax1 = fig.add_subplot(2, 3, (1, 2), projection='3d')
    
    # Create time-based color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, len(path)))
    
    for i in range(len(path) - 1):
        ax1.plot([path[i, 0], path[i+1, 0]], 
                 [path[i, 1], path[i+1, 1]], 
                 [path[i, 2], path[i+1, 2]], 
                 color=colors[i], linewidth=2, alpha=0.8)
    
    # Mark start and end points
    ax1.scatter([path[0, 0]], [path[0, 1]], [path[0, 2]], 
                c='green', s=100, marker='o', label='Initial State')
    ax1.scatter([path[-1, 0]], [path[-1, 1]], [path[-1, 2]], 
                c='red', s=100, marker='*', label='Converged State')
    
    ax1.set_xlabel('Perception Weight', fontsize=11)
    ax1.set_ylabel('Action Weight', fontsize=11)
    ax1.set_zlabel('Internal State', fontsize=11)
    ax1.set_title('Figure 13: 4D Natural Gradient Trajectory\n(Active Inference Dynamics)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.legend()
    
    # Panel 2: Free energy evolution over time
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.plot(range(len(free_energy_trace)), free_energy_trace, 
             linewidth=2, color='#d62728', marker='o', markersize=4)
    ax2.set_xlabel('Optimization Step', fontsize=11)
    ax2.set_ylabel('Free Energy (Squared Loss)', fontsize=11)
    ax2.set_title('Free Energy Minimization\n(Active Inference Principle)', 
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Component evolution over time
    ax3 = fig.add_subplot(2, 3, 4)
    steps = range(len(perception_trace))
    ax3.plot(steps, perception_trace, 'b-', linewidth=2, label='Perception (μ)', marker='o', markersize=3)
    ax3.plot(steps, action_trace, 'r-', linewidth=2, label='Action (a)', marker='s', markersize=3)
    ax3.plot(steps, internal_trace, 'g-', linewidth=2, label='Internal (s)', marker='^', markersize=3)
    ax3.plot(steps, external_trace, 'm-', linewidth=2, label='External (ψ)', marker='d', markersize=3)
    ax3.set_xlabel('Optimization Step', fontsize=11)
    ax3.set_ylabel('Parameter Value', fontsize=11)
    ax3.set_title('Four-Fold Partition Evolution\n(Fuller.4D Tetrahedral Structure)', 
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Information geometry context
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.axis('off')
    
    # Create a comprehensive explanation box
    explanation_text = """4D Framework Integration:
    
• Coxeter.4D (Euclidean): 3D parameter space with 
  Euclidean metric for exact measurements
    
• Einstein.4D (Minkowski): Fisher metric replaces 
  spacetime metric; geodesics follow F⁻¹∇L
    
• Fuller.4D (Synergetics): Tetrahedral coordinate 
  system with IVM quantization
    
Active Inference Context:
• Natural gradient descent minimizes free energy
• Four-fold partition: μ, s, a, ψ
• Information-geometric flows drive perception-action
• Biological plausibility through geodesic motion"""
    
    ax4.text(0.05, 0.95, explanation_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Panel 5: Fisher Information Matrix heatmap
    ax5 = fig.add_subplot(2, 3, 6)
    im = ax5.imshow(F, cmap="viridis", aspect='equal', interpolation='nearest')
    ax5.set_title('Fisher Information Matrix\n(Information Geometry Metric)', 
                  fontsize=11, fontweight='bold')
    ax5.set_xlabel('Parameter Index', fontsize=11)
    ax5.set_ylabel('Parameter Index', fontsize=11)
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    ax5.set_xticklabels(['μ', 's', 'a', 'ψ'])
    ax5.set_yticklabels(['μ', 's', 'a', 'ψ'])
    
    # Add value annotations
    for i in range(4):
        for j in range(4):
            text = ax5.text(j, i, f'{F[i, j]:.2f}', 
                           ha="center", va="center", color="white", fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label("$F_{ij}$ (Information Content)", fontsize=11)
    
    plt.tight_layout()
    
    # Save the enhanced figure
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    
    fig13_path = os.path.join(figure_dir, "enhanced_figure_13_4d_trajectory.png")
    fig.savefig(fig13_path, dpi=300, bbox_inches='tight')
    print(f"Figure 13 saved: {fig13_path}")
    plt.close(fig)
    
    # Save raw data
    np.savez(
        os.path.join(data_dir, "enhanced_figure_13_data.npz"),
        path=path,
        free_energy_trace=free_energy_trace,
        perception_trace=perception_trace,
        action_trace=action_trace,
        internal_trace=internal_trace,
        external_trace=external_trace,
        F=F,
        w_true=w_true,
        w_est=w_est
    )
    
    return fig13_path


def create_enhanced_free_energy_landscape():
    """Create Figure 14: Enhanced Free Energy Landscape with 4D Context."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()
    
    from paths import get_output_dir, get_data_dir, get_figure_dir
    from information import free_energy
    
    # Set style for professional appearance
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Create a more sophisticated free energy landscape
    # This represents the variational free energy in Active Inference
    
    # Create a 2D free energy landscape over two variational parameters
    # representing the balance between perception and action
    q1_vals = np.linspace(0.01, 0.99, 100)
    q2_vals = np.linspace(0.01, 0.99, 100)
    Q1, Q2 = np.meshgrid(q1_vals, q2_vals)
    
    # Create synthetic log-likelihoods that mimic Active Inference dynamics
    # We need 4-element arrays to match the 4-state distributions
    log_p_o_given_s = np.log(np.array([0.6, 0.4, 0.3, 0.5]))  # Sensory likelihood for 4 states
    
    # Compute free energy over the 2D parameter space
    F_landscape = np.zeros_like(Q1)
    for i in range(len(q1_vals)):
        for j in range(len(q2_vals)):
            # Create variational distribution over 4 states
            # representing the four-fold partition
            q = np.array([Q1[i, j], Q2[i, j], 
                         1 - Q1[i, j] - Q2[i, j], 0.0])
            q = np.maximum(q, 1e-10)  # Ensure positivity
            q = q / np.sum(q)  # Normalize
            
            # Prior distribution (uniform for simplicity)
            p = np.array([0.25, 0.25, 0.25, 0.25])
            
            # Compute free energy
            F_landscape[i, j] = free_energy(log_p_o_given_s, q, p)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: 3D free energy landscape
    ax1 = fig.add_subplot(2, 3, (1, 2), projection='3d')
    
    surf = ax1.plot_surface(Q1, Q2, F_landscape, cmap='viridis', 
                           alpha=0.8, linewidth=0, antialiased=True)
    
    # Find the minimum
    min_idx = np.unravel_index(np.argmin(F_landscape), F_landscape.shape)
    min_q1, min_q2 = q1_vals[min_idx[1]], q2_vals[min_idx[0]]
    min_F = F_landscape[min_idx]
    
    # Mark the minimum
    ax1.scatter([min_q1], [min_q2], [min_F], 
                c='red', s=100, marker='*', label='Global Minimum')
    
    ax1.set_xlabel('Perception Parameter (q₁)', fontsize=11)
    ax1.set_ylabel('Action Parameter (q₂)', fontsize=11)
    ax1.set_zlabel('Free Energy $\\mathcal{F}$', fontsize=11)
    ax1.set_title('Figure 14: Enhanced Free Energy Landscape\n(4D Active Inference Framework)', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.legend()
    
    # Add colorbar
    cbar1 = fig.colorbar(surf, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Free Energy Value", fontsize=11)
    
    # Panel 2: 2D contour plot
    ax2 = fig.add_subplot(2, 3, 3)
    contour = ax2.contour(Q1, Q2, F_landscape, levels=20, colors='black', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Mark the minimum
    ax2.scatter(min_q1, min_q2, c='red', s=100, marker='*', label='Global Minimum')
    
    ax2.set_xlabel('Perception Parameter (q₁)', fontsize=11)
    ax2.set_ylabel('Action Parameter (q₂)', fontsize=11)
    ax2.set_title('Free Energy Contours\n(Information Geometry)', 
                  fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Free energy cross-sections
    ax3 = fig.add_subplot(2, 3, 4)
    
    # Cross-section at optimal q2
    cross_section_q2 = F_landscape[min_idx[0], :]
    ax3.plot(q1_vals, cross_section_q2, 'b-', linewidth=2, 
             label=f'q₂ = {min_q2:.2f} (optimal)')
    
    # Cross-section at suboptimal q2
    subopt_idx = len(q2_vals) // 3
    cross_section_subopt = F_landscape[subopt_idx, :]
    ax3.plot(q1_vals, cross_section_subopt, 'r--', linewidth=2, 
             label=f'q₂ = {q2_vals[subopt_idx]:.2f}')
    
    ax3.set_xlabel('Perception Parameter (q₁)', fontsize=11)
    ax3.set_ylabel('Free Energy $\\mathcal{F}$', fontsize=11)
    ax3.set_title('Free Energy Cross-Sections\n(Parameter Sensitivity)', 
                  fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Active Inference four-fold partition visualization
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.axis('off')
    
    # Create a tetrahedral representation of the four-fold partition
    # This connects to Fuller.4D concepts
    
    partition_text = """Active Inference Four-Fold Partition:
    
    μ (Internal States)
         ↙     ↘
    s (Sensory)  a (Actions)
         ↘     ↙
    ψ (External Causes)
    
4D Framework Integration:
• Fuller.4D: Tetrahedral structure
• Coxeter.4D: Exact Euclidean geometry  
• Einstein.4D: Information-geometric flows
    
Free Energy Principle:
• Minimization drives perception-action
• Balances prediction error vs. complexity
• Natural gradient follows geodesics"""
    
    ax4.text(0.05, 0.95, partition_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Panel 5: Information geometry metrics
    ax5 = fig.add_subplot(2, 3, 6)
    
    # Compute local curvature information
    # This represents the Fisher information structure
    # Use numpy's gradient function for numerical derivatives
    
    # Compute gradients directly from the landscape
    grad_q1 = np.gradient(F_landscape, axis=1)
    grad_q2 = np.gradient(F_landscape, axis=0)
    
    # Compute local curvature (second derivatives)
    grad_q1_q1 = np.gradient(grad_q1, axis=1)
    grad_q2_q2 = np.gradient(grad_q2, axis=0)
    
    # Local curvature at minimum
    curvature_at_min = grad_q1_q1[min_idx] + grad_q2_q2[min_idx]
    
    # Plot curvature information
    curvature_plot = ax5.imshow(grad_q1_q1 + grad_q2_q2, cmap='RdBu_r', 
                               extent=[q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]])
    
    # Mark the minimum
    ax5.scatter(min_q1, min_q2, c='red', s=100, marker='*', label='Global Minimum')
    
    ax5.set_xlabel('Perception Parameter (q₁)', fontsize=11)
    ax5.set_ylabel('Action Parameter (q₂)', fontsize=11)
    ax5.set_title('Local Curvature\n(Fisher Information Structure)', 
                  fontsize=11, fontweight='bold')
    ax5.legend()
    
    cbar2 = fig.colorbar(curvature_plot, ax=ax5, fraction=0.046, pad=0.04)
    cbar2.set_label("Local Curvature", fontsize=11)
    
    plt.tight_layout()
    
    # Save the enhanced figure
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    
    fig14_path = os.path.join(figure_dir, "enhanced_figure_14_free_energy_landscape.png")
    fig.savefig(fig14_path, dpi=300, bbox_inches='tight')
    print(f"Figure 14 saved: {fig14_path}")
    plt.close(fig)
    
    # Save raw data
    np.savez(
        os.path.join(data_dir, "enhanced_figure_14_data.npz"),
        Q1=Q1,
        Q2=Q2,
        F_landscape=F_landscape,
        min_idx=min_idx,
        min_q1=min_q1,
        min_q2=min_q2,
        min_F=min_F,
        curvature_at_min=curvature_at_min
    )
    
    return fig14_path


def main() -> None:
    """Generate both enhanced figures."""
    print("Generating enhanced Figures 13 and 14 for Active Inference...")
    
    # Generate Figure 13: 4D Natural Gradient Trajectory
    fig13_path = create_4d_trajectory_visualization()
    
    # Generate Figure 14: Enhanced Free Energy Landscape
    fig14_path = create_enhanced_free_energy_landscape()
    
    print("\nEnhanced figures generated successfully!")
    print(f"Figure 13: {fig13_path}")
    print(f"Figure 14: {fig14_path}")
    print("\nThese figures demonstrate:")
    print("• 4D trajectory evolution in Active Inference context")
    print("• Free energy principle in biological systems")
    print("• Information-geometric optimization")
    print("• Integration of Coxeter.4D, Einstein.4D, and Fuller.4D frameworks")


if __name__ == "__main__":
    main()
