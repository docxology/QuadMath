#!/usr/bin/env python3
"""Generate Active Inference figures demonstrating 4D framework integration.

This script creates comprehensive visualizations that demonstrate the integration of:
- Natural gradient descent with Active Inference principles
- Free energy landscapes in 4D frameworks
- Information-geometric optimization in biological contexts
- Coxeter.4D (Euclidean), Einstein.4D (Minkowski), and Fuller.4D (Synergetics) concepts

The figures focus on:
- 4D trajectory evolution over time with Active Inference dynamics
- Free energy principle in biological systems with information geometry
- Four-fold partition dynamics mapped to tetrahedral structures
- Natural gradient flows following geodesics on information manifolds

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
    """Create 4D Natural Gradient Trajectory with Active Inference Context."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()
    
    from paths import get_output_dir, get_data_dir, get_figure_dir
    from information import fisher_information_matrix, natural_gradient_step
    
    # Set style for professional appearance
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.dpi'] = 300
    
    # Create synthetic data for demonstration with more realistic Active Inference dynamics
    rng = np.random.default_rng(42)
    
    def create_active_inference_objective():
        """Create an objective function that mimics Active Inference dynamics."""
        # Parameters: [perception_weight, action_weight, internal_state, external_state]
        # This represents the four-fold partition of Active Inference
        
        # Create a synthetic dataset that mimics sensory observations
        # Use smaller scale for more stable optimization
        X = 0.1 * rng.normal(size=(100, 4))  # Reduced scale
        noise = 0.01 * rng.normal(size=X.shape[0])  # Reduced noise
        
        # True parameters representing optimal Active Inference configuration
        w_true = np.array([0.8, -0.6, 0.4, -0.3])  # Smaller scale
        y = X @ w_true + noise
        
        # Evaluate at a suboptimal estimate
        w_est = np.array([0.2, -0.1, 0.05, -0.05])  # Closer to true values
        
        return w_est, w_true, X, y
    
    w_est, w_true, X, y = create_active_inference_objective()
    
    # Natural gradient descent with Active Inference interpretation
    w = w_est.copy()
    path = [w.copy()]
    free_energy_trace = []
    
    # Track the evolution of the four Active Inference components
    perception_trace = []
    action_trace = []
    internal_trace = []
    external_trace = []
    
    # Track step sizes and convergence metrics
    step_sizes = []
    gradient_norms = []
    parameter_changes = []
    
    # Optimization loop with adaptive step sizes and better stability
    max_steps = 50
    min_step_size = 1e-6
    max_step_size = 0.1
    
    for step in range(max_steps):
        # Compute gradient (prediction error in Active Inference)
        residuals = X @ w - y
        g = X.T @ residuals
        
        # Compute current free energy (squared loss)
        free_energy_val = float(np.mean(residuals ** 2))
        free_energy_trace.append(free_energy_val)
        
        # Store individual component evolution
        perception_trace.append(w[0])
        action_trace.append(w[1])
        internal_trace.append(w[2])
        external_trace.append(w[3])
        
        # Compute gradient norm for diagnostics
        grad_norm = np.linalg.norm(g)
        gradient_norms.append(grad_norm)
        
        # Early stopping if gradient is very small
        if grad_norm < 1e-6:
            print(f"Converged at step {step} with gradient norm {grad_norm:.2e}")
            break
        
        # Compute Fisher Information Matrix for this step
        # Use gradients of log-likelihood (residuals)
        grads = (X.T * residuals).T
        F = fisher_information_matrix(grads)
        
        # Ensure positive definiteness with larger ridge
        ridge = 1e-3
        
        # Adaptive step size based on gradient magnitude and curvature
        if step == 0:
            # Initial step size based on gradient magnitude
            step_size = min(0.01, max_step_size)
        else:
            # Adaptive step size based on previous step success
            if len(parameter_changes) > 0:
                prev_change = parameter_changes[-1]
                if prev_change < 1e-4:  # Very small change
                    step_size = min(step_size * 1.1, max_step_size)
                elif prev_change > 0.1:  # Large change
                    step_size = max(step_size * 0.9, min_step_size)
        
        # Natural gradient step (geodesic motion on information manifold)
        try:
            step_update = natural_gradient_step(g, F + ridge * np.eye(4), step_size=step_size)
            
            # Check for numerical instability
            if np.any(np.isnan(step_update)) or np.any(np.isinf(step_update)):
                print(f"Numerical instability detected at step {step}, reducing step size")
                step_size *= 0.5
                step_update = natural_gradient_step(g, F + ridge * np.eye(4), step_size=step_size)
            
            # Apply update
            w_new = w + step_update
            
            # Check if update is reasonable
            param_change = np.linalg.norm(step_update)
            parameter_changes.append(param_change)
            
            if param_change > 1.0:  # Unreasonably large step
                print(f"Large step detected at step {step}, reducing step size")
                step_size *= 0.5
                step_update = natural_gradient_step(g, F + ridge * np.eye(4), step_size=step_size)
                w_new = w + step_update
                param_change = np.linalg.norm(step_update)
                parameter_changes[-1] = param_change
            
            w = w_new
            step_sizes.append(step_size)
            
        except np.linalg.LinAlgError:
            print(f"Linear algebra error at step {step}, using gradient descent fallback")
            # Fallback to standard gradient descent
            w = w - 0.001 * g
            step_sizes.append(0.001)
            parameter_changes.append(0.001 * grad_norm)
        
        # Store trajectory
        path.append(w.copy())
        
        # Check for convergence
        if len(free_energy_trace) > 1:
            energy_change = abs(free_energy_trace[-1] - free_energy_trace[-2])
            if energy_change < 1e-8:
                print(f"Energy converged at step {step}")
                break
    
    path = np.array(path)
    
    print(f"Optimization completed in {len(path)-1} steps")
    print(f"Final free energy: {free_energy_trace[-1]:.2e}")
    print(f"Parameter change from initial: {np.linalg.norm(path[-1] - path[0]):.2e}")
    
    # Create comprehensive 4D visualization with improved layout
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: 3D trajectory with time as color and improved aesthetics
    ax1 = fig.add_subplot(2, 3, (1, 2), projection='3d')
    
    # Create time-based color mapping with better contrast
    colors = plt.cm.plasma(np.linspace(0, 1, len(path)))
    
    # Plot trajectory with improved styling
    for i in range(len(path) - 1):
        ax1.plot([path[i, 0], path[i+1, 0]], 
                 [path[i, 1], path[i+1, 1]], 
                 [path[i, 2], path[i+1, 2]], 
                 color=colors[i], linewidth=2.5, alpha=0.9)
    
    # Mark start and end points with better visibility
    ax1.scatter([path[0, 0]], [path[0, 1]], [path[0, 2]], 
                c='green', s=150, marker='o', label='Initial State', edgecolors='black', linewidth=1)
    ax1.scatter([path[-1, 0]], [path[-1, 1]], [path[-1, 2]], 
                c='red', s=150, marker='*', label='Converged State', edgecolors='black', linewidth=1)
    
    # Mark true optimal point
    ax1.scatter([w_true[0]], [w_true[1]], [w_true[2]], 
                c='blue', s=120, marker='^', label='True Optimal', edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('Perception Weight (μ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Action Weight (a)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Internal State (s)', fontsize=12, fontweight='bold')
    ax1.set_title('4D Natural Gradient Trajectory\n(Active Inference Dynamics)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Free energy evolution over time with improved styling
    ax2 = fig.add_subplot(2, 3, 3)
    steps = range(len(free_energy_trace))
    ax2.plot(steps, free_energy_trace, 
             linewidth=3, color='#d62728', marker='o', markersize=5, markevery=3)
    ax2.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Free Energy (Squared Loss)', fontsize=12, fontweight='bold')
    ax2.set_title('Free Energy Minimization\n(Active Inference Principle)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.4)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    # Panel 3: Component evolution over time with styling
    ax3 = fig.add_subplot(2, 3, 4)
    steps = range(len(perception_trace))
    ax3.plot(steps, perception_trace, 'b-', linewidth=2.5, label='Perception (μ)', marker='o', markersize=4, markevery=4)
    ax3.plot(steps, action_trace, 'r-', linewidth=2.5, label='Action (a)', marker='s', markersize=4, markevery=4)
    ax3.plot(steps, internal_trace, 'g-', linewidth=2.5, label='Internal (s)', marker='^', markersize=4, markevery=4)
    ax3.plot(steps, external_trace, 'm-', linewidth=2.5, label='External (ψ)', marker='d', markersize=4, markevery=4)
    
    # Add horizontal lines for true values
    ax3.axhline(y=w_true[0], color='b', linestyle='--', alpha=0.7, label='μ* (true)')
    ax3.axhline(y=w_true[1], color='r', linestyle='--', alpha=0.7, label='a* (true)')
    ax3.axhline(y=w_true[2], color='g', linestyle='--', alpha=0.7, label='s* (true)')
    ax3.axhline(y=w_true[3], color='m', linestyle='--', alpha=0.7, label='ψ* (true)')
    
    ax3.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Parameter Value', fontsize=12, fontweight='bold')
    ax3.set_title('Four-Fold Partition Evolution\n(Fuller.4D Tetrahedral Structure)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Optimization diagnostics with step sizes and gradients
    ax4 = fig.add_subplot(2, 3, 5)
    
    # Plot step sizes and gradient norms
    if len(step_sizes) > 0:
        ax4_twin = ax4.twinx()
        
        # Step sizes
        ax4.plot(range(len(step_sizes)), step_sizes, 'b-', linewidth=2, label='Step Size', marker='o', markersize=3)
        ax4.set_ylabel('Step Size', color='b', fontsize=11, fontweight='bold')
        ax4.tick_params(axis='y', labelcolor='b')
        
        # Gradient norms
        ax4_twin.plot(range(len(gradient_norms)), gradient_norms, 'r-', linewidth=2, label='Gradient Norm', marker='s', markersize=3)
        ax4_twin.set_ylabel('Gradient Norm', color='r', fontsize=11, fontweight='bold')
        ax4_twin.tick_params(axis='y', labelcolor='r')
        ax4_twin.set_yscale('log')
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    
    ax4.set_xlabel('Optimization Step', fontsize=12, fontweight='bold')
    ax4.set_title('Optimization Diagnostics\n(Step Size & Gradient Evolution)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Fisher Information Matrix heatmap with visualization
    ax5 = fig.add_subplot(2, 3, 6)
    
    # Use the final Fisher Information Matrix
    F_final = fisher_information_matrix((X.T * (X @ w - y)).T)
    
    # Use a better colormap for the heatmap
    im = ax5.imshow(F_final, cmap="viridis", aspect='equal', interpolation='nearest')
    ax5.set_title('Fisher Information Matrix\n(Information Geometry Metric)', 
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel('Parameter Index', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Parameter Index', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    ax5.set_xticklabels(['μ', 's', 'a', 'ψ'])
    ax5.set_yticklabels(['μ', 's', 'a', 'ψ'])
    
    # Add value annotations with better formatting
    for i in range(4):
        for j in range(4):
            text = ax5.text(j, i, f'{F_final[i, j]:.2f}', 
                           ha="center", va="center", color="white", fontweight='bold', fontsize=9)
    
    cbar = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    cbar.set_label("$F_{ij}$ (Information Content)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    
    fig13_path = os.path.join(figure_dir, "figure_13_4d_trajectory.png")
    fig.savefig(fig13_path, dpi=300, bbox_inches='tight')
    print(f"4D trajectory figure saved: {fig13_path}")
    plt.close(fig)
    
    # Save raw data with metadata
    np.savez(
        os.path.join(data_dir, "figure_13_data.npz"),
        path=path,
        free_energy_trace=free_energy_trace,
        perception_trace=perception_trace,
        action_trace=action_trace,
        internal_trace=internal_trace,
        external_trace=external_trace,
        step_sizes=step_sizes,
        gradient_norms=gradient_norms,
        parameter_changes=parameter_changes,
        F_final=F_final,
        w_true=w_true,
        w_est=w_est,
        metadata={
            'description': '4D Natural Gradient Trajectory for Active Inference',
            'parameters': ['perception_weight', 'action_weight', 'internal_state', 'external_state'],
            'framework': ['Coxeter.4D', 'Einstein.4D', 'Fuller.4D'],
            'optimization_steps': len(free_energy_trace),
            'convergence_achieved': len(free_energy_trace) < max_steps
        }
    )
    
    return fig13_path


def create_free_energy_landscape():
    """Create Free Energy Landscape with 4D Context."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()
    
    from paths import get_output_dir, get_data_dir, get_figure_dir
    from information import free_energy
    
    # Set style for professional appearance
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.dpi'] = 300
    
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
    
    # Find the minimum
    min_idx = np.unravel_index(np.argmin(F_landscape), F_landscape.shape)
    min_q1, min_q2 = q1_vals[min_idx[1]], q2_vals[min_idx[0]]
    min_F = F_landscape[min_idx]
    
    # Create comprehensive visualization with improved layout
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: 3D free energy landscape with styling
    ax1 = fig.add_subplot(2, 3, (1, 2), projection='3d')
    
    # Use a better colormap for the surface
    surf = ax1.plot_surface(Q1, Q2, F_landscape, cmap='viridis', 
                           alpha=0.85, linewidth=0.5, antialiased=True)
    
    # Mark the minimum with better visibility
    ax1.scatter([min_q1], [min_q2], [min_F], 
                c='red', s=150, marker='*', label='Global Minimum', edgecolors='black', linewidth=1)
    
    ax1.set_xlabel('Perception Parameter (q₁)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Action Parameter (q₂)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Free Energy $\\mathcal{F}$', fontsize=12, fontweight='bold')
    ax1.set_title('Free Energy Landscape\n(4D Active Inference Framework)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar1 = fig.colorbar(surf, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Free Energy Value", fontsize=11, fontweight='bold')
    
    # Panel 2: 2D contour plot with styling
    ax2 = fig.add_subplot(2, 3, 3)
    contour = ax2.contour(Q1, Q2, F_landscape, levels=25, colors='black', alpha=0.7, linewidths=1.2)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Mark the minimum
    ax2.scatter(min_q1, min_q2, c='red', s=150, marker='*', label='Global Minimum', edgecolors='black', linewidth=1)
    
    ax2.set_xlabel('Perception Parameter (q₁)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Action Parameter (q₂)', fontsize=12, fontweight='bold')
    ax2.set_title('Free Energy Contours\n(Information Geometry)', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    
    # Panel 3: Free energy cross-sections with styling
    ax3 = fig.add_subplot(2, 3, 4)
    
    # Cross-section at optimal q2
    cross_section_q2 = F_landscape[min_idx[0], :]
    ax3.plot(q1_vals, cross_section_q2, 'b-', linewidth=3, 
             label=f'q₂ = {min_q2:.2f} (optimal)', marker='o', markersize=4, markevery=10)
    
    # Cross-section at suboptimal q2
    subopt_idx = len(q2_vals) // 3
    cross_section_subopt = F_landscape[subopt_idx, :]
    ax3.plot(q1_vals, cross_section_subopt, 'r--', linewidth=2.5, 
             label=f'q₂ = {q2_vals[subopt_idx]:.2f}', marker='s', markersize=4, markevery=10)
    
    # Add cross-section at another suboptimal point
    subopt_idx2 = 2 * len(q2_vals) // 3
    cross_section_subopt2 = F_landscape[subopt_idx2, :]
    ax3.plot(q1_vals, cross_section_subopt2, 'g:', linewidth=2.5, 
             label=f'q₂ = {q2_vals[subopt_idx2]:.2f}', marker='^', markersize=4, markevery=10)
    
    ax3.set_xlabel('Perception Parameter (q₁)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Free Energy $\\mathcal{F}$', fontsize=12, fontweight='bold')
    ax3.set_title('Free Energy Cross-Sections\n(Parameter Sensitivity)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.4)
    
    # Panel 4: Active Inference four-fold partition visualization with content
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
• Fuller.4D: Tetrahedral structure with IVM quantization
• Coxeter.4D: Exact Euclidean geometry for measurements  
• Einstein.4D: Information-geometric flows and geodesics

Free Energy Principle:
• Minimization drives perception-action cycles
• Balances prediction error vs. complexity
• Natural gradient follows geodesics on information manifold

Mathematical Structure:
• Variational posterior Q(s) over latent states
• Prior P(s) encoding environmental structure
• Likelihood P(o|s) connecting observations to states"""
    
    ax4.text(0.05, 0.95, partition_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.9, edgecolor='navy'))
    
    # Panel 5: Information geometry metrics with analysis
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
    
    # Plot curvature information with better colormap
    curvature_plot = ax5.imshow(grad_q1_q1 + grad_q2_q2, cmap='RdBu_r', 
                               extent=[q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]])
    
    # Mark the minimum
    ax5.scatter(min_q1, min_q2, c='red', s=150, marker='*', label='Global Minimum', edgecolors='black', linewidth=1)
    
    ax5.set_xlabel('Perception Parameter (q₁)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Action Parameter (q₂)', fontsize=12, fontweight='bold')
    ax5.set_title('Local Curvature\n(Fisher Information Structure)', 
                  fontsize=12, fontweight='bold')
    ax5.legend()
    
    cbar2 = fig.colorbar(curvature_plot, ax=ax5, fraction=0.046, pad=0.04)
    cbar2.set_label("Local Curvature", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    
    outpath = os.path.join(figure_dir, "figure_14_free_energy_landscape.png")
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure 14 saved to: {outpath}")
    
    # Save raw data with metadata
    np.savez(
        os.path.join(data_dir, "figure_14_data.npz"),
        Q1=Q1,
        Q2=Q2,
        F_landscape=F_landscape,
        min_idx=min_idx,
        min_q1=min_q1,
        min_q2=min_q2,
        min_F=min_F,
        curvature_at_min=curvature_at_min,
        metadata={
            'description': 'Free Energy Landscape for Active Inference',
            'parameters': ['perception_parameter', 'action_parameter'],
            'framework': ['Fuller.4D', 'Coxeter.4D', 'Einstein.4D'],
            'resolution': [len(q1_vals), len(q2_vals)]
        }
    )
    
    return outpath


def main():
    """Generate both figures."""
    print("Generating Active Inference figures...")
    
    # Generate 4D Natural Gradient Trajectory
    fig13_path = create_4d_trajectory_visualization()
    
    # Generate Free Energy Landscape
    fig14_path = create_free_energy_landscape()
    
    print("\nFigures generated successfully!")
    print(f"• Figure 13: {fig13_path}")
    print(f"• Figure 14: {fig14_path}")
    print("\nKey features:")
    print("• 4D trajectory visualization with Active Inference dynamics")
    print("• Free energy landscape with information geometry context")
    print("• Natural gradient flows on information manifolds")
    print("• Four-fold partition visualization")
    print("• Visual quality and mathematical rigor")
    print("• Integration with Coxeter.4D, Einstein.4D, and Fuller.4D frameworks")


if __name__ == "__main__":
    main()
