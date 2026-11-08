#!/usr/bin/env python3
"""Render a four-coefficient regression and a finite-state variational model.

These are separate numerical demonstrations, not biological state dynamics,
perception/action simulations, spacetime models or geodesic integrations.
PNG figures and reproducible NPZ data use the configured QuadMath output root.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def create_4d_trajectory_visualization():
    """Render ridge-preconditioned regression with state-aligned diagnostics."""
    _ensure_src_on_path()
    from quadmath.paths import get_data_dir, get_figure_dir
    from quadmath.inference.information import fisher_information_matrix, natural_gradient_step
    from quadmath.tools.atomic_write import atomic_savez
    from quadmath.viz._common import figure_scope, save_figure, set_axes_equal

    rng = np.random.default_rng(42)
    X = 0.1 * rng.normal(size=(100, 4))
    noise = 0.01 * rng.normal(size=X.shape[0])
    w_true = np.array([0.8, -0.6, 0.4, -0.3])
    y = X @ w_true + noise
    w_est = np.array([0.2, -0.1, 0.05, -0.05])
    w = w_est.copy()
    path = [w.copy()]
    step_sizes = []
    ridge = 1e-3
    step_size = 0.01
    previous_change = None
    stop_reason = "iteration_limit"
    for _ in range(50):
        residuals = X @ w - y
        gradient = X.T @ residuals
        if np.linalg.norm(gradient) < 1e-6:
            stop_reason = "gradient_norm_below_1e-6"
            break
        if previous_change is not None:
            if previous_change < 1e-4:
                step_size = min(step_size * 1.1, 0.1)
            elif previous_change > 0.1:
                step_size = max(step_size * 0.9, 1e-6)
        sample_gradients = X * residuals[:, None]
        gram = fisher_information_matrix(sample_gradients)
        update = natural_gradient_step(gradient, gram, step_size=step_size, ridge=ridge)
        if np.linalg.norm(update) > 1.0:
            step_size *= 0.5
            update = natural_gradient_step(gradient, gram, step_size=step_size, ridge=ridge)
        w = w + update
        previous_change = float(np.linalg.norm(update))
        path.append(w.copy())
        step_sizes.append(step_size)
        if abs(np.mean((X @ w - y) ** 2) - np.mean(residuals ** 2)) < 1e-8:
            stop_reason = "mean_squared_loss_change_below_1e-8"
            break

    path = np.asarray(path)
    residual_history = X @ path.T - y[:, None]
    loss_trace = 0.5 * np.sum(residual_history ** 2, axis=0)
    gradient_norms = np.linalg.norm(X.T @ residual_history, axis=0)
    parameter_changes = np.linalg.norm(np.diff(path, axis=0), axis=1)
    sample_gradient_gram = fisher_information_matrix(X * residual_history[:, -1, None])
    w_least_squares = np.linalg.lstsq(X, y, rcond=None)[0]
    output_path = os.path.join(get_figure_dir(), "figure_13_4d_trajectory.png")

    with plt.rc_context({"font.size": 10}), figure_scope(figsize=(15, 10), layout="constrained") as fig:
        trajectory = fig.add_subplot(2, 3, 1, projection="3d")
        colors = plt.cm.plasma(np.linspace(0, 1, len(path)))
        for index in range(len(path) - 1):
            trajectory.plot(*path[index:index + 2, :3].T, color=colors[index], linewidth=2)
        for position, marker, color, label in (
            (path[0], "o", "green", "Initial coefficients"),
            (path[-1], "*", "red", "Final iterate"),
            (w_true, "^", "gray", "Generating coefficients"),
            (w_least_squares, "x", "blue", "Sample least-squares optimum"),
        ):
            trajectory.scatter(*position[:3], marker=marker, color=color, s=90, label=label)
        trajectory.set(xlabel="w1", ylabel="w2", zlabel="w3")
        trajectory.set_title("Three-coordinate projection\nFourth coefficient shown at right")
        trajectory.legend(fontsize=7)
        set_axes_equal(trajectory)

        loss = fig.add_subplot(2, 3, 2)
        loss.plot(loss_trace, marker="o", markersize=3)
        loss.set(xlabel="Recorded state", ylabel="Half summed squared residuals", yscale="log")
        loss.set_title("Objective at every recorded state")

        coefficients = fig.add_subplot(2, 3, 3)
        for index, color in enumerate(("tab:blue", "tab:orange", "tab:green", "tab:red")):
            coefficients.plot(path[:, index], color=color, label=f"w{index + 1}")
            coefficients.axhline(w_true[index], color=color, linestyle="--", alpha=0.6)
        coefficients.set(xlabel="Recorded state", ylabel="Coefficient value")
        coefficients.set_title("All four coefficients\nDashed lines: generating values")
        coefficients.legend(ncol=4)

        diagnostics = fig.add_subplot(2, 3, 4)
        diagnostics.plot(gradient_norms, color="tab:red", label="Objective gradient norm")
        diagnostics.set(xlabel="Recorded state", ylabel="Gradient L2 norm", yscale="log")
        steps = diagnostics.twinx()
        steps.plot(np.arange(1, len(path)), step_sizes, color="tab:blue", marker=".", label="Applied step size")
        steps.set_ylabel("Applied step-size multiplier", color="tab:blue")
        diagnostics.set_title("Gradient diagnostics and applied updates")
        diagnostics.legend(loc="upper right", fontsize=8)
        steps.legend(loc="lower left", fontsize=8)

        matrix = fig.add_subplot(2, 3, 5)
        magnitude = float(np.max(np.abs(sample_gradient_gram)))
        image = matrix.imshow(sample_gradient_gram, cmap="coolwarm", vmin=-magnitude, vmax=magnitude)
        matrix.set_title("Final sample-gradient Gram matrix\nNot automatically population Fisher information")
        matrix.set_xticks(range(4), labels=[f"w{i + 1}" for i in range(4)])
        matrix.set_yticks(range(4), labels=[f"w{i + 1}" for i in range(4)])
        for row in range(4):
            for col in range(4):
                value = sample_gradient_gram[row, col]
                matrix.text(
                    col, row, f"{value:.1e}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 * magnitude else "black",
                )
        fig.colorbar(image, ax=matrix, fraction=0.046, pad=0.04, label="Mean sample-gradient product")

        residuals_plot = fig.add_subplot(2, 3, 6)
        residuals_plot.scatter(X @ path[-1], residual_history[:, -1], s=14, alpha=0.6)
        residuals_plot.axhline(0, color="black", linewidth=1)
        residuals_plot.set(xlabel="Final fitted response", ylabel="Fitted minus observed response")
        residuals_plot.set_title("Final residuals of the actual linear model")
        fig.suptitle("Seeded linear regression: ridge-preconditioned updates, not biological state dynamics", fontsize=14)
        save_figure(fig, output_path, dpi=180, bbox_inches="tight")

    atomic_savez(
        os.path.join(get_data_dir(), "figure_13_data.npz"),
        X=X, y=y, path=path, loss_trace=loss_trace, gradient_norms=gradient_norms,
        step_sizes=np.asarray(step_sizes), parameter_changes=parameter_changes,
        sample_gradient_gram=sample_gradient_gram, w_true=w_true, w_est=w_est,
        w_least_squares=w_least_squares,
        metadata_json=json.dumps({
            "description": "Ridge-preconditioned four-coefficient linear regression",
            "seed": 42, "parameters": ["w1", "w2", "w3", "w4"],
            "objective": "0.5 * sum((X @ w - y)**2)",
            "metric": "mean outer product of per-sample half-squared-loss gradients",
            "ridge": ridge, "optimization_steps": len(step_sizes), "stop_reason": stop_reason,
            "state_arrays": ["path", "loss_trace", "gradient_norms"],
            "update_arrays": ["step_sizes", "parameter_changes"],
        }, sort_keys=True),
    )
    print(f"Regression figure saved: {output_path}")
    print(f"Recorded updates: {len(step_sizes)}; final loss: {loss_trace[-1]:.6g}; stop: {stop_reason}")
    return output_path


def create_free_energy_landscape():
    """Render a valid three-state face of a four-state categorical simplex."""
    _ensure_src_on_path()
    from matplotlib.colors import LogNorm
    from quadmath.paths import get_data_dir, get_figure_dir
    from quadmath.inference.information import free_energy
    from quadmath.tools.atomic_write import atomic_savez
    from quadmath.viz._common import figure_scope, save_figure

    grid_indices = np.arange(1, 100)
    first, second = np.meshgrid(grid_indices, grid_indices)
    Q1, Q2 = first / 100, second / 100
    Q3 = (100 - first - second) / 100
    valid = Q3 >= 0
    prior = np.full(4, 0.25)
    likelihood = np.array([0.6, 0.4, 0.3, 0.5])
    log_likelihood = np.log(likelihood)
    F_landscape = np.full(Q1.shape, np.nan)
    for row, col in zip(*np.nonzero(valid)):
        q = np.array([Q1[row, col], Q2[row, col], Q3[row, col], 0.0])
        F_landscape[row, col] = free_energy(log_likelihood, q, prior)
    min_idx = np.unravel_index(np.nanargmin(F_landscape), F_landscape.shape)
    min_q1, min_q2, min_F = Q1[min_idx], Q2[min_idx], F_landscape[min_idx]

    # Analytic entropy Hessian on the interior of the ideal unsmoothed face.
    # The evaluated library KL uses normalized probability pseudocounts 1e-15.
    interior = Q3 > 0
    curvature = np.full(Q1.shape, np.nan)
    curvature[interior] = 1 / Q1[interior] + 1 / Q2[interior] + 2 / Q3[interior]
    output_path = os.path.join(get_figure_dir(), "figure_14_free_energy_landscape.png")

    with plt.rc_context({"font.size": 10}), figure_scope(figsize=(15, 10), layout="constrained") as fig:
        surface = fig.add_subplot(2, 3, 1, projection="3d", computed_zorder=False)
        image = surface.plot_surface(Q1, Q2, F_landscape, cmap="viridis", linewidth=0, alpha=0.9)
        surface.scatter(min_q1, min_q2, min_F, color="red", s=90, marker="*", edgecolors="white", depthshade=False, zorder=10, label="Lowest sampled value")
        surface.set(xlabel="q1", ylabel="q2", zlabel="")
        surface.set_title("Variational free energy F(q), in nats\nq3 = 1 - q1 - q2; q4 = 0")
        surface.legend(fontsize=8)

        contours = fig.add_subplot(2, 3, 2)
        levels = contours.contour(Q1, Q2, F_landscape, levels=14)
        contours.clabel(levels, fontsize=7)
        contours.scatter(min_q1, min_q2, color="red", marker="*", s=90)
        contours.set(xlabel="Posterior probability q1", ylabel="Posterior probability q2", aspect="equal")
        contours.set_title("Feasible posterior contours\nExterior masked, not renormalized")

        sections = fig.add_subplot(2, 3, 3)
        rows = sorted({min_idx[0], len(grid_indices) // 3, 2 * len(grid_indices) // 3})
        for row in rows:
            sections.plot(Q1[row], F_landscape[row], label=f"q2 = {Q2[row, 0]:.2f}")
        sections.set(xlabel="Posterior probability q1", ylabel="Free energy (nats)")
        sections.set_title("Cross-sections stop at the simplex boundary")
        sections.legend()

        model = fig.add_subplot(2, 3, 4)
        model.axis("off")
        model.set_title("Declared finite-state model\nand numerical limits")
        model.text(0, 0.95, "\n\n".join((
            "F(q) = expected negative log likelihood + KL(q || p)",
            "Four latent states; only the q4 = 0 face is sampled.",
            "Prior p = (0.25, 0.25, 0.25, 0.25)",
            "Likelihood L = (0.6, 0.4, 0.3, 0.5)",
            "Red star: minimum over feasible sampled points,\nnot an unrestricted global optimum.",
            "Evaluated KL: normalized pseudocount 1e-15.\nCurvature: ideal unsmoothed interior formula.",
        )), transform=model.transAxes, va="top", fontsize=9)

        metric = fig.add_subplot(2, 3, 5)
        image = metric.pcolormesh(Q1, Q2, np.ma.masked_invalid(curvature), cmap="viridis", norm=LogNorm(), shading="auto")
        metric.scatter(min_q1, min_q2, color="red", marker="*", s=90)
        metric.set(xlabel="Posterior probability q1", ylabel="Posterior probability q2", aspect="equal")
        metric.set_title("Ideal entropy Hessian trace\n1/q1 + 1/q2 + 2/q3; interior only")
        fig.colorbar(image, ax=metric, fraction=0.046, pad=0.04, label="Trace (nats / probability squared)")

        posterior = np.array([Q1[min_idx], Q2[min_idx], Q3[min_idx], 0.0])
        probabilities = fig.add_subplot(2, 3, 6)
        indices = np.arange(4)
        probabilities.bar(indices - 0.18, prior, width=0.36, label="Prior")
        probabilities.bar(indices + 0.18, posterior, width=0.36, label="Sampled face minimum")
        probabilities.set_xticks(indices, labels=[f"State {i + 1}" for i in indices])
        probabilities.set_ylabel("Probability")
        probabilities.set_title("Four latent states, restricted posterior\nFourth-state zero is a modeling constraint")
        probabilities.legend(fontsize=8)
        fig.suptitle("Categorical variational free energy: an explicitly restricted probability simplex", fontsize=14)
        save_figure(fig, output_path, dpi=180, bbox_inches="tight")

    atomic_savez(
        os.path.join(get_data_dir(), "figure_14_data.npz"),
        Q1=Q1, Q2=Q2, Q3=Q3, valid=valid, F_landscape=F_landscape,
        min_idx=min_idx, min_q1=min_q1, min_q2=min_q2, min_F=min_F,
        curvature=curvature, curvature_at_min=curvature[min_idx],
        prior=prior, likelihood=likelihood,
        metadata_json=json.dumps({
            "description": "Four-state categorical variational free energy restricted to q4=0",
            "parameters": ["q1", "q2"], "posterior": "(q1, q2, 1-q1-q2, 0)",
            "invalid_domain": "NaN outside q1+q2<=1; no clipping or renormalized square grid",
            "minimum": "feasible sampled-grid minimum only",
            "curvature": "ideal unsmoothed entropy Hessian trace 1/q1+1/q2+2/q3, q3>0",
            "kl_pseudocount": 1e-15, "resolution": list(Q1.shape),
        }, sort_keys=True),
    )
    print(f"Categorical landscape saved: {output_path}")
    return output_path


def main():
    """Generate the separate regression and categorical-model diagnostics."""
    create_4d_trajectory_visualization()
    create_free_energy_landscape()


if __name__ == "__main__":
    main()
