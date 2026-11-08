#!/usr/bin/env python3
"""Render regression gradient-Gram diagnostics and separate optimization examples.

Squared-loss sample gradients are not automatically likelihood scores. The
figures distinguish their second moment, a constant-metric quadratic descent,
a finite-state variational objective and a discrete Quadray path. Artifact
filenames remain stable; figure labels and saved metadata identify quantities.
"""
from __future__ import annotations

import io
import json
import os
import sys
import numpy as np


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _savetxt_atomic(path: str, array: np.ndarray) -> None:
    """Write a comma-delimited ``np.savetxt`` table to ``path`` atomically."""
    from quadmath.tools.atomic_write import atomic_write_text  # noqa: WPS433

    buf = io.StringIO()
    np.savetxt(buf, array, delimiter=",")
    atomic_write_text(path, buf.getvalue())


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    from quadmath.paths import get_data_dir, get_figure_dir
    from quadmath.inference.information import fisher_information_matrix, natural_gradient_step, free_energy
    from quadmath.optimize.discrete_variational import discrete_ivm_descent
    from quadmath.core.quadray import Quadray, DEFAULT_EMBEDDING, to_xyz
    from quadmath.core.metrics import fim_eigenspectrum, fisher_curvature_analysis
    from quadmath.viz.visualize import animate_discrete_path, plot_partition_tetrahedron
    from quadmath.viz._common import figure_scope, save_figure, set_axes_equal
    from quadmath.tools.atomic_write import atomic_savez

    rng = np.random.default_rng(0)
    w_true = np.array([1.0, -2.0, 0.5])
    X = rng.normal(size=(200, 3))
    y = X @ w_true + 0.1 * rng.normal(size=X.shape[0])
    w_est = np.array([0.3, -1.2, 0.0])
    residuals = X @ w_est - y
    grads = 2.0 * X * residuals[:, None]
    gram = fisher_information_matrix(grads)
    figure_dir, data_dir = get_figure_dir(), get_data_dir()
    metadata = json.dumps({
        "seed": 0, "model": "y = X @ w_true + independent Gaussian noise, sigma=0.1",
        "matrix": "mean outer product of per-sample squared-loss gradients",
        "gradient": "2 * x_i * (x_i @ w_est - y_i)",
        "units": "dimensionless synthetic features, targets and coefficients",
    }, sort_keys=True)

    matrix_path = os.path.join(figure_dir, "fisher_information_matrix.png")
    with figure_scope(figsize=(12, 5.5), layout="constrained") as fig:
        fit = fig.add_subplot(1, 2, 1)
        generating_prediction, estimate_prediction = X @ w_true, X @ w_est
        fit.scatter(y, generating_prediction, s=14, alpha=0.6, label="Generating coefficients")
        fit.scatter(y, estimate_prediction, s=14, alpha=0.6, label="Evaluation estimate")
        limits = [min(y.min(), estimate_prediction.min()), max(y.max(), estimate_prediction.max())]
        fit.plot(limits, limits, "k--", linewidth=1, label="Predicted = observed")
        fit.set(xlabel="Observed response y", ylabel="Full three-feature prediction Xw")
        fit.set_title("Actual linear model: predictions versus observations")
        fit.legend(fontsize=8)
        fit.grid(alpha=0.25)

        matrix = fig.add_subplot(1, 2, 2)
        magnitude = float(np.max(np.abs(gram)))
        image = matrix.imshow(gram, cmap="coolwarm", vmin=-magnitude, vmax=magnitude)
        matrix.set_title("Squared-loss sample-gradient Gram matrix")
        matrix.set_xticks(range(3), labels=[f"w{i}" for i in range(3)])
        matrix.set_yticks(range(3), labels=[f"w{i}" for i in range(3)])
        for row in range(3):
            for col in range(3):
                matrix.text(
                    col, row, f"{gram[row, col]:.3f}", ha="center", va="center",
                    color="white" if abs(gram[row, col]) > 0.6 * magnitude else "black",
                )
        fig.colorbar(image, ax=matrix, fraction=0.046, pad=0.04, label="Mean gradient product")
        fig.suptitle("Seeded regression diagnostics — no substituted polynomial or spacetime schematic")
        save_figure(fig, matrix_path, dpi=180, bbox_inches="tight")
    print(matrix_path)
    _savetxt_atomic(os.path.join(data_dir, "fisher_information_matrix.csv"), gram)
    atomic_savez(
        os.path.join(data_dir, "fisher_information_matrix.npz"),
        F=gram, grads=grads, w_true=w_true, w_est=w_est, X=X, y=y, metadata_json=metadata,
    )

    eigenvalues, eigenvectors = fim_eigenspectrum(gram)
    summary = fisher_curvature_analysis(gram)
    # eigh and fim_eigenspectrum return eigenvectors in columns.
    display_vertices = np.vstack((np.zeros(3), eigenvectors.T * np.sqrt(eigenvalues[:, None])))
    spectrum_path = os.path.join(figure_dir, "fisher_information_eigenspectrum.png")
    with figure_scope(figsize=(15, 5.5), layout="constrained") as fig:
        spectrum = fig.add_subplot(1, 3, 1)
        bars = spectrum.bar(range(3), eigenvalues, color=("tab:blue", "tab:orange", "tab:green"))
        spectrum.bar_label(bars, fmt="%.3f", padding=3)
        spectrum.set(xlabel="Descending eigenvalue index", ylabel="Gradient-product eigenvalue")
        spectrum.set_title("Sample-gradient second-moment spectrum")
        spectrum.set_ylim(0, 1.15 * eigenvalues.max())

        diagnostics = fig.add_subplot(1, 3, 2)
        diagnostics.axis("off")
        diagnostics.set_title("Spectral descriptors, not manifold curvature")
        diagnostics.text(0, 0.95, "\n\n".join((
            f"Condition number: {summary['condition_number']:.4g}",
            f"Trace (mean squared gradient norm): {summary['trace']:.4g}",
            f"Determinant: {summary['determinant']:.4g}",
            f"Spectral anisotropy index: {summary['anisotropy_index']:.4g}",
            "Each tip = sqrt(lambda_i) times eigenvector_i.",
            "This is an anisotropy display, not a loss surface,\na Hessian estimate or Riemannian curvature.",
        )), va="top", transform=diagnostics.transAxes, fontsize=10)

        tetrahedron = fig.add_subplot(1, 3, 3, projection="3d")
        for first in range(4):
            for second in range(first + 1, 4):
                tetrahedron.plot(*display_vertices[[first, second]].T, color="gray", linewidth=1)
        tetrahedron.scatter(*display_vertices.T, c=("black", "tab:blue", "tab:orange", "tab:green"), s=50)
        for index, point in enumerate(display_vertices[1:]):
            tetrahedron.text(*point, f"λ{index}", fontsize=9)
        tetrahedron.set(xlabel="w0", ylabel="w1", zlabel="w2")
        tetrahedron.set_title("Column-eigenvector tetrahedron\nEqual parameter-axis scale")
        set_axes_equal(tetrahedron)
        tetrahedron.set_box_aspect(tetrahedron.get_box_aspect(), zoom=0.83)
        save_figure(fig, spectrum_path, dpi=180, bbox_inches="tight")
    print(spectrum_path)
    eigenvalue_path = os.path.join(data_dir, "fisher_information_eigenvalues.csv")
    _savetxt_atomic(eigenvalue_path, eigenvalues[None, :])
    print(eigenvalue_path)
    atomic_savez(
        os.path.join(data_dir, "fisher_information_eigensystem.npz"),
        eigenvalues=eigenvalues, eigenvectors=eigenvectors, display_vertices=display_vertices,
        F=gram, condition_number=summary["condition_number"], trace=summary["trace"],
        determinant=summary["determinant"], anisotropy_index=summary["anisotropy_index"],
        metadata_json=metadata,
    )

    print(plot_partition_tetrahedron((2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 2, 0), (2, 2, 1, 1)))

    # A separate quadratic objective, preconditioned by the fixed Gram above.
    A = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
    w = np.array([2.0, 2.0, 2.0])
    path = [w.copy()]
    for _ in range(20):
        w = w + natural_gradient_step(A @ (w - w_true), gram, step_size=0.5, ridge=1e-3)
        path.append(w.copy())
    path = np.asarray(path)
    delta = path - w_true
    quadratic_loss = 0.5 * np.sum((delta @ A) * delta, axis=1)
    trajectory_path = os.path.join(figure_dir, "natural_gradient_path.png")
    with figure_scope(figsize=(10, 4.5), layout="constrained") as fig:
        trajectory = fig.add_subplot(1, 2, 1)
        trajectory.plot(*path[:, :2].T, marker=".", label="Recorded path")
        trajectory.scatter(*path[0, :2], color="green", label="Initial")
        trajectory.scatter(*path[-1, :2], color="red", label="Final iterate")
        trajectory.scatter(*w_true[:2], marker="x", color="black", label="Quadratic minimizer")
        trajectory.set(xlabel="w0", ylabel="w1", title="Two-coordinate projection; w2 omitted")
        trajectory.legend(fontsize=8)
        loss = fig.add_subplot(1, 2, 2)
        loss.plot(quadratic_loss, marker=".")
        loss.set(xlabel="Recorded state", ylabel="0.5 (w - w*)^T A (w - w*)", yscale="log")
        loss.set_title("Actual quadratic objective along the path")
        fig.suptitle("Constant-metric preconditioned descent, not geodesic integration")
        save_figure(fig, trajectory_path, dpi=180, bbox_inches="tight")
    print(trajectory_path)
    trajectory_csv = os.path.join(data_dir, "natural_gradient_path.csv")
    from quadmath.tools.atomic_write import atomic_write_text
    buffer = io.StringIO()
    np.savetxt(buffer, path, delimiter=",", header="w0,w1,w2", comments="")
    atomic_write_text(trajectory_csv, buffer.getvalue())
    print(trajectory_csv)
    atomic_savez(
        os.path.join(data_dir, "natural_gradient_path.npz"),
        path=path, A=A, F=gram, w_true=w_true, loss=quadratic_loss, ridge=1e-3, step_size=0.5,
    )

    likelihood = np.array([0.7, 0.3])
    q_values = np.linspace(1e-3, 0.999, 200)
    values = np.asarray([
        free_energy(np.log(likelihood), np.array([q, 1 - q]), np.array([0.5, 0.5]))
        for q in q_values
    ])
    free_energy_path = os.path.join(figure_dir, "free_energy_curve.png")
    with figure_scope(figsize=(6, 4), layout="constrained") as fig:
        ax = fig.add_subplot()
        ax.plot(q_values, values)
        minimum = np.argmin(values)
        ax.scatter(q_values[minimum], values[minimum], color="red", label="Lowest sampled value")
        ax.set(xlabel="Posterior probability q(state 0)", ylabel="Variational free energy (nats)")
        ax.set_title("Two-state model: uniform prior, likelihood (0.7, 0.3)")
        ax.legend()
        save_figure(fig, free_energy_path, dpi=180, bbox_inches="tight")
    print(free_energy_path)

    def lattice_objective(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        return float((x - 0.5) ** 2 + (y + 0.2) ** 2 + (z - 0.1) ** 2)

    discrete_path = discrete_ivm_descent(lattice_objective, Quadray(6, 0, 0, 0))
    print(animate_discrete_path(discrete_path, save=True))


if __name__ == "__main__":
    main()


