#!/usr/bin/env python3
"""Generate illustrative figures for information theory components.

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
    import matplotlib.pyplot as plt  # noqa: WPS433

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

    # Visualize F matrix
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(F, cmap="viridis")
    ax.set_title("Fisher information (empirical)")
    ax.set_xlabel("parameter index")
    ax.set_ylabel("parameter index")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F_ij")
    figure_dir = get_figure_dir()
    data_dir = get_data_dir()
    fim_path = os.path.join(figure_dir, "fisher_information_matrix.png")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.92, bottom=0.12)
    fig.savefig(fim_path, dpi=160)
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

    # Eigenspectrum of the Fisher information (curvature along principal axes)
    from metrics import fim_eigenspectrum  # noqa: WPS433
    evals, evecs = fim_eigenspectrum(F)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.bar(np.arange(evals.size), evals)
    ax.set_title("Fisher information eigenspectrum")
    ax.set_xlabel("eigen-index")
    ax.set_ylabel("eigenvalue (curvature)")
    eig_path = os.path.join(figure_dir, "fisher_information_eigenspectrum.png")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.15)
    fig.savefig(eig_path, dpi=160)
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
    ax.plot(path[:, 0], path[:, 1], marker="o")
    ax.set_xlabel("w0")
    ax.set_ylabel("w1")
    ax.set_title("Natural gradient trajectory (proj w0-w1)")
    ng_path = os.path.join(figure_dir, "natural_gradient_path.png")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.14)
    fig.savefig(ng_path, dpi=160)
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
    ax.plot(q_vals, F_vals)
    ax.set_xlabel("q(state=0)")
    ax.set_ylabel("Free energy")
    ax.set_title("Variational free energy (2-state)")
    fe_path = os.path.join(figure_dir, "free_energy_curve.png")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.18)
    fig.savefig(fe_path, dpi=160)
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


