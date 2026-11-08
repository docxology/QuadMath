"""Scientific invariants of the real manuscript figures and their saved data."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from active_inference_figures import create_4d_trajectory_visualization, create_free_energy_landscape
from quadmath.inference.information import free_energy


def test_regression_diagnostics_include_the_actual_final_state(tmp_path, monkeypatch):
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(tmp_path))
    create_4d_trajectory_visualization()
    with np.load(tmp_path / "data/figure_13_data.npz") as data:
        X, y, path = data["X"], data["y"], data["path"]
        residuals = [X @ state - y for state in path]
        expected_loss = [0.5 * float(residual @ residual) for residual in residuals]
        expected_gradient = [np.linalg.norm(X.T @ residual) for residual in residuals]
        np.testing.assert_allclose(data["loss_trace"], expected_loss, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(data["gradient_norms"], expected_gradient, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(data["parameter_changes"], np.linalg.norm(np.diff(path, axis=0), axis=1))
        assert len(data["step_sizes"]) == len(path) - 1
        assert data["loss_trace"][-1] < data["loss_trace"][0]
        optimum_gradient = X.T @ (X @ data["w_least_squares"] - y)
        np.testing.assert_allclose(optimum_gradient, np.zeros(X.shape[1]), atol=1e-13)


def test_categorical_surface_does_not_relabel_normalized_invalid_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(tmp_path))
    create_free_energy_landscape()
    with np.load(tmp_path / "data/figure_14_data.npz") as data:
        q1, q2, q3 = data["Q1"], data["Q2"], data["Q3"]
        valid = q3 >= 0
        assert np.array_equal(np.isfinite(data["F_landscape"]), valid)
        posterior = np.column_stack((q1[valid], q2[valid], q3[valid], np.zeros(valid.sum())))
        np.testing.assert_allclose(posterior.sum(axis=1), 1, atol=1e-15)
        log_q = np.zeros_like(posterior)
        np.log(posterior, out=log_q, where=posterior > 0)
        expected = np.sum(posterior * (log_q - np.log(data["prior"]) - np.log(data["likelihood"])), axis=1)
        np.testing.assert_allclose(data["F_landscape"][valid], expected, atol=1e-11)
        assert np.array_equal(np.isfinite(data["curvature"]), q3 > 0)

        # Independently differentiate the actual variational objective in q units.
        row, col = np.unravel_index(np.nanargmin(data["curvature"]), q1.shape)
        a, b = q1[row, col], q2[row, col]
        step = 1e-4

        def objective(first, second):
            return free_energy(np.log(data["likelihood"]), np.array([first, second, 1 - first - second, 0]), data["prior"])

        center = objective(a, b)
        hessian_trace = (
            objective(a + step, b) + objective(a - step, b)
            + objective(a, b + step) + objective(a, b - step) - 4 * center
        ) / step**2
        np.testing.assert_allclose(data["curvature"][row, col], hessian_trace, rtol=1e-6, atol=1e-5)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="real manuscript animation requires ffmpeg")
def test_displayed_spectral_tips_are_scaled_column_eigenvectors(tmp_path, monkeypatch):
    import information_demo

    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(tmp_path))
    information_demo.main()
    with np.load(tmp_path / "data/fisher_information_eigensystem.npz") as data:
        tips = data["display_vertices"][1:]
        np.testing.assert_allclose(tips @ data["F"], data["eigenvalues"][:, None] * tips, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.sum(tips**2, axis=1), data["eigenvalues"], rtol=1e-12)
        # This catches row/column confusion without depending on eigenvector signs.
        projector = tips.T @ tips
        np.testing.assert_allclose(projector, data["F"], rtol=1e-12, atol=1e-12)
