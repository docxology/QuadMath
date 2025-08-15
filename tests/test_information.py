import numpy as np
import pytest

from information import fisher_information_matrix, natural_gradient_step, free_energy, fisher_information_quadray


def test_fisher_information_matrix_shapes():
    grads = np.array([[1.0, 0.0], [0.0, 2.0]])
    F = fisher_information_matrix(grads)
    assert F.shape == (2, 2)


def test_fisher_information_matrix_bad_ndim():
    try:
        fisher_information_matrix(np.array([1.0, 2.0]))
        assert False
    except ValueError:
        assert True


def test_fisher_information_matrix_empty_gradients():
    try:
        fisher_information_matrix(np.array([]).reshape(0, 3))
        assert False
    except ValueError:
        assert True


def test_fisher_information_matrix_symmetry():
    """Test that the FIM is symmetric."""
    grads = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    F = fisher_information_matrix(grads)
    assert np.allclose(F, F.T)


def test_fisher_information_matrix_normalize_false():
    """Test FIM computation without normalization."""
    grads = np.array([[1.0, 0.0], [0.0, 2.0]])
    F_no_norm = fisher_information_matrix(grads, normalize=False)
    F_with_norm = fisher_information_matrix(grads, normalize=True)
    
    # F_no_norm should be 2x the normalized version
    assert np.allclose(F_no_norm, 2.0 * F_with_norm)


def test_fisher_information_matrix_numerical_stability():
    """Test FIM computation with near-symmetric input."""
    # Create slightly asymmetric gradients
    grads = np.array([[1.0, 0.0], [0.0, 2.0], [1.0 + 1e-10, 1.0]])
    F = fisher_information_matrix(grads)
    
    # Should be exactly symmetric despite input asymmetry
    assert np.allclose(F, F.T)


def test_fisher_information_quadray_basic():
    """Test basic quadray FIM functionality."""
    grads = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F_cart, F_quad = fisher_information_quadray(grads)
    
    assert F_cart.shape == (3, 3)
    assert F_quad.shape == (3, 3)
    assert np.allclose(F_cart, F_quad)  # Placeholder implementation


def test_natural_gradient_step_runs():
    g = np.array([1.0, -2.0])
    F = np.eye(2)
    step = natural_gradient_step(g, F, step_size=0.1, ridge=0.0)
    assert step.shape == (2,)


def test_natural_gradient_step_positive_definite():
    """Test that natural gradient step handles positive definiteness."""
    g = np.array([1.0, -2.0])
    F = np.array([[1.0, 0.5], [0.5, 1.0]])  # Positive definite
    step = natural_gradient_step(g, F, step_size=0.1, ridge=1e-9)
    assert step.shape == (2,)


def test_natural_gradient_step_ridge_stabilization():
    """Test that ridge parameter stabilizes ill-conditioned matrices."""
    g = np.array([1.0, -2.0])
    F = np.array([[1.0, 0.99], [0.99, 1.0]])  # Nearly singular
    step = natural_gradient_step(g, F, step_size=0.1, ridge=1e-3)
    assert step.shape == (2,)


def test_free_energy_basic():
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])
    F = free_energy(logp, q, p)
    assert np.isfinite(F)


def test_free_energy_shape_mismatch():
    try:
        free_energy(np.array([0.0, 0.0]), np.array([1.0]), np.array([1.0]))
        assert False
    except ValueError:
        assert True


def test_free_energy_normalization():
    """Test that free energy is invariant to scaling of inputs."""
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])
    
    F1 = free_energy(logp, q, p)
    F2 = free_energy(logp, 2.0 * q, 3.0 * p)
    
    assert np.allclose(F1, F2, rtol=1e-10)


def test_free_energy_numerical_stability():
    """Test free energy with very small probabilities."""
    logp = np.log(np.array([1e-10, 1.0]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])
    
    F = free_energy(logp, q, p)
    assert np.isfinite(F)
    assert F >= 0.0


def test_natural_gradient_shape_mismatch():
    g = np.array([1.0, -2.0, 3.0])
    F = np.eye(2)
    try:
        natural_gradient_step(g, F)
        assert False
    except ValueError:
        assert True


def test_fisher_information_matrix_theoretical():
    """Test FIM computation with known theoretical result."""
    # For gradients g_i = [a_i, b_i], the FIM should be:
    # F = (1/N) * sum_i [a_i^2, a_i*b_i; a_i*b_i, b_i^2]
    
    grads = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    F = fisher_information_matrix(grads)
    
    # Expected result: 35/3 ≈ 11.67, 44/3 ≈ 14.67, 56/3 ≈ 18.67
    expected = np.array([
        [35.0/3, 44.0/3],
        [44.0/3, 56.0/3]
    ])
    
    assert np.allclose(F, expected, rtol=1e-10)


def test_fisher_information_matrix_orthogonal_gradients():
    """Test FIM with orthogonal gradients (should give diagonal matrix)."""
    grads = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    F = fisher_information_matrix(grads)
    
    # Should be diagonal
    assert np.allclose(F[0, 1], 0.0, atol=1e-10)
    assert np.allclose(F[1, 0], 0.0, atol=1e-10)
    assert F[0, 0] > 0
    assert F[1, 1] > 0
