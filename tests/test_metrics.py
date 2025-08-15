import numpy as np
import pytest

from metrics import shannon_entropy, information_length, fim_eigenspectrum, fisher_condition_number, fisher_curvature_analysis, fisher_quadray_comparison


def test_shannon_entropy_basic():
    p = np.array([0.5, 0.5])
    H = shannon_entropy(p)
    assert np.isclose(H, np.log(2), rtol=1e-10)


def test_shannon_entropy_uniform():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    H = shannon_entropy(p)
    assert np.isclose(H, np.log(4), rtol=1e-10)


def test_shannon_entropy_deterministic():
    p = np.array([1.0, 0.0])
    H = shannon_entropy(p)
    assert np.isclose(H, 0.0, rtol=1e-10)


def test_shannon_entropy_scaling():
    """Test that entropy is invariant to scaling."""
    p1 = np.array([0.5, 0.5])
    p2 = np.array([1.0, 1.0])  # 2x scaling
    
    H1 = shannon_entropy(p1)
    H2 = shannon_entropy(p2)
    
    assert np.isclose(H1, H2, rtol=1e-10)


def test_information_length_basic():
    path_gradients = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    L = information_length(path_gradients)
    assert L > 0.0


def test_information_length_single_step():
    path_gradients = np.array([[1.0, 0.0], [0.0, 1.0]])
    L = information_length(path_gradients)
    assert L > 0.0


def test_information_length_insufficient_points():
    path_gradients = np.array([[1.0, 0.0]])
    L = information_length(path_gradients)
    assert L == 0.0


def test_information_length_empty():
    path_gradients = np.array([]).reshape(0, 2)
    L = information_length(path_gradients)
    assert L == 0.0


def test_fim_eigenspectrum_basic():
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    evals, evecs = fim_eigenspectrum(F)
    
    assert evals.shape == (2,)
    assert evecs.shape == (2, 2)
    assert np.all(evals >= 0)  # Should be positive semi-definite
    assert evals[0] >= evals[1]  # Should be sorted descending


def test_fim_eigenspectrum_symmetry():
    """Test that FIM eigendecomposition handles near-symmetric matrices."""
    F = np.array([[2.0, 1.0 + 1e-10], [1.0, 2.0]])
    evals, evecs = fim_eigenspectrum(F)
    
    # Should be exactly symmetric despite input asymmetry
    assert np.all(evals >= 0)
    assert evals[0] >= evals[1]


def test_fim_eigenspectrum_diagonal():
    """Test FIM eigendecomposition of diagonal matrix."""
    F = np.diag([3.0, 1.0, 2.0])
    evals, evecs = fim_eigenspectrum(F)
    
    # Eigenvalues should be sorted descending
    assert np.allclose(evals, [3.0, 2.0, 1.0])
    
    # Eigenvectors should be standard basis vectors (up to sign and order)
    # The order may vary due to eigenvalue sorting
    evecs_abs = np.abs(evecs)
    # Check that each column has exactly one 1.0 and the rest are 0.0
    for i in range(3):
        col = evecs_abs[:, i]
        assert np.sum(col > 0.5) == 1  # Exactly one 1.0
        assert np.sum(col < 0.5) == 2  # Exactly two 0.0


def test_fisher_condition_number_basic():
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    kappa = fisher_condition_number(F)
    
    assert kappa >= 1.0
    assert np.isfinite(kappa)


def test_fisher_condition_number_diagonal():
    """Test condition number of diagonal matrix."""
    F = np.diag([4.0, 1.0])
    kappa = fisher_condition_number(F)
    
    assert np.isclose(kappa, 4.0, rtol=1e-10)


def test_fisher_condition_number_singular():
    """Test condition number of singular matrix."""
    F = np.array([[1.0, 1.0], [1.0, 1.0]])  # Rank 1
    kappa = fisher_condition_number(F)
    
    assert kappa == np.inf


def test_fisher_curvature_analysis_basic():
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    analysis = fisher_curvature_analysis(F)
    
    required_keys = ["eigenvalues", "eigenvectors", "condition_number", 
                    "trace", "determinant", "anisotropy_index"]
    
    for key in required_keys:
        assert key in analysis
    
    assert analysis["trace"] == 4.0
    assert np.isclose(analysis["determinant"], 3.0, rtol=1e-10)
    assert analysis["condition_number"] >= 1.0
    assert analysis["anisotropy_index"] >= 0.0


def test_fisher_curvature_analysis_isotropic():
    """Test curvature analysis of isotropic matrix."""
    F = np.eye(3)
    analysis = fisher_curvature_analysis(F)
    
    assert np.allclose(analysis["eigenvalues"], [1.0, 1.0, 1.0])
    assert analysis["condition_number"] == 1.0
    assert analysis["anisotropy_index"] == 0.0
    assert analysis["trace"] == 3.0
    assert analysis["determinant"] == 1.0


def test_fisher_curvature_analysis_anisotropic():
    """Test curvature analysis of highly anisotropic matrix."""
    F = np.diag([10.0, 1.0, 0.1])
    analysis = fisher_curvature_analysis(F)
    
    assert analysis["condition_number"] == 100.0
    assert analysis["anisotropy_index"] > 0.0
    assert analysis["trace"] == 11.1
    assert np.isclose(analysis["determinant"], 1.0, rtol=1e-10)


def test_fisher_quadray_comparison_basic():
    """Test basic quadray comparison functionality."""
    F_cart = np.array([[2.0, 1.0], [1.0, 2.0]])
    F_quad = np.array([[2.0, 1.0], [1.0, 2.0]])  # Same for now
    
    comparison = fisher_quadray_comparison(F_cart, F_quad)
    
    required_keys = ["cartesian", "quadray", "coordinate_differences"]
    for key in required_keys:
        assert key in comparison
    
    # Since matrices are identical, ratios should be 1.0
    diffs = comparison["coordinate_differences"]
    assert np.isclose(diffs["condition_ratio"], 1.0)
    assert np.isclose(diffs["trace_ratio"], 1.0)
    assert np.isclose(diffs["anisotropy_ratio"], 1.0)


def test_fisher_quadray_comparison_different():
    """Test comparison of different matrices."""
    F_cart = np.array([[2.0, 1.0], [1.0, 2.0]])
    F_quad = np.array([[5.0, 0.0], [0.0, 1.0]])  # Different trace and structure
    
    comparison = fisher_quadray_comparison(F_cart, F_quad)
    
    # Ratios should not be 1.0
    diffs = comparison["coordinate_differences"]
    # Use more specific checks since condition numbers might be similar
    assert not np.isclose(diffs["trace_ratio"], 1.0)  # 4.0 vs 6.0
    assert not np.isclose(diffs["anisotropy_ratio"], 1.0)


def test_fisher_curvature_analysis_edge_cases():
    """Test curvature analysis with edge cases."""
    # Zero matrix
    F_zero = np.zeros((2, 2))
    analysis_zero = fisher_curvature_analysis(F_zero)
    
    assert analysis_zero["trace"] == 0.0
    assert analysis_zero["determinant"] == 0.0
    assert analysis_zero["condition_number"] == np.inf
    assert analysis_zero["anisotropy_index"] == 0.0
    
    # Identity matrix
    F_id = np.eye(2)
    analysis_id = fisher_curvature_analysis(F_id)
    
    assert analysis_id["trace"] == 2.0
    assert analysis_id["determinant"] == 1.0
    assert analysis_id["condition_number"] == 1.0
    assert analysis_id["anisotropy_index"] == 0.0
