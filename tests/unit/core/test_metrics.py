import numpy as np
import pytest
import math

from quadmath.core.metrics import shannon_entropy, information_length, fim_eigenspectrum, fisher_condition_number, fisher_curvature_analysis, fisher_quadray_comparison, kl_divergence, jensen_shannon_divergence, fisher_rao_metric, angle_error, quat_log_euclidean_dispersion


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


def test_fisher_condition_number_edge_cases():
    """Test condition number with edge cases."""
    # Matrix with negative eigenvalues (should return inf)
    F_negative = np.array([[1.0, 0.0], [0.0, -0.1]])
    kappa_neg = fisher_condition_number(F_negative)
    assert kappa_neg == np.inf
    
    # Matrix with zero eigenvalue (should return inf)
    F_zero_eval = np.array([[1.0, 0.0], [0.0, 0.0]])
    kappa_zero = fisher_condition_number(F_zero_eval)
    assert kappa_zero == np.inf


def test_fisher_curvature_analysis_zero_mean_eigenvalues():
    """Test curvature analysis when mean eigenvalue is zero."""
    # Matrix with eigenvalues that sum to zero
    F_zero_mean = np.array([[1.0, 0.0], [0.0, -1.0]])
    analysis = fisher_curvature_analysis(F_zero_mean)
    
    # Anisotropy index should be 0.0 when mean eigenvalue is zero
    assert analysis["anisotropy_index"] == 0.0
    assert analysis["trace"] == 0.0
    assert analysis["condition_number"] == np.inf


def test_fisher_quadray_comparison_edge_cases():
    """Test quadray comparison with edge cases."""
    # Zero matrices
    F_zero = np.zeros((2, 2))
    comparison_zero = fisher_quadray_comparison(F_zero, F_zero)
    
    # Should handle zero matrices gracefully
    assert comparison_zero["cartesian"]["trace"] == 0.0
    assert comparison_zero["quadray"]["trace"] == 0.0
    
    # Ratios should be 0.0 when both matrices are zero (0/0 = 0.0 by safe_ratio)
    diffs = comparison_zero["coordinate_differences"]
    assert diffs["trace_ratio"] == 0.0
    assert diffs["anisotropy_ratio"] == 0.0
    
    # Test with one zero matrix and one non-zero matrix
    F_nonzero = np.array([[1.0, 0.0], [0.0, 1.0]])
    comparison_mixed = fisher_quadray_comparison(F_zero, F_nonzero)
    
    # Trace ratio should be 0.0 (0/2)
    assert comparison_mixed["coordinate_differences"]["trace_ratio"] == 0.0
    
    # Different sized matrices (should raise error)
    F_2x2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    F_3x3 = np.eye(3)
    
    with pytest.raises(ValueError):
        fisher_quadray_comparison(F_2x2, F_3x3)


def test_non_square_matrix_errors():
    """Test error handling for non-square matrices."""
    # Non-square matrix for fim_eigenspectrum
    F_rect = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="F must be square"):
        fim_eigenspectrum(F_rect)
    
    # Non-square matrix for fisher_condition_number
    with pytest.raises(ValueError, match="F must be square"):
        fisher_condition_number(F_rect)
    
    # Non-square matrix for fisher_curvature_analysis
    with pytest.raises(ValueError, match="F must be square"):
        fisher_curvature_analysis(F_rect)


# --------------- KL divergence tests ---------------


def test_kl_divergence_identical():
    p = np.array([0.5, 0.5])
    kl = kl_divergence(p, p)
    assert np.isclose(kl, 0.0, atol=1e-10)


def test_kl_divergence_different():
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    kl = kl_divergence(p, q)
    assert kl > 0.0


def test_kl_divergence_shape_error():
    with pytest.raises(ValueError):
        kl_divergence(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


def test_kl_divergence_non_negative():
    rng = np.random.RandomState(42)
    p = rng.dirichlet(np.ones(5))
    q = rng.dirichlet(np.ones(5))
    assert kl_divergence(p, q) >= -1e-12  # Non-negative (up to numerics)


# --------------- Jensen-Shannon divergence tests ---------------


def test_jsd_identical():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    jsd = jensen_shannon_divergence(p, p)
    assert np.isclose(jsd, 0.0, atol=1e-10)


def test_jsd_symmetric():
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    assert np.isclose(jensen_shannon_divergence(p, q), jensen_shannon_divergence(q, p))


def test_jsd_bounded():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    jsd = jensen_shannon_divergence(p, q)
    assert jsd <= np.log(2) + 1e-10


def test_jsd_shape_error():
    with pytest.raises(ValueError):
        jensen_shannon_divergence(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


# --------------- Fisher-Rao metric tests ---------------


def test_fisher_rao_identical():
    p = np.array([0.5, 0.5])
    d = fisher_rao_metric(p, p)
    assert np.isclose(d, 0.0, atol=1e-6)


def test_fisher_rao_positive():
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    d = fisher_rao_metric(p, q)
    assert d > 0.0


def test_fisher_rao_shape_error():
    with pytest.raises(ValueError):
        fisher_rao_metric(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


def test_fisher_rao_symmetric():
    p = np.array([0.7, 0.3])
    q = np.array([0.2, 0.8])
    assert np.isclose(fisher_rao_metric(p, q), fisher_rao_metric(q, p))


# --------------- Quaternion metric tests ---------------


def test_angle_error_self_zero():
    q = (0.5, 0.5, 0.5, 0.5)
    assert np.isclose(angle_error(q, q), 0.0, atol=1e-12)


def test_angle_error_antipodal_zero():
    q = (1.0, 0.0, 0.0, 0.0)
    assert np.isclose(angle_error(q, (-1.0, 0.0, 0.0, 0.0)), 0.0, atol=1e-12)


def test_angle_error_known_quarter_turn():
    # 45-degree quaternion separation encodes a 90-degree rotation difference
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0, 0.0)
    assert np.isclose(angle_error(q1, q2), math.pi / 2.0, atol=1e-12)


def test_angle_error_orthogonal_is_pi():
    # 4D-orthogonal unit quaternions encode opposite rotations
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.0, 1.0, 0.0, 0.0)
    assert np.isclose(angle_error(q1, q2), math.pi, atol=1e-12)


def test_angle_error_symmetric():
    q1 = (math.cos(0.3), math.sin(0.3), 0.0, 0.0)
    q2 = (math.cos(1.1), 0.0, math.sin(1.1), 0.0)
    assert np.isclose(angle_error(q1, q2), angle_error(q2, q1), rtol=1e-12, atol=1e-12)


def test_angle_error_range_and_determinism():
    qs = [
        (1.0, 0.0, 0.0, 0.0),
        (math.cos(0.7), math.sin(0.7), 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5),
        (math.cos(2.0), 0.0, 0.0, math.sin(2.0)),
    ]
    for q1 in qs:
        for q2 in qs:
            e1 = angle_error(q1, q2)
            assert e1 == angle_error(q1, q2)  # repeat call identical
            assert 0.0 <= e1 <= math.pi + 1e-12


def test_angle_error_normalizes_inputs():
    # Non-unit (non-zero) quaternions are normalized internally
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0, 0.0)
    scaled = tuple(3.0 * c for c in q2)
    assert np.isclose(angle_error(q1, scaled), math.pi / 2.0, atol=1e-12)


def test_angle_error_zero_norm_raises():
    with pytest.raises(ValueError):
        angle_error((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def test_angle_error_shape_mismatch_raises():
    with pytest.raises(ValueError):
        angle_error((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def test_quat_dispersion_identical_zero():
    qs = [(0.5, 0.5, 0.5, 0.5)] * 3
    assert np.isclose(quat_log_euclidean_dispersion(qs), 0.0, atol=1e-12)


def test_quat_dispersion_single_zero():
    assert np.isclose(quat_log_euclidean_dispersion([(1.0, 0.0, 0.0, 0.0)]), 0.0, atol=1e-12)


def test_quat_dispersion_positive_and_exact():
    qs = [
        (1.0, 0.0, 0.0, 0.0),
        (math.cos(0.1), math.sin(0.1), 0.0, 0.0),
    ]
    d = quat_log_euclidean_dispersion(qs)
    assert d > 0.0
    # The normalized mean is the halfway quaternion; each point sits at
    # chord distance sqrt(2 - 2*cos(0.05)) = 2*sin(0.025) from it
    assert np.isclose(d, 2.0 * math.sin(0.025), atol=1e-12)


def test_quat_dispersion_sign_alignment():
    # q and -q encode the same rotation: aligned dispersion must vanish
    q = (0.5, 0.5, 0.5, 0.5)
    assert np.isclose(quat_log_euclidean_dispersion([q, (-0.5, -0.5, -0.5, -0.5)]), 0.0, atol=1e-12)


def test_quat_dispersion_empty_raises():
    with pytest.raises(ValueError):
        quat_log_euclidean_dispersion([])


def test_quat_dispersion_bad_shape_raises():
    with pytest.raises(ValueError):
        quat_log_euclidean_dispersion([(1.0, 0.0, 0.0)])


def test_quat_dispersion_degenerate_mean_raises():
    # With a non-zero first (reference) quaternion the aligned mean can
    # never vanish (mean . ref = mean_i |<q_i, q_0>| / n > 0), so the
    # zero-mean branch is reachable only from a zero first quaternion:
    # every dot product is 0, nothing is sign-flipped, and the mean is
    # exactly the zero vector.
    qs = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
    ]
    with pytest.raises(ValueError):
        quat_log_euclidean_dispersion(qs)


def test_quat_dispersion_deterministic():
    qs = [
        (1.0, 0.0, 0.0, 0.0),
        (math.cos(0.2), math.sin(0.2), 0.0, 0.0),
        (math.cos(0.2), 0.0, math.sin(0.2), 0.0),
    ]
    assert quat_log_euclidean_dispersion(qs) == quat_log_euclidean_dispersion(qs)


@pytest.mark.parametrize("scale", [np.nextafter(0.0, 1.0), 1e-308, 1.0, 1e308])
def test_entropy_extreme_weight_scales(scale):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert shannon_entropy([scale, scale]) == pytest.approx(np.log(2))


def test_entropy_integer_totals_and_joint_weights():
    assert shannon_entropy(np.array([2**62, 2**62], dtype=np.int64)) == pytest.approx(np.log(2))
    assert shannon_entropy([[1, 0], [0, 1]], eps=0) == pytest.approx(np.log(2))
    assert shannon_entropy(7, eps=0) == 0.0


@pytest.mark.parametrize("metric", [kl_divergence, jensen_shannon_divergence, fisher_rao_metric])
@pytest.mark.parametrize("scale", [1e-307, 1.0, 1e308])
def test_probability_metrics_scale_invariance_and_joint_arrays(metric, scale):
    p = np.array([[1.0, 0.5], [0.25, 0.0]])
    q = np.array([[0.5, 1.0], [0.0, 0.25]])
    expected = metric(p.ravel(), q.ravel())
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert metric(p * scale, q * scale) == pytest.approx(expected)
    assert metric(np.array([2**62, 2**62], dtype=np.int64), [1, 1]) == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("weights", [
    [], [0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [1j, 1],
])
def test_probability_metrics_reject_undefined_weights(weights):
    with pytest.raises(ValueError):
        shannon_entropy(weights)
    valid = np.ones(np.asarray(weights).shape)
    for metric in (kl_divergence, jensen_shannon_divergence, fisher_rao_metric):
        with pytest.raises(ValueError):
            metric(weights, valid)
        with pytest.raises(ValueError):
            metric(valid, weights)


@pytest.mark.parametrize("eps", [-1.0, np.nan, np.inf])
def test_probability_metrics_reject_invalid_smoothing(eps):
    with pytest.raises(ValueError):
        shannon_entropy([1, 0], eps=eps)
    for metric in (kl_divergence, jensen_shannon_divergence, fisher_rao_metric):
        with pytest.raises(ValueError):
            metric([1, 0], [0, 1], eps=eps)


def test_probability_pseudocounts_define_real_smoothed_distributions():
    entropy = -0.75 * np.log(0.75) - 0.25 * np.log(0.25)
    assert shannon_entropy([1, 0], eps=0.5) == pytest.approx(entropy)
    assert kl_divergence([1, 0], [0, 1], eps=0.5) == pytest.approx(0.5 * np.log(3))
    assert jensen_shannon_divergence([1, 0], [0, 1], eps=0.5) == pytest.approx(np.log(2) - entropy)
    assert fisher_rao_metric([1, 0], [0, 1], eps=0.5) == pytest.approx(np.pi / 3)
    assert fisher_rao_metric([1, 0], [1, 0], eps=0.5) == pytest.approx(0, abs=1e-7)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert shannon_entropy([1, 0], eps=1e308) == pytest.approx(np.log(2))
        assert kl_divergence([1, 0], [0, 1], eps=1e308) == 0.0


def test_probability_exact_support_boundaries():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert shannon_entropy([1, 0], eps=0) == 0.0
        assert kl_divergence([1, 0], [1, 0], eps=0) == 0.0
        assert kl_divergence([1, 0], [0, 1], eps=0) == np.inf
        assert jensen_shannon_divergence([1, 0], [0, 1], eps=0) == pytest.approx(np.log(2))
        assert fisher_rao_metric([1, 0], [0, 1], eps=0) == pytest.approx(np.pi)


@pytest.mark.parametrize("matrix", [
    np.array([[2**62]], dtype=np.int64),
    np.array([[1e308]]),
    np.diag([1e308, 5e307]),
])
def test_fisher_eigenspectrum_avoids_symmetrization_overflow(matrix):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values, vectors = fim_eigenspectrum(matrix)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0)
        assert np.allclose(values, np.diag(matrix))
        assert np.allclose(vectors.T @ vectors, np.eye(matrix.shape[0]))
        assert fisher_condition_number(matrix) == pytest.approx(values[0] / values[-1])


def test_fisher_curvature_anisotropy_is_scale_invariant():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        large = fisher_curvature_analysis(np.array([[1e308]]))
    assert large["anisotropy_index"] == 0.0
    assert large["trace"] == 1e308
    assert large["condition_number"] == 1.0


@pytest.mark.parametrize("metric", [fim_eigenspectrum, fisher_condition_number, fisher_curvature_analysis])
def test_fisher_metrics_reject_nonmatrix_inputs(metric):
    with pytest.raises(ValueError, match="square"):
        metric(np.array([1.0]))


def test_fisher_condition_number_rejects_empty_spectrum():
    with pytest.raises(ValueError, match="square"):
        fisher_condition_number(np.empty((0, 0)))


def test_fisher_eigenspectrum_retains_subnormal_diagonal():
    smallest = np.nextafter(0.0, 1.0)
    values, _ = fim_eigenspectrum(np.array([[smallest]]))
    assert values[0] == smallest


def test_jsd_identical_distributions_preserve_subnormal_support():
    distribution = [np.nextafter(0.0, 1.0), 1.0]
    assert jensen_shannon_divergence(distribution, distribution, eps=0) == 0.0


def test_kl_retains_positive_reference_support_below_probability_float_range():
    smallest = np.nextafter(0.0, 1.0)
    expected = np.log(1e308) - np.log(smallest)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = kl_divergence([1, 0], [smallest, 1e308], eps=0)
    assert actual == pytest.approx(expected)


def test_entropy_retains_representable_terms_from_underflowed_probabilities():
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 80
        probability = Decimal("1e-16") / Decimal("1e308")
        expected = float(-probability * probability.ln())
    assert shannon_entropy([1e-16, 1e308], eps=0) == pytest.approx(expected, rel=0.01, abs=0)
