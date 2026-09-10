import numpy as np
import pytest

from information import (
    fisher_information_matrix, 
    natural_gradient_step, 
    free_energy, 
    fisher_information_quadray,
    expected_free_energy,
    active_inference_step,
    information_geometric_distance,
    mutual_information,
    information_gain,
)


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


def test_natural_gradient_step_runs():
    g = np.array([1.0, -2.0])
    F = np.eye(2)
    step = natural_gradient_step(g, F, step_size=0.1, ridge=1e-9)
    assert step.shape == (2,)


def test_fisher_information_quadray_pullback():
    """F_quadray must be the pullback E^T F_cart E under the embedding."""
    grads = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F_cart, F_quad = fisher_information_quadray(grads)

    assert F_cart.shape == (3, 3)
    assert F_quad.shape == (4, 4)

    from quadray import DEFAULT_EMBEDDING
    E = np.asarray(DEFAULT_EMBEDDING, dtype=float)
    assert np.allclose(F_quad, E.T @ F_cart @ E)


def test_fisher_information_quadray_null_direction():
    """The quadray null direction (1,1,1,1) is in the kernel of F_quadray."""
    grads = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0], [0.3, -0.7, 1.1]])
    _F_cart, F_quad = fisher_information_quadray(grads)
    assert np.allclose(F_quad @ np.ones(4), 0.0, atol=1e-12)


def test_fisher_information_quadray_explicit_embedding():
    """An explicit 3x4 embedding overrides the default; wrong shapes raise."""
    grads = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    from quadray import DEFAULT_EMBEDDING
    E = 0.5 * np.asarray(DEFAULT_EMBEDDING, dtype=float)
    F_cart, F_quad = fisher_information_quadray(grads, embedding_matrix=E)
    assert np.allclose(F_quad, E.T @ F_cart @ E)

    with pytest.raises(ValueError):
        fisher_information_quadray(grads, embedding_matrix=np.eye(3))


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


def test_expected_free_energy_basic():
    """Test basic expected free energy computation (pinned value)."""
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])
    # Canonical G = KL - H - E_q[log p(o|s)] - log p(o); here KL = 0 and
    # G = 0.7135581778 - 0.6931471806 = 0.0204109973.
    G = expected_free_energy(logp, q, p)
    assert np.isfinite(G)
    assert abs(G - 0.020410997260129515) < 1e-12


def test_expected_free_energy_shape_mismatch():
    """Test expected free energy with mismatched shapes."""
    try:
        expected_free_energy(
            np.array([0.0, 0.0]), 
            np.array([1.0]), 
            np.array([1.0])
        )
        assert False
    except ValueError:
        assert True


def test_expected_free_energy_normalization():
    """Test that expected free energy is invariant to scaling of inputs."""
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])
    
    G1 = expected_free_energy(logp, q, p)
    G2 = expected_free_energy(logp, 2.0 * q, 3.0 * p)
    
    assert np.allclose(G1, G2, rtol=1e-10)


def test_expected_free_energy_with_preferences():
    """Test expected free energy with prior preferences (canonical sign)."""
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])

    G_no_pref = expected_free_energy(logp, q, p)
    G_with_pref = expected_free_energy(logp, q, p, log_p_o=-1.0)

    # Canonical G: preferred outcomes (log p(o) = -1) LOWER G by 1.
    assert np.allclose(G_with_pref - G_no_pref, 1.0, rtol=1e-10)


def test_expected_free_energy_pinned_nonuniform():
    """Pinned value for the canonical G decomposition with a non-uniform prior."""
    logp = np.log(np.array([0.9, 0.1]))
    q = np.array([0.8, 0.2])
    p = np.array([0.3, 0.7])
    # ambiguity 0.5448054311 - entropy 0.5004024235 + KL 0.5341108087
    G = expected_free_energy(logp, q, p)
    assert abs(G - 0.5785138162971909) < 1e-12


def test_expected_free_energy_term_signs():
    """Each canonical term enters with its documented sign: ambiguity and KL
    positive, posterior entropy as E_q[log q] = -H, preference negative."""
    logp = np.log(np.array([0.9, 0.1]))
    q = np.array([0.8, 0.2])
    p = np.array([0.3, 0.7])
    qn = q / np.sum(q)
    pn = p / np.sum(p)
    eps = 1e-15
    ambiguity = -float(np.sum(qn * logp))
    neg_entropy = float(np.sum(qn * np.log(qn + eps)))
    kl = float(np.sum(qn * (np.log(qn + eps) - np.log(pn + eps))))
    G = expected_free_energy(logp, q, p, log_p_o=-0.25)
    assert np.allclose(G, ambiguity + neg_entropy + kl + 0.25, rtol=1e-10)


def test_expected_free_energy_prior_enters_via_kl():
    """The prior term is epistemic: G rises with KL(q || p) for fixed q, logp."""
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.8, 0.2])
    G_close = expected_free_energy(logp, q, np.array([0.8, 0.2]))
    G_far = expected_free_energy(logp, q, np.array([0.05, 0.95]))
    assert G_far > G_close


def test_active_inference_step_basic():
    """Test basic active inference step functionality."""
    def joint_free_energy(mu, a):
        return float(np.sum(mu * mu) + np.sum(a * a))
    
    def derivative_op(mu):
        return np.zeros_like(mu)
    
    mu = np.array([1.0, -2.0])
    action = np.array([0.5, -0.25, 1.5])
    
    dmu, da = active_inference_step(mu, action, joint_free_energy, derivative_op)
    
    assert dmu.shape == mu.shape
    assert da.shape == action.shape
    assert np.allclose(dmu, -2.0 * mu)  # grad of mu^2 is 2*mu
    assert np.allclose(da, -2.0 * action)  # grad of a^2 is 2*a


def test_active_inference_step_shape_checks():
    """Test active inference step with invalid shapes."""
    def joint_free_energy(mu, a):
        return float(np.sum(mu * mu) + np.sum(a * a))
    
    def derivative_op(mu):
        return np.zeros_like(mu)
    
    try:
        active_inference_step(
            np.zeros((1, 2)),  # 2D mu
            np.array([1.0, 2.0]), 
            joint_free_energy, 
            derivative_op
        )
        assert False
    except ValueError:
        assert True
    
    try:
        active_inference_step(
            np.array([1.0, 2.0]), 
            np.zeros((1, 2)),  # 2D action
            joint_free_energy, 
            derivative_op
        )
        assert False
    except ValueError:
        assert True


def test_information_geometric_distance_basic():
    """Test basic information-geometric distance computation."""
    F = np.eye(2)  # Identity metric
    x1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 4.0])
    
    distance = information_geometric_distance(F, x1, x2)
    expected = np.sqrt(3*3 + 4*4)  # Euclidean distance for identity metric
    
    assert np.allclose(distance, expected, rtol=1e-10)


def test_information_geometric_distance_non_identity():
    """Test information-geometric distance with non-identity metric."""
    F = np.array([[2.0, 0.5], [0.5, 1.0]])  # Non-identity positive definite
    x1 = np.array([0.0, 0.0])
    x2 = np.array([1.0, 1.0])
    
    distance = information_geometric_distance(F, x1, x2)
    
    # Manual calculation: sqrt([1,1]^T [[2,0.5],[0.5,1]] [1,1])
    # = sqrt(1*2*1 + 1*0.5*1 + 1*0.5*1 + 1*1*1) = sqrt(4)
    expected = 2.0
    
    assert np.allclose(distance, expected, rtol=1e-10)


def test_information_geometric_distance_shape_checks():
    """Test information-geometric distance with invalid shapes."""
    F = np.eye(2)
    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0])
    
    # Test non-square F
    try:
        information_geometric_distance(np.array([[1.0, 2.0]]), x1, x2)
        assert False
    except ValueError:
        assert True
    
    # Test dimension mismatch
    try:
        information_geometric_distance(F, np.array([1.0]), x2)
        assert False
    except ValueError:
        assert True
    
    try:
        information_geometric_distance(F, x1, np.array([1.0]))
        assert False
    except ValueError:
        assert True


def test_information_geometric_distance_zero_distance():
    """Test information-geometric distance for identical points."""
    F = np.array([[2.0, 0.5], [0.5, 1.0]])
    x = np.array([1.0, -2.0])
    
    distance = information_geometric_distance(F, x, x)
    assert np.allclose(distance, 0.0, rtol=1e-10)


# --------------- Mutual information tests ---------------


def test_mutual_information_independent():
    """Independent X and Y should have zero MI."""
    # p(x, y) = p(x) * p(y) => MI = 0
    p_x = np.array([0.3, 0.7])
    p_y = np.array([0.4, 0.6])
    p_joint = np.outer(p_x, p_y)
    mi = mutual_information(p_joint)
    assert np.isclose(mi, 0.0, atol=1e-10)


def test_mutual_information_correlated():
    """Correlated X and Y should have positive MI."""
    # Strongly correlated: diagonal-dominant joint
    p_joint = np.array([[0.45, 0.05], [0.05, 0.45]])
    mi = mutual_information(p_joint)
    assert mi > 0.0


def test_mutual_information_perfect():
    """Perfectly correlated variables: MI = H(X) = H(Y)."""
    p_joint = np.array([[0.5, 0.0], [0.0, 0.5]])
    mi = mutual_information(p_joint)
    expected = np.log(2)  # H(X) for uniform binary
    assert np.isclose(mi, expected, rtol=1e-6)


def test_mutual_information_not_2d():
    with pytest.raises(ValueError):
        mutual_information(np.array([0.5, 0.5]))


def test_mutual_information_zero_mass():
    with pytest.raises(ValueError):
        mutual_information(np.zeros((2, 2)))


# --------------- Information gain tests ---------------


def test_information_gain_no_change():
    """Identical prior and posterior implies zero gain."""
    p = np.array([0.5, 0.5])
    ig = information_gain(p, p)
    assert np.isclose(ig, 0.0, atol=1e-10)


def test_information_gain_positive():
    """Updated beliefs should show positive information gain."""
    prior = np.array([0.5, 0.5])
    posterior = np.array([0.9, 0.1])
    ig = information_gain(prior, posterior)
    assert ig > 0.0


def test_information_gain_shape_mismatch():
    with pytest.raises(ValueError):
        information_gain(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))
