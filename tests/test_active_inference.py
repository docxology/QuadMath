import numpy as np

from information import (
    free_energy,
    finite_difference_gradient,
    perception_update,
    action_update,
)


def test_finite_difference_gradient_quadratic():
    def f(x: np.ndarray) -> float:
        return float(0.5 * x @ x)

    x0 = np.array([1.0, -2.0, 3.0])
    g = finite_difference_gradient(f, x0, epsilon=1e-6)
    # grad of 0.5 x^T x is x
    assert np.allclose(g, x0, atol=1e-5)


def test_perception_action_updates_reduce_free_energy():
    # Simple toy model: 2-state categorical observation with soft alignment to mu and action
    # Let free energy be a convex quadratic around mu* and a* for testing monotonic reduction
    mu_star = np.array([0.2, -0.1])
    a_star = np.array([0.5, -0.3, 0.1])

    def F_mu(mu: np.ndarray) -> float:
        d = mu - mu_star
        return float(0.5 * d @ d)

    def F_a(a: np.ndarray) -> float:
        d = a - a_star
        return float(0.5 * d @ d)

    # derivative operator is zero for this static test
    D = lambda x: np.zeros_like(x)

    mu = np.array([1.0, 1.0])
    a = np.array([0.0, 0.0, 0.0])

    F0_mu = F_mu(mu)
    F0_a = F_a(a)

    # Take several Euler steps along the flows
    for _ in range(25):
        dmu = perception_update(mu, D, F_mu, step_size=0.2)
        mu = mu + dmu  # Euler integration
        da = action_update(a, F_a, step_size=0.2)
        a = a + da

    F1_mu = F_mu(mu)
    F1_a = F_a(a)

    assert F1_mu < F0_mu
    assert F1_a < F0_a


def test_free_energy_discrete_sanity():
    logp = np.log(np.array([0.7, 0.3]))
    q = np.array([0.6, 0.4])
    p = np.array([0.5, 0.5])
    F = free_energy(logp, q, p)
    assert F >= 0.0

