from __future__ import annotations

import numpy as np


def fisher_information_matrix(gradients: np.ndarray) -> np.ndarray:
    """Estimate the Fisher information matrix via sample gradients.

    Uses the empirical outer-product estimator: F = (1/N) sum g g^T.

    Parameters
    - gradients: Array of shape (num_samples, num_params) containing per-sample
      gradients with respect to parameters.

    Returns
    - ndarray[num_params, num_params]: Symmetric positive semi-definite matrix.
    """
    if gradients.ndim != 2:
        raise ValueError("gradients must be 2D (num_samples, num_params)")
    return gradients.T @ gradients / float(gradients.shape[0])


def natural_gradient_step(
    gradient: np.ndarray,
    fisher: np.ndarray,
    step_size: float = 1.0,
    ridge: float = 1e-9,
) -> np.ndarray:
    """Compute a natural gradient step using a damped inverse Fisher.

    Solves (F + ridge I) delta = gradient and returns -step_size * delta.

    Parameters
    - gradient: Array of shape (num_params,).
    - fisher: Square array of shape (num_params, num_params).
    - step_size: Multiplicative step magnitude.
    - ridge: Tikhonov damping added to the Fisher diagonal for stability.

    Returns
    - ndarray[num_params]: The natural gradient update direction.
    """
    if fisher.shape[0] != fisher.shape[1] or fisher.shape[0] != gradient.shape[0]:
        raise ValueError("shape mismatch between fisher and gradient")
    A = fisher + ridge * np.eye(fisher.shape[0], dtype=fisher.dtype)
    return -step_size * np.linalg.solve(A, gradient)


def free_energy(log_p_o_given_s: np.ndarray, q: np.ndarray, p: np.ndarray) -> float:
    """Variational free energy for discrete latent states.

    Computes F = E_q[-log p(o|s)] + KL(q || p), with simple normalization of
    q and p to avoid sensitivity to scaling. A small epsilon protects logs.

    Parameters
    - log_p_o_given_s: Log-likelihoods for each latent state.
    - q: Unnormalized variational posterior over states.
    - p: Unnormalized prior over states.

    Returns
    - float: Non-negative scalar free energy (lower is better).
    """
    if not (log_p_o_given_s.shape == q.shape == p.shape):
        raise ValueError("shapes of inputs must match")
    qn = q / np.sum(q)
    pn = p / np.sum(p)
    expected_nll = -float(np.sum(qn * log_p_o_given_s))
    eps = 1e-15
    kl = float(np.sum(qn * (np.log(qn + eps) - np.log(pn + eps))))
    return expected_nll + kl


def finite_difference_gradient(function: callable, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute numerical gradient of a scalar function via central differences.

    Parameters
    - function: Callable mapping ndarray[D] -> float
    - x: Point of shape (D,) where gradient is evaluated
    - epsilon: Small perturbation size for finite differences

    Returns
    - ndarray[D]: Numerical gradient estimate
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[i] = epsilon
        f_plus = float(function(x + dx))
        f_minus = float(function(x - dx))
        grad[i] = (f_plus - f_minus) / (2.0 * epsilon)
    return grad


def perception_update(
    mu: np.ndarray,
    derivative_operator: callable,
    free_energy_fn: callable,
    step_size: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Continuous-time perception update: dmu/dt = D mu - dF/dmu.

    Parameters
    - mu: Current internal state (D,)
    - derivative_operator: Callable D(mu) -> (D,) providing generalized time-derivative term
    - free_energy_fn: Callable F(mu) -> float (scalar variational free energy as function of mu)
    - step_size: Scaling factor applied to the update (Euler step magnitude if used discretely)
    - epsilon: Finite-difference epsilon for gradient of F

    Returns
    - ndarray[D]: The time derivative dmu/dt (if used as a flow), scaled by step_size
    """
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1:
        raise ValueError("mu must be 1D")
    d_mu = np.asarray(derivative_operator(mu), dtype=float)
    grad_F = finite_difference_gradient(free_energy_fn, mu, epsilon)
    return step_size * (d_mu - grad_F)


def action_update(
    action: np.ndarray,
    free_energy_fn: callable,
    step_size: float = 1.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Continuous-time action update: da/dt = - dF/da.

    Parameters
    - action: Current action vector (K,)
    - free_energy_fn: Callable F(a) -> float (scalar free energy as a function of action)
    - step_size: Scaling factor applied to the update (Euler step magnitude if used discretely)
    - epsilon: Finite-difference epsilon for gradient of F

    Returns
    - ndarray[K]: The time derivative da/dt (if used as a flow), scaled by step_size
    """
    action = np.asarray(action, dtype=float)
    if action.ndim != 1:
        raise ValueError("action must be 1D")
    grad_F = finite_difference_gradient(free_energy_fn, action, epsilon)
    return -step_size * grad_F
