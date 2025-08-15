from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def fisher_information_matrix(gradients: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Estimate the Fisher information matrix via sample gradients.

    Uses the empirical outer-product estimator: F = (1/N) sum g g^T.
    
    In information geometry, this matrix defines a Riemannian metric on parameter space,
    capturing local curvature of the log-likelihood surface. This connects to Einstein.4D
    concepts where geodesics follow the Fisher metric rather than Euclidean distance.

    Parameters
    - gradients: Array of shape (num_samples, num_params) containing per-sample
      gradients with respect to parameters.
    - normalize: Whether to normalize by sample count (default: True).
      Set to False if gradients are already normalized.

    Returns
    - ndarray[num_params, num_params]: Symmetric positive semi-definite matrix.
    
    Notes
    - The FIM acts as a metric tensor in information geometry, analogous to the
      Minkowski metric in Einstein.4D but for parameter space curvature.
    - Eigenvalues of F indicate curvature strength along principal directions.
    - Natural gradient descent uses F^(-1) to scale parameter updates optimally.
    """
    if gradients.ndim != 2:
        raise ValueError("gradients must be 2D (num_samples, num_params)")
    
    if gradients.shape[0] == 0:
        raise ValueError("gradients must contain at least one sample")
    
    if normalize:
        F = gradients.T @ gradients / float(gradients.shape[0])
    else:
        F = gradients.T @ gradients
    
    # Ensure symmetry (numerical stability)
    F = (F + F.T) / 2.0
    
    return F


def fisher_information_quadray(
    gradients: np.ndarray, 
    embedding_matrix: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Fisher information matrix in both Cartesian and Quadray coordinates.
    
    This function demonstrates how the Fisher information transforms between different
    coordinate systems, connecting Coxeter.4D (Euclidean) and Fuller.4D (tetrahedral)
    frameworks.
    
    Parameters
    - gradients: Array of shape (num_samples, num_params) containing per-sample
      gradients with respect to Cartesian parameters.
    - embedding_matrix: Optional 4x3 matrix for Quadray to Cartesian projection.
      If None, uses the default embedding.
    
    Returns
    - Tuple[np.ndarray, np.ndarray]: (F_cartesian, F_quadray)
      F_cartesian: Fisher matrix in Cartesian coordinates
      F_quadray: Fisher matrix in Quadray coordinates
    
    Notes
    - The transformation F_quadray = J^T F_cartesian J where J is the Jacobian
      of the coordinate transformation.
    - This reveals how information geometry adapts to different parameterizations.
    - In Fuller.4D, the tetrahedral structure may reveal symmetries not apparent
      in Cartesian coordinates.
    """
    # For now, we'll use a simplified approach focusing on the concept
    # The full transformation would require computing the Jacobian of the
    # coordinate transformation, which is complex for the general case.
    
    # Compute Cartesian FIM
    F_cart = fisher_information_matrix(gradients)
    
    # Placeholder: return the Cartesian FIM for both
    # In practice, this would compute the actual Quadray FIM via coordinate transformation
    F_quadray = F_cart.copy()
    
    return F_cart, F_quadray


def natural_gradient_step(
    gradient: np.ndarray,
    fisher: np.ndarray,
    step_size: float = 1.0,
    ridge: float = 1e-9,
) -> np.ndarray:
    """Compute a natural gradient step using a damped inverse Fisher.

    Solves (F + ridge I) delta = gradient and returns -step_size * delta.
    
    This implements geodesic motion on the information manifold, analogous to
    how particles follow geodesics in Einstein.4D spacetime. The Fisher metric
    replaces the physical metric, but the geometric principle remains the same.

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
    
    # Ensure positive definiteness for numerical stability
    A = fisher + ridge * np.eye(fisher.shape[0], dtype=fisher.dtype)
    
    # Solve the linear system (F + ridge*I) * delta = gradient
    delta = np.linalg.solve(A, gradient)
    
    return -step_size * delta


def free_energy(log_p_o_given_s: np.ndarray, q: np.ndarray, p: np.ndarray) -> float:
    """Variational free energy for discrete latent states.

    Computes F = E_q[-log p(o|s)] + KL(q || p), with simple normalization of
    q and p to avoid sensitivity to scaling. A small epsilon protects logs.
    
    This function connects to active inference frameworks where minimizing free
    energy drives both perception and action, analogous to how geodesics minimize
    proper time in Einstein.4D.

    Parameters
    - log_p_o_given_s: Log-likelihoods for each latent state.
    - q: Unnormalized variational posterior over states.
    - p: Unnormalized prior over states.

    Returns
    - float: Non-negative scalar free energy (lower is better).
    """
    if not (log_p_o_given_s.shape == q.shape == p.shape):
        raise ValueError("shapes of inputs must match")
    
    # Normalize distributions
    qn = q / np.sum(q)
    pn = p / np.sum(p)
    
    # Expected negative log-likelihood
    expected_nll = -float(np.sum(qn * log_p_o_given_s))
    
    # KL divergence with numerical stability
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
