from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from quadray import DEFAULT_EMBEDDING


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
    - gradients: Array of shape (num_samples, 3) containing per-sample
      gradients with respect to Cartesian (XYZ) parameters.
    - embedding_matrix: Optional 3x4 embedding matrix mapping the quadray
      axes (a, b, c, d) to Cartesian XYZ, as used by `quadray.to_xyz`.
      If None, uses `quadray.DEFAULT_EMBEDDING`.
    
    Returns
    - Tuple[np.ndarray, np.ndarray]: (F_cartesian, F_quadray)
      F_cartesian: Fisher matrix in Cartesian coordinates, shape (3, 3)
      F_quadray: Fisher matrix in Quadray coordinates, shape (4, 4)
    
    Notes
    - The transformation is the standard pullback of the Fisher metric under
      the linear embedding x = E q: with Jacobian J = dx/dq = E (3x4),
      F_quadray = J^T F_cartesian J.
    - Because E (1,1,1,1) = 0 (the quadray null direction), F_quadray is
      singular along (1,1,1,1); the metric acts on the 3D quotient lattice.
    - This reveals how information geometry adapts to different
      parameterizations: the anisotropy of the tetrahedral basis becomes
      explicit in F_quadray.
    """

    F_cart = fisher_information_matrix(gradients)
    
    E = np.asarray(
        DEFAULT_EMBEDDING if embedding_matrix is None else embedding_matrix,
        dtype=float,
    )
    if E.shape != (3, 4):
        raise ValueError("embedding_matrix must have shape (3, 4)")
    
    # Pullback of the Fisher metric under x = E q: F_q = E^T F_cart E
    F_quadray = E.T @ F_cart @ E
    
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


def expected_free_energy(
    log_p_o_given_s: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    log_p_o: float = 0.0
) -> float:
    """Expected free energy for Active Inference with prior preferences.
    
    Computes the expected free energy G = -E_q[log p(o|s)] - E_q[log q(s)] + log p(o),
    which is minimized during action selection in Active Inference. The first two
    terms combine expected surprise (negative expected log-likelihood) with the
    entropy of the variational posterior.
    
    This function connects to the expected free energy principle where agents
    select actions that minimize expected surprise, analogous to how geodesics
    minimize proper time in Einstein.4D spacetime.

    Parameters
    - log_p_o_given_s: Log-likelihoods for each latent state.
    - q: Unnormalized variational posterior over states.
    - p: Unnormalized prior over states.
    - log_p_o: Log of prior preference over outcomes (default: 0.0 for uniform).

    Returns
    - float: Expected free energy (lower is better for action selection).
    """
    if not (log_p_o_given_s.shape == q.shape == p.shape):
        raise ValueError("shapes of inputs must match")
    
    # Normalize distributions
    qn = q / np.sum(q)
    pn = p / np.sum(p)
    
    # Expected log-likelihood
    expected_ll = float(np.sum(qn * log_p_o_given_s))
    
    # Entropy of variational posterior
    eps = 1e-15
    entropy = -float(np.sum(qn * np.log(qn + eps)))
    
    return -expected_ll + entropy + log_p_o


def active_inference_step(
    mu: np.ndarray,
    action: np.ndarray,
    free_energy_fn: callable,
    derivative_operator: callable,
    step_size: float = 1.0,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint perception-action update step in Active Inference.
    
    This function implements a single step of the Active Inference cycle,
    updating both internal states (perception) and actions simultaneously.
    The updates follow the continuous-time flows that minimize free energy,
    connecting to the four-fold partition structure in Fuller.4D.

    Parameters
    - mu: Current internal state (D,)
    - action: Current action vector (K,)
    - free_energy_fn: Callable F(mu, a) -> float (joint free energy)
    - derivative_operator: Callable D(mu) -> (D,) for perception dynamics
    - step_size: Scaling factor for Euler integration
    - epsilon: Finite-difference epsilon for gradients

    Returns
    - Tuple[np.ndarray, np.ndarray]: (dmu, da) time derivatives for integration
    """
    mu = np.asarray(mu, dtype=float)
    action = np.asarray(action, dtype=float)
    
    if mu.ndim != 1:
        raise ValueError("mu must be 1D")
    if action.ndim != 1:
        raise ValueError("action must be 1D")
    
    # Create wrapper functions for partial derivatives
    def F_mu(mu_val: np.ndarray) -> float:
        return free_energy_fn(mu_val, action)
    
    def F_a(a_val: np.ndarray) -> float:
        return free_energy_fn(mu, a_val)
    
    # Compute updates
    dmu = perception_update(mu, derivative_operator, F_mu, step_size, epsilon)
    da = action_update(action, F_a, step_size, epsilon)
    
    return dmu, da


def information_geometric_distance(
    F: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray
) -> float:
    """Compute information-geometric distance between two points.
    
    Computes the geodesic distance d(x1, x2) = sqrt((x2-x1)^T F (x2-x1))
    on the information manifold defined by Fisher metric F.
    
    This function connects to Einstein.4D concepts where the Fisher metric
    replaces the spacetime metric, and geodesics follow information-geometric
    flows rather than physical trajectories.

    Parameters
    - F: Fisher information matrix (positive definite)
    - x1: First point (D,)
    - x2: Second point (D,)

    Returns
    - float: Information-geometric distance
    """
    F = np.asarray(F, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    
    if F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")
    if x1.shape[0] != F.shape[0] or x2.shape[0] != F.shape[0]:
        raise ValueError("dimension mismatch")
    
    # Compute displacement vector
    dx = x2 - x1
    
    # Compute quadratic form dx^T F dx
    distance_squared = float(dx.T @ F @ dx)
    
    return np.sqrt(max(0.0, distance_squared))  # Ensure non-negative


def mutual_information(p_joint: np.ndarray, eps: float = 1e-15) -> float:
    """Mutual information I(X; Y) from a joint probability matrix.

    Computes I(X; Y) = H(X) + H(Y) - H(X, Y) where H denotes Shannon
    entropy.  The joint distribution is normalized internally.

    Parameters
    - p_joint: 2D array of shape (|X|, |Y|) with non-negative entries.
    - eps: Small constant for numerical stability in the log.

    Returns
    - float: Mutual information in nats (>= 0).

    Raises
    - ValueError: If p_joint is not 2D or has non-positive total mass.
    """
    p_joint = np.asarray(p_joint, dtype=float)
    if p_joint.ndim != 2:
        raise ValueError("p_joint must be 2D")
    total = np.sum(p_joint)
    if total <= 0.0:
        raise ValueError("p_joint must have positive total mass")
    pn = p_joint / total

    # Marginals
    p_x = np.sum(pn, axis=1)  # shape (|X|,)
    p_y = np.sum(pn, axis=0)  # shape (|Y|,)

    # Entropies
    h_x = float(-np.sum(p_x * np.log(p_x + eps)))
    h_y = float(-np.sum(p_y * np.log(p_y + eps)))
    h_xy = float(-np.sum(pn * np.log(pn + eps)))

    return max(0.0, h_x + h_y - h_xy)


def information_gain(prior: np.ndarray, posterior: np.ndarray, eps: float = 1e-15) -> float:
    """Information gain (Bayesian surprise) between prior and posterior.

    Computes D_KL(posterior || prior), measuring how much the beliefs changed
    after observing data.  In active inference this quantifies epistemic value.

    Parameters
    - prior: Non-negative weights for the prior distribution.
    - posterior: Non-negative weights for the posterior distribution.
    - eps: Small constant for numerical stability.

    Returns
    - float: KL divergence in nats (>= 0).

    Raises
    - ValueError: If shapes do not match.
    """
    prior = np.asarray(prior, dtype=float)
    posterior = np.asarray(posterior, dtype=float)
    if prior.shape != posterior.shape:
        raise ValueError("prior and posterior must have the same shape")
    pn = posterior / np.sum(posterior)
    qn = prior / np.sum(prior)
    return float(np.sum(pn * (np.log(pn + eps) - np.log(qn + eps))))
