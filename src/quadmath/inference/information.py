from __future__ import annotations

from fractions import Fraction

import numpy as np
from typing import Optional, Tuple

from quadmath.core.quadray import DEFAULT_EMBEDDING
from quadmath.core.metrics import (
    _normalized_log_weights,
    _relative_entropy,
    _symmetric_part,
    _weighted_sum,
    kl_divergence,
)


def fisher_information_matrix(gradients: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute a sample-gradient outer-product matrix.

    With normalization, F = (1/N) sum g g^T. This estimates empirical Fisher
    information only when the rows are appropriate likelihood scores.
    Arbitrary loss gradients instead yield their uncentered second moment;
    it is not automatically a population Fisher matrix or a loss Hessian.

    Parameters
    - gradients: Finite real array of shape (num_samples, num_params)
      containing per-sample gradients. Integer inputs are converted to floating
      point before multiplication, avoiding fixed-width integer overflow.
    - normalize: Whether to normalize by sample count (default: True).
      Overflow-risk batches use exact product accumulation before conversion,
      preserving representable small entries in mixed-scale parameter sets.
      False returns the unnormalized floating-point sum.

    Returns
    - ndarray[num_params, num_params]: Symmetric positive semi-definite matrix.
    
    Notes
    - The output is positive semidefinite and can be singular; a Riemannian
      metric requires a positive-definite parameterization or damping.
    - Eigenvalues describe this quadratic form, not intrinsic manifold curvature.
    - A statistical Fisher metric is positive semidefinite, unlike a Lorentzian
      spacetime metric. These are not interchangeable geometric structures.
    """
    if np.iscomplexobj(gradients):
        raise ValueError("gradients must be real")
    gradients = np.asarray(gradients, dtype=float)
    if gradients.ndim != 2:
        raise ValueError("gradients must be 2D (num_samples, num_params)")
    if gradients.shape[0] == 0:
        raise ValueError("gradients must contain at least one sample")
    if not np.all(np.isfinite(gradients)):
        raise ValueError("gradients must be finite")
    count = float(gradients.shape[0])
    if normalize and np.max(np.abs(gradients), initial=0.0) > np.sqrt(np.finfo(float).max / count):
        # Dividing every score first can erase small, unrelated curvature.
        # Only the exceptional overflow-risk path pays for exact accumulation.
        columns = [[Fraction.from_float(float(value)) for value in column]
                   for column in gradients.T]
        fisher = np.empty((len(columns), len(columns)), dtype=float)
        for i, column in enumerate(columns):
            for j in range(i, len(columns)):
                total = sum((x * y for x, y in zip(column, columns[j])), Fraction(0))
                fisher[i, j] = fisher[j, i] = float(total / gradients.shape[0])
    else:
        fisher = gradients.T @ gradients
        if normalize:
            fisher /= count
    return _symmetric_part(fisher)


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

    gradients = np.asarray(gradients)
    F_cart = fisher_information_matrix(gradients)
    if gradients.shape[1] != 3:
        raise ValueError("gradients must have shape (num_samples, 3)")
    
    if np.iscomplexobj(embedding_matrix):
        raise ValueError("embedding_matrix must be real")
    E = np.asarray(
        DEFAULT_EMBEDDING if embedding_matrix is None else embedding_matrix,
        dtype=float,
    )
    if E.shape != (3, 4):
        raise ValueError("embedding_matrix must have shape (3, 4)")
    if not np.all(np.isfinite(E)):
        raise ValueError("embedding_matrix must be finite")

    # Equivalent pullback via transformed scores, retaining the stable Gram path.
    F_quadray = fisher_information_matrix(np.asarray(gradients, dtype=float) @ E)
    
    return F_cart, F_quadray


def natural_gradient_step(
    gradient: np.ndarray,
    fisher: np.ndarray,
    step_size: float = 1.0,
    ridge: float = 1e-9,
) -> np.ndarray:
    """Compute a natural gradient step using a damped inverse Fisher.

    Solves (F + ridge I) delta = gradient and returns -step_size * delta.
    
    For a specified positive-definite metric, this is a local metric-steepest-
    descent update. It does not integrate a geodesic. Statistical Fisher
    information requires a likelihood model and appropriate score functions.

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
    
    # Damping shifts eigenvalues; arbitrary indefinite inputs need not become SPD.
    A = fisher + ridge * np.eye(fisher.shape[0], dtype=fisher.dtype)
    
    # Solve the linear system (F + ridge*I) * delta = gradient
    delta = np.linalg.solve(A, gradient)
    
    return -step_size * delta


def free_energy(log_p_o_given_s: np.ndarray, q: np.ndarray, p: np.ndarray) -> float:
    """Variational free energy for discrete latent states.

    Computes F = E_q[-log p(o|s)] + KL(q || p). Finite nonnegative q and p
    are normalized without overflowing their total mass. The KL term uses
    normalized probability pseudocounts of 1e-15, as in kl_divergence.
    
    This is a finite-state variational objective. Perception/action dynamics,
    Markov-blanket independences and a spacetime metric are separate modeling
    assumptions, not consequences of this scalar calculation.

    Parameters
    - log_p_o_given_s: Log-likelihoods for each latent state.
    - q: Finite nonnegative posterior weights of any shape, with positive mass.
    - p: Finite nonnegative prior weights of the same shape, with positive mass.

    Returns
    - float: Scalar free energy (lower is better). Zero-posterior states do not
      contribute to expected surprise, even if their log-likelihood is -inf.
    """
    log_p_o_given_s = np.asarray(log_p_o_given_s)
    q = np.asarray(q)
    p = np.asarray(p)
    if not (log_p_o_given_s.shape == q.shape == p.shape):
        raise ValueError("shapes of inputs must match")
    
    log_q = _normalized_log_weights(q)
    expected_nll = -_weighted_sum(log_q, log_p_o_given_s)
    return expected_nll + kl_divergence(q, p)


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

    Computes the canonical expected free energy G (Parr, Pezzulo & Friston,
    *Active Inference*, MIT Press, 2022):

        G = KL[q(s) || p(s)] - E_q[log p(o|s)] - log p(o)

    Terms: the epistemic KL divergence between posterior and prior over
    states (which already contains the posterior entropy, since
    KL = E_q[log q] - E_q[log p]), the ambiguity/expected-surprise term
    (negative expected log-likelihood of outcomes), and the pragmatic
    preference term (prior preference over outcomes enters negatively, so
    preferred outcomes lower G). G is minimized during action selection in
    Active Inference.

    This function connects to the expected free energy principle where agents
    select actions that minimize expected surprise, analogous to how geodesics
    minimize proper time in Einstein.4D spacetime.

    Parameters
    - log_p_o_given_s: Log-likelihoods log p(o|s) for each latent state.
    - q: Finite nonnegative posterior weights of any shape, with positive mass.
    - p: Finite nonnegative prior weights of the same shape, with positive mass;
      the epistemic KL uses normalized probability pseudocounts of 1e-15.
    - log_p_o: Log of prior preference over outcomes (default: 0.0 for uniform).

    Returns
    - float: Expected free energy (lower is better for action selection).
    """
    return free_energy(log_p_o_given_s, q, p) - log_p_o


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
    - p_joint: Finite nonnegative 2D weights of shape (|X|, |Y|).
    - eps: Finite nonnegative probability pseudocount added to each normalized
      joint entry, followed by renormalization. Zero preserves exact support.

    Returns
    - float: Mutual information in nats (>= 0).

    Raises
    - ValueError: If p_joint is not 2D, weights are undefined, or eps is invalid.
    """
    p_joint = np.asarray(p_joint)
    if p_joint.ndim != 2:
        raise ValueError("p_joint must be 2D")
    log_joint = _normalized_log_weights(p_joint, eps)
    log_x = np.logaddexp.reduce(log_joint, axis=1)
    log_y = np.logaddexp.reduce(log_joint, axis=0)
    return max(0.0, _relative_entropy(log_joint, log_x[:, None] + log_y[None, :]))


def information_gain(prior: np.ndarray, posterior: np.ndarray, eps: float = 1e-15) -> float:
    """Information gain (Bayesian surprise) between prior and posterior.

    Computes D_KL(posterior || prior), measuring how much the beliefs changed
    after observing data.  In active inference this quantifies epistemic value.

    Parameters
    - prior: Finite nonnegative weights of any shape with positive mass.
    - posterior: Finite nonnegative weights of the same shape with positive mass.
    - eps: Finite nonnegative normalized probability pseudocount; see
      kl_divergence. Zero preserves exact supports and may return infinity.

    Returns
    - float: KL divergence in nats (>= 0).

    Raises
    - ValueError: If shapes differ or weights/eps are undefined.
    """
    return kl_divergence(posterior, prior, eps)
