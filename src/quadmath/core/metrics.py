from __future__ import annotations

from typing import Tuple, Dict, Any, Sequence
import numpy as np


def _normalized_log_weights(weights: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Normalize in log space without inventing zeros from finite positive weights."""
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be finite and non-negative")
    if np.iscomplexobj(weights):
        raise ValueError("weights must be real")
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    scale = np.max(weights, initial=0.0)
    if scale == 0.0:
        raise ValueError("weights must have positive total mass")
    scaled = weights / scale
    logs = np.full_like(scaled, -np.inf)
    np.log(scaled, out=logs, where=scaled > 0.0)
    lost = (weights > 0.0) & (scaled == 0.0)
    logs[lost] = np.log(weights[lost]) - np.log(scale)
    logs -= np.log(np.sum(scaled))
    if eps > 0.0:
        log_eps = np.log(eps)
        logs = np.logaddexp(logs, log_eps)
        logs -= np.logaddexp(0.0, log_eps + np.log(weights.size))
    return logs


def _weighted_sum(log_p: np.ndarray, values: np.ndarray) -> float:
    """Weight real values, retaining support and recoverable underflowed terms."""
    if np.iscomplexobj(values):
        raise ValueError("weighted values must be real")
    support = np.isfinite(log_p)
    logs = log_p[support]
    values = np.asarray(values, dtype=float)[support]
    probabilities = np.exp(logs)
    finite = np.isfinite(values)
    terms = np.empty_like(probabilities)
    terms[~finite] = values[~finite]
    terms[finite] = probabilities[finite] * values[finite]
    lost = finite & (probabilities == 0.0) & (values != 0.0)
    terms[lost] = np.copysign(
        np.exp(logs[lost] + np.log(np.abs(values[lost]))), values[lost],
    )
    return float(np.sum(terms))


def _entropy(log_p: np.ndarray) -> float:
    """Entropy from normalized log probabilities, with zero support omitted."""
    return -_weighted_sum(log_p, log_p)


def _relative_entropy(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """KL from log probabilities, including genuinely missing reference support."""
    support = np.isfinite(log_p)
    if np.any(np.isneginf(log_q[support])):
        return np.inf
    return _weighted_sum(log_p[support], log_p[support] - log_q[support])


def _symmetric_part(matrix: np.ndarray) -> np.ndarray:
    """Average transposed entries without overflowing their finite sum."""
    matrix = np.asarray(matrix, dtype=float)
    averaged = 0.5 * matrix + 0.5 * matrix.T
    # Equal entries need no averaging; in particular, retain subnormal values.
    return np.where(matrix == matrix.T, matrix, averaged)


def shannon_entropy(p: np.ndarray, eps: float = 1e-15) -> float:
    """Shannon entropy H(p) for a discrete distribution.

    Parameters
    - p: Finite nonnegative weights of any shape; normalized internally.
      Empty, zero-mass, negative and nonfinite weights raise ValueError.
    - eps: Finite nonnegative probability pseudocount added to every normalized
      entry before renormalization (default: 1e-15). Set to zero for exact
      support boundaries, using 0 log 0 = 0.

    Returns
    - float: Entropy in nats.
    """
    return _entropy(_normalized_log_weights(p, eps))


def information_length(path_gradients: np.ndarray) -> float:
    """Gradient-weighted proxy for informational path length (NOT the
    information-geometric arc length).

    Given a sequence of parameter-space vectors along a path, accumulate
    sum ||Δθ_t|| · ||g_t|| where Δθ_t is the difference of consecutive rows.
    Note the rows play two roles (positions for the difference term, gradient
    magnitudes for the weight); this is a heuristic proxy, not the Fisher-Rao
    arc length ∫ √(dθᵀ F dθ), and nothing in the codebase minimizes it.
    
    This connects to Einstein.4D concepts where proper time is measured along
    geodesics, but here we measure information-theoretic "distance" along
    optimization trajectories.

    Parameters
    - path_gradients: Array of shape (T, D), T >= 2.

    Returns
    - float: Non-negative scalar; 0 if T < 2.
    """
    if path_gradients.ndim != 2 or path_gradients.shape[0] < 2:
        return 0.0
    
    L = 0.0
    for t in range(path_gradients.shape[0] - 1):
        g = path_gradients[t]
        dtheta = path_gradients[t + 1] - path_gradients[t]
        L += float(np.linalg.norm(dtheta) * np.linalg.norm(g))
    
    return L


def fim_eigenspectrum(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of a Fisher information matrix.

    Returns eigenvalues and eigenvectors sorted in descending eigenvalue order.
    
    In information geometry, this reveals the principal curvature directions
    of the parameter manifold. Large eigenvalues indicate directions of high
    curvature (tight constraints), while small eigenvalues indicate directions
    of low curvature (loose constraints).
    
    This analysis connects to the anisotropic nature of parameter spaces,
    explaining why natural gradient descent (which scales updates by F^(-1))
    converges more efficiently than standard gradient descent.

    Parameters
    - F: Square symmetric matrix.

    Returns
    - Tuple[np.ndarray, np.ndarray]: (eigenvalues, eigenvectors)
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")

    F_sym = _symmetric_part(F)
    
    # Compute eigendecomposition
    w, V = np.linalg.eigh(F_sym)
    
    # Sort in descending order
    idx = np.argsort(w)[::-1]
    
    return w[idx], V[:, idx]


def fisher_condition_number(F: np.ndarray) -> float:
    """Compute the condition number of the Fisher information matrix.
    
    The condition number κ(F) = λ_max / λ_min measures the anisotropy
    of the parameter space. High condition numbers indicate ill-conditioned
    problems where natural gradient descent provides significant benefits.
    
    This metric connects to the geometric interpretation of the Fisher
    information as a Riemannian metric on parameter space.

    Parameters
    - F: Square symmetric matrix.

    Returns
    - float: Condition number (≥ 1, with 1 indicating perfect conditioning).
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] == 0:
        raise ValueError("F must be square")
    
    evals, _ = fim_eigenspectrum(F)
    
    # Avoid division by zero
    if evals[-1] <= 0:
        return np.inf
    
    return float(evals[0] / evals[-1])


def fisher_curvature_analysis(F: np.ndarray) -> Dict[str, Any]:
    """Comprehensive analysis of Fisher information matrix curvature.
    
    This function provides a complete geometric analysis of the parameter
    space curvature, revealing the anisotropic structure that guides
    optimization strategies.
    
    The analysis connects to information geometry principles where the
    Fisher metric defines the intrinsic geometry of the parameter space,
    analogous to how the Minkowski metric defines spacetime geometry
    in Einstein.4D.

    Parameters
    - F: Square symmetric matrix.

    Returns
    - Dict containing:
        - eigenvalues: Principal curvature strengths
        - eigenvectors: Principal curvature directions
        - condition_number: Anisotropy measure
        - trace: Total curvature
        - determinant: Volume element scaling
        - anisotropy_index: Normalized measure of directional variation
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")
    
    # Compute eigendecomposition
    evals, evecs = fim_eigenspectrum(F)
    
    # Basic curvature measures
    trace = float(np.trace(F))
    determinant = float(np.linalg.det(F))
    condition_number = fisher_condition_number(F)
    
    # Scale before taking moments: std squares and mean sums can overflow.
    scale = np.max(np.abs(evals), initial=0.0)
    scaled_evals = evals / scale if scale > 0.0 else evals
    mean_eval = np.mean(scaled_evals)
    if mean_eval > 0:
        anisotropy_index = float(np.std(scaled_evals) / mean_eval)
    else:
        anisotropy_index = 0.0
    
    return {
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "condition_number": condition_number,
        "trace": trace,
        "determinant": determinant,
        "anisotropy_index": anisotropy_index
    }


def fisher_quadray_comparison(
    F_cartesian: np.ndarray, 
    F_quadray: np.ndarray
) -> Dict[str, Any]:
    """Compare Fisher information matrices between coordinate systems.
    
    This function analyzes how the Fisher information transforms between
    different coordinate representations, revealing coordinate-dependent
    geometric properties.
    
    The comparison connects Coxeter.4D (Euclidean) and Fuller.4D
    (tetrahedral) frameworks, showing how the same underlying geometry
    manifests differently under coordinate transformations.

    Parameters
    - F_cartesian: Fisher matrix in Cartesian coordinates
    - F_quadray: Fisher matrix in Quadray coordinates

    Returns
    - Dict containing comparison metrics and analysis
    """
    # Check matrix compatibility
    if F_cartesian.shape != F_quadray.shape:
        raise ValueError("Matrices must have the same dimensions")
    
    # Analyze both matrices
    cart_analysis = fisher_curvature_analysis(F_cartesian)
    quad_analysis = fisher_curvature_analysis(F_quadray)
    
    # Compare key properties with safe division
    def safe_ratio(a, b):
        """Safely compute ratio, handling zero cases."""
        if b == 0:
            return np.inf if a > 0 else 0.0 if a == 0 else -np.inf
        return a / b
    
    comparison = {
        "cartesian": cart_analysis,
        "quadray": quad_analysis,
        "coordinate_differences": {
            "condition_ratio": safe_ratio(cart_analysis["condition_number"], quad_analysis["condition_number"]),
            "trace_ratio": safe_ratio(cart_analysis["trace"], quad_analysis["trace"]),
            "anisotropy_ratio": safe_ratio(cart_analysis["anisotropy_index"], quad_analysis["anisotropy_index"])
        }
    }
    
    return comparison


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """Kullback–Leibler divergence between two discrete distributions.

    Measures the information lost when q is used to approximate p.  Always
    non-negative; equals zero iff p == q (up to normalization).

    Parameters
    - p: Finite nonnegative weights of any shape with positive total mass.
    - q: Finite nonnegative weights of the same shape and positive total mass.
    - eps: Finite nonnegative probability pseudocount, added after each input
      is normalized and followed by renormalization. With eps=0, missing
      q support where p is positive gives infinity; zero p terms contribute zero.

    Returns
    - float: D_KL(p || q) in nats (>= 0).

    Raises
    - ValueError: If shapes differ or weights/eps are undefined.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    return _relative_entropy(_normalized_log_weights(p, eps), _normalized_log_weights(q, eps))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """Jensen–Shannon divergence between two discrete distributions.

    A symmetric, bounded divergence defined as:
        JSD(p || q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q).  The result is in [0, ln(2)] nats.

    Parameters
    - p: Finite nonnegative weights of any shape with positive total mass.
    - q: Finite nonnegative weights of the same shape and positive total mass.
    - eps: Finite nonnegative probability pseudocount added after normalization
      to each input, followed by renormalization; zero preserves exact supports.

    Returns
    - float: JSD in nats, in [0, ln(2)].

    Raises
    - ValueError: If shapes differ or weights/eps are undefined.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    log_p = _normalized_log_weights(p, eps)
    log_q = _normalized_log_weights(q, eps)
    log_m = np.logaddexp(log_p, log_q) - np.log(2.0)
    return 0.5 * _relative_entropy(log_p, log_m) + 0.5 * _relative_entropy(log_q, log_m)


def fisher_rao_metric(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """Fisher–Rao geodesic distance on the probability simplex.

    The Fisher–Rao metric is the unique Riemannian metric (up to scale)
    that is invariant under sufficient statistics.  For discrete distributions:
        d_FR(p, q) = 2 * arccos( sum_i sqrt(p_i * q_i) )

    This is the geodesic distance on the statistical manifold and connects
    to the Fisher information matrix as its infinitesimal form.

    Parameters
    - p: Finite nonnegative weights of any shape with positive total mass.
    - q: Finite nonnegative weights of the same shape and positive total mass.
    - eps: Finite nonnegative probability pseudocount added after normalization
      to each input, followed by renormalization; zero preserves exact supports.

    Returns
    - float: Geodesic distance in [0, pi].

    Raises
    - ValueError: If shapes differ or weights/eps are undefined.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    log_p = _normalized_log_weights(p, eps)
    log_q = _normalized_log_weights(q, eps)
    bc = float(np.sum(np.exp(0.5 * (log_p + log_q))))
    bc = min(1.0, bc)
    return float(2.0 * np.arccos(bc))


def angle_error(q1: Sequence[float], q2: Sequence[float]) -> float:
    """Geodesic rotation angle between two quaternions, in radians.

    Computes 2*acos(|<q1, q2>|) — the geodesic distance on SO(3) up to the
    quaternion double cover — with the acos argument clamped for numerical
    safety, so the result lies in [0, pi].  Inputs are normalized
    internally (module style, cf. shannon_entropy), so non-unit but
    non-zero quaternions are accepted.  The metric is invariant under
    q -> -q and symmetric in its arguments.  Component order matches the
    (w, x, y, z) convention of the quadmath.core.quadray quaternion helpers.

    Parameters
    - q1, q2: Quaternions as 4-component (w, x, y, z) sequences

    Returns
    - float: Rotation angle in radians, in [0, pi]

    Raises
    - ValueError: If the quaternions are not 4-component sequences of the
      same shape, or either has zero norm
    """
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size != 4:
        raise ValueError("q1 and q2 must be 4-component quaternions of the same shape")
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("quaternions must be non-zero")
    cos_half = abs(float(np.dot(a, b))) / (norm_a * norm_b)
    cos_half = max(-1.0, min(1.0, cos_half))
    return float(2.0 * np.arccos(cos_half))


def quat_log_euclidean_dispersion(quats: Sequence[Sequence[float]]) -> float:
    """Root-mean-square chordal dispersion of quaternions about their mean.

    Computes sqrt(mean_i ||q_i - m||^2), where m is the normalized
    component-wise mean of the inputs and ||.|| the Euclidean (chordal)
    norm in R^4.  Signs are aligned to the first quaternion before
    averaging (each q_i with <q_i, q_0> < 0 is negated) because q and -q
    encode the same rotation; without this the mean of antipodally written
    identical rotations would cancel.  The alignment is deterministic.  The
    component order matches the (w, x, y, z) convention of the
    quadmath.core.quadray quaternion helpers.

    Parameters
    - quats: Non-empty sequence of quaternions, each a 4-component
      (w, x, y, z) sequence

    Returns
    - float: Root-mean-square chord distance to the normalized mean
      quaternion (>= 0); 0 for a single quaternion

    Raises
    - ValueError: If quats is empty, an entry is not 4-component, or the
      sign-aligned mean vanishes (possible only when the first quaternion
      has zero norm, since a non-zero reference forces
      mean . reference = mean_i |<q_i, q_0>| / n > 0)
    """
    arr = np.asarray(quats, dtype=float)
    if arr.size == 0:
        raise ValueError("quats must be non-empty")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("quats must be a sequence of 4-component quaternions")
    reference = arr[0]
    signs = np.where(np.dot(arr, reference) < 0.0, -1.0, 1.0)
    aligned = arr * signs[:, None]
    mean = aligned.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise ValueError("sign-aligned mean quaternion vanishes; inputs are pathologically spread")
    mean_unit = mean / norm
    diffs = aligned - mean_unit
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


__all__ = [
    "shannon_entropy",
    "information_length",
    "fim_eigenspectrum",
    "fisher_condition_number",
    "fisher_curvature_analysis",
    "fisher_quadray_comparison",
    "kl_divergence",
    "jensen_shannon_divergence",
    "fisher_rao_metric",
    "angle_error",
    "quat_log_euclidean_dispersion",
]
