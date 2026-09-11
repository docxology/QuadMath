from __future__ import annotations

from typing import Tuple, Dict, Any, Sequence
import numpy as np


def shannon_entropy(p: np.ndarray, eps: float = 1e-15) -> float:
    """Shannon entropy H(p) for a discrete distribution.

    Parameters
    - p: Nonnegative weights; normalized internally.
    - eps: Small constant for numerical stability in the log.

    Returns
    - float: Entropy in nats.
    """
    pn = p / np.sum(p)
    return float(-np.sum(pn * np.log(pn + eps)))


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
    if F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")
    
    # Ensure symmetry for numerical stability
    F_sym = (F + F.T) / 2.0
    
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
    if F.shape[0] != F.shape[1]:
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
    if F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")
    
    # Compute eigendecomposition
    evals, evecs = fim_eigenspectrum(F)
    
    # Basic curvature measures
    trace = float(np.trace(F))
    determinant = float(np.linalg.det(F))
    condition_number = fisher_condition_number(F)
    
    # Anisotropy index: normalized variance of eigenvalues
    mean_eval = np.mean(evals)
    if mean_eval > 0:
        anisotropy_index = float(np.std(evals) / mean_eval)
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
    """Kullback–Leibler divergence D_KL(p || q) for discrete distributions.

    Measures the information lost when q is used to approximate p.  Always
    non-negative; equals zero iff p == q (up to normalization).

    Parameters
    - p: Non-negative weights for the reference distribution.
    - q: Non-negative weights for the approximating distribution.
    - eps: Small constant for numerical stability in the log.

    Returns
    - float: D_KL(p || q) in nats (>= 0).

    Raises
    - ValueError: If p and q have different shapes.
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    pn = p / np.sum(p)
    qn = q / np.sum(q)
    return float(np.sum(pn * (np.log(pn + eps) - np.log(qn + eps))))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """Jensen–Shannon divergence JSD(p || q) for discrete distributions.

    A symmetric, bounded divergence defined as:
        JSD(p || q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q).  The result is in [0, ln(2)] nats.

    Parameters
    - p: Non-negative weights for distribution p.
    - q: Non-negative weights for distribution q.
    - eps: Small constant for numerical stability.

    Returns
    - float: JSD in nats, in [0, ln(2)].

    Raises
    - ValueError: If p and q have different shapes.
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    pn = p / np.sum(p)
    qn = q / np.sum(q)
    m = 0.5 * (pn + qn)
    kl_pm = float(np.sum(pn * (np.log(pn + eps) - np.log(m + eps))))
    kl_qm = float(np.sum(qn * (np.log(qn + eps) - np.log(m + eps))))
    return 0.5 * kl_pm + 0.5 * kl_qm


def fisher_rao_metric(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """Fisher–Rao geodesic distance on the probability simplex.

    The Fisher–Rao metric is the unique Riemannian metric (up to scale)
    that is invariant under sufficient statistics.  For discrete distributions:
        d_FR(p, q) = 2 * arccos( sum_i sqrt(p_i * q_i) )

    This is the geodesic distance on the statistical manifold and connects
    to the Fisher information matrix as its infinitesimal form.

    Parameters
    - p: Non-negative weights for distribution p.
    - q: Non-negative weights for distribution q.
    - eps: Small constant for numerical stability.

    Returns
    - float: Geodesic distance in [0, pi].

    Raises
    - ValueError: If p and q have different shapes.
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    pn = p / np.sum(p)
    qn = q / np.sum(q)
    # Bhattacharyya coefficient
    bc = float(np.sum(np.sqrt(pn * qn + eps * eps)))
    # Clamp for numerical safety
    bc = max(-1.0, min(1.0, bc))
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
