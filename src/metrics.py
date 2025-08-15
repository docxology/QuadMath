from __future__ import annotations

from typing import Tuple, Dict, Any
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
    """Path length in information space via gradient-weighted arc length.

    Given a sequence of parameter gradients along a path, accumulate
    sum ||Δθ_t|| · ||g_t|| as a simple proxy for informational path length.
    
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
    # Analyze both matrices
    cart_analysis = fisher_curvature_analysis(F_cartesian)
    quad_analysis = fisher_curvature_analysis(F_quadray)
    
    # Compare key properties
    comparison = {
        "cartesian": cart_analysis,
        "quadray": quad_analysis,
        "coordinate_differences": {
            "condition_ratio": cart_analysis["condition_number"] / quad_analysis["condition_number"],
            "trace_ratio": cart_analysis["trace"] / quad_analysis["trace"],
            "anisotropy_ratio": cart_analysis["anisotropy_index"] / quad_analysis["anisotropy_index"]
        }
    }
    
    return comparison
