from __future__ import annotations

from typing import Tuple

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

    Parameters
    - F: Square symmetric matrix.

    Returns
    - Tuple[np.ndarray, np.ndarray]: (eigenvalues, eigenvectors)
    """
    if F.shape[0] != F.shape[1]:
        raise ValueError("F must be square")
    w, V = np.linalg.eigh(F)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]
