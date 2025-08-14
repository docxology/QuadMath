from __future__ import annotations

from typing import Tuple

import numpy as np

from quadray import Quadray


def urner_embedding(scale: float = 1.0) -> np.ndarray:
    """Return a 3x4 Urner-style symmetric embedding matrix (Fuller.4D -> Coxeter.4D slice).

    The rows map the four quadray axes (A,B,C,D) to the vertices of a regular
    tetrahedron in R^3. Scaling the matrix scales all resulting XYZ coordinates.

    Parameters
    - scale: Uniform scalar applied to the embedding (default 1.0).

    Returns
    - np.ndarray: A 3x4 matrix suitable for use with `quadray_to_xyz`.
    """
    M = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    return scale * M


def quadray_to_xyz(q: Quadray, M: np.ndarray) -> Tuple[float, float, float]:
    """Map a `Quadray` to Cartesian XYZ via a 3x4 embedding matrix (Fuller.4D -> Coxeter.4D slice).

    This is a thin wrapper around a matrix-vector product where columns of `M`
    correspond to the A,B,C,D axes, and rows to X,Y,Z.

    Parameters
    - q: The input quadray coordinate with non-negative components.
    - M: 3x4 embedding matrix (e.g., from `urner_embedding`).

    Returns
    - Tuple[float, float, float]: (x, y, z) in R^3.
    """
    v = np.array([[q.a], [q.b], [q.c], [q.d]], dtype=float)
    xyz = M @ v
    return (float(xyz[0, 0]), float(xyz[1, 0]), float(xyz[2, 0]))

