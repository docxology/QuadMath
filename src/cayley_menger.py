from __future__ import annotations

import numpy as np


def tetra_volume_cayley_menger(d2: np.ndarray) -> float:
    """Compute Euclidean tetrahedron volume from squared distances (Coxeter.4D).

    Given an all-pairs squared-distance matrix among the four vertices, this
    constructs the 5x5 Cayley–Menger (CM) matrix and applies the formula
    288 V^2 = det(CM). Negative or zero determinant implies a degenerate
    configuration, for which this function returns 0.0.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices (zeros on diagonal).

    Returns
    - float: Non-negative Euclidean volume (Coxeter.4D/E^3 slice).
    """
    if d2.shape != (4, 4):
        raise ValueError("d2 must be 4x4")
    CM = np.ones((5, 5))
    CM[0, 0] = 0.0
    CM[1:, 1:] = d2
    det = np.linalg.det(CM)
    V2 = det / 288.0
    if V2 <= 0:
        return 0.0
    return float(np.sqrt(V2))


def ivm_tetra_volume_cayley_menger(d2: np.ndarray) -> float:
    """Compute IVM tetravolume from squared distances via Cayley–Menger.

    This applies the synergetics scale factor S3 = sqrt(9/8) to convert the
    Euclidean (XYZ) volume returned by ``tetra_volume_cayley_menger`` into IVM
    tetra-units, consistent with synergetics conventions.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices (zeros on diagonal).

    Returns
    - float: Non-negative tetravolume in IVM units.
    """
    v_xyz = tetra_volume_cayley_menger(d2)
    S3 = float(np.sqrt(9.0 / 8.0))
    return S3 * v_xyz
