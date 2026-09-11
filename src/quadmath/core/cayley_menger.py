from __future__ import annotations

from typing import Iterable, Tuple

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

    Unit convention: ``d2`` must be measured in units where the
    close-packing sphere radius is 1, i.e. the unit IVM tetrahedron has edge
    2 (XYZ volume 4/(3*sqrt(2)) <-> IVM volume 1). Feeding distances at other
    scales (e.g. directly from `squared_distances_from_quadrays` with
    `DEFAULT_EMBEDDING`, whose unit tetra edge is 2*sqrt(2)) rescales IVM
    volumes by (edge/2)^3.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices (zeros on diagonal).

    Returns
    - float: Non-negative tetravolume in IVM units.
    """
    v_xyz = tetra_volume_cayley_menger(d2)
    S3 = float(np.sqrt(9.0 / 8.0))
    return S3 * v_xyz


def squared_distances_from_quadrays(
    p0: "Quadray", p1: "Quadray", p2: "Quadray", p3: "Quadray",
    embedding: Iterable[Iterable[float]],
) -> np.ndarray:
    """Build the 4x4 squared-distance matrix from four quadray vertices.

    Maps each quadray to R^3 via the given embedding and computes all pairwise
    squared Euclidean distances.  The result can be passed directly to
    ``tetra_volume_cayley_menger`` or ``ivm_tetra_volume_cayley_menger``.

    Parameters
    - p0, p1, p2, p3: Quadray vertices (Fuller.4D)
    - embedding: 3x4 matrix mapping Fuller.4D -> Coxeter.4D/XYZ slice

    Returns
    - np.ndarray: 4x4 symmetric matrix with zeros on the diagonal.

    Note: with `quadray.DEFAULT_EMBEDDING` the unit IVM tetrahedron (origin
    plus three IVM neighbor moves) has edge 2*sqrt(2); rescale the embedding
    by 1/sqrt(2) if results must match the sphere-radius-1 convention of
    `ivm_tetra_volume_cayley_menger`.
    """
    from quadmath.core.quadray import to_xyz
    pts = [np.array(to_xyz(p, embedding)) for p in (p0, p1, p2, p3)]
    d2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            diff = pts[i] - pts[j]
            val = float(diff @ diff)
            d2[i, j] = val
            d2[j, i] = val
    return d2


def tetra_circumradius(d2: np.ndarray) -> float:
    """Circumscribed sphere radius of a tetrahedron from squared distances.

    Uses the Cayley–Menger determinant relation:
        R = sqrt( -det(CM_ext) / (2 * det(CM)) )
    where CM is the standard 5x5 Cayley–Menger matrix and CM_ext is the
    bordered matrix that yields the circumradius.

    For degenerate tetrahedra (zero volume) the circumradius is undefined
    and this function returns 0.0.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices.

    Returns
    - float: Non-negative circumradius in Euclidean units.
    """
    if d2.shape != (4, 4):
        raise ValueError("d2 must be 4x4")
    # Build standard 5x5 CM matrix
    CM = np.ones((5, 5))
    CM[0, 0] = 0.0
    CM[1:, 1:] = d2
    det_CM = np.linalg.det(CM)
    if abs(det_CM) < 1e-30:
        return 0.0
    # Circumradius formula: R^2 = -CM_cofactor(0,0) / (2 * det(CM)), where the
    # cofactor is det(minor(0,0)) of the 4x4 minor obtained by deleting row 0
    # and column 0 of the 5x5 Cayley-Menger matrix.
    minor_00 = CM[1:, 1:]
    det_minor = np.linalg.det(minor_00)
    R2 = -det_minor / (2.0 * det_CM)
    if R2 <= 0:
        return 0.0
    return float(np.sqrt(R2))


def tetra_inradius(d2: np.ndarray) -> float:
    """Inscribed sphere radius of a tetrahedron from squared distances.

    Uses the relation r = 3V / A where V is the volume and A is the total
    surface area.  Surface area is computed by summing the areas of the four
    triangular faces using Heron's formula.

    For degenerate tetrahedra (zero volume) returns 0.0.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices.

    Returns
    - float: Non-negative inradius in Euclidean units.
    """
    if d2.shape != (4, 4):
        raise ValueError("d2 must be 4x4")
    V = tetra_volume_cayley_menger(d2)
    if V <= 0.0:
        return 0.0

    # Compute surface area via Heron's formula for each face
    # Faces are (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    total_area = 0.0
    for i, j, k in faces:
        a = np.sqrt(max(0.0, d2[i, j]))
        b = np.sqrt(max(0.0, d2[i, k]))
        c = np.sqrt(max(0.0, d2[j, k]))
        s = (a + b + c) / 2.0
        area2 = s * (s - a) * (s - b) * (s - c)
        total_area += np.sqrt(max(0.0, area2))

    # Note: if V > 0, at least one face must have positive area,
    # so total_area > 0 is guaranteed here.
    return float(3.0 * V / total_area)
