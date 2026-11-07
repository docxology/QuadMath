from __future__ import annotations

from math import frexp, ldexp, sqrt
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from quadmath.core.quadray import Quadray


def _normalized_distances(d2: np.ndarray) -> tuple[np.ndarray, float]:
    """Separate dimensionless squared distances from their squared-length scale."""
    distances = np.asarray(d2, dtype=np.float64)
    if distances.shape != (4, 4):
        raise ValueError("d2 must be 4x4")
    scale = float(np.max(np.abs(distances)))
    return (distances / scale if scale > 0.0 else distances), scale


def _cayley_menger_matrix(d2: np.ndarray) -> np.ndarray:
    """Build a dimensionless Cayley–Menger matrix."""
    matrix = np.ones((5, 5))
    matrix[0, 0] = 0.0
    matrix[1:, 1:] = d2
    return matrix


def _normalized_volume(d2: np.ndarray) -> float:
    """Compute volume before restoring dimensional units."""
    determinant = float(np.linalg.det(_cayley_menger_matrix(d2)))
    if determinant <= 0.0:
        return 0.0
    return sqrt(determinant) / sqrt(288.0)


def _restore_units(value: float, squared_scale: float, dimension: int) -> float:
    """Multiply by scale**(dimension/2) without overflowing intermediate powers."""
    if value == 0.0:
        return 0.0
    value_mantissa, value_exponent = frexp(value)
    scale_mantissa, scale_exponent = frexp(squared_scale)
    exponent, odd = divmod(scale_exponent * dimension, 2)
    mantissa = value_mantissa * scale_mantissa ** (dimension / 2.0)
    if odd:
        mantissa *= sqrt(2.0)
    try:
        return ldexp(mantissa, value_exponent + exponent)
    except OverflowError:
        return float("inf")


def tetra_volume_cayley_menger(d2: np.ndarray) -> float:
    """Compute Euclidean tetrahedron volume from squared distances (Coxeter.4D).

    Given an all-pairs squared-distance matrix among the four vertices, this
    constructs the 5x5 Cayley–Menger (CM) matrix and applies the formula
    288 V^2 = det(CM). Negative or zero determinant implies a degenerate
    configuration, for which this function returns 0.0.

    Squared distances are normalized before taking the determinant. Units are
    restored with exponent scaling, so uniformly tiny or huge tetrahedra do
    not underflow or overflow unless the final volume itself is out of range.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices (zeros on diagonal).

    Returns
    - float: Non-negative Euclidean volume (Coxeter.4D/E^3 slice).
    """
    normalized, scale = _normalized_distances(d2)
    return _restore_units(_normalized_volume(normalized), scale, 3)


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
    normalized, scale = _normalized_distances(d2)
    S3 = float(np.sqrt(9.0 / 8.0))
    return _restore_units(S3 * _normalized_volume(normalized), scale, 3)


def squared_distances_from_quadrays(
    p0: "Quadray", p1: "Quadray", p2: "Quadray", p3: "Quadray",
    embedding: Iterable[Iterable[float]],
) -> np.ndarray:
    """Build the 4x4 squared-distance matrix from four quadray vertices.

    Embeds each integer pairwise displacement, rather than subtracting
    floating images of absolute positions, preserving spatial translations.
    The result can be passed directly to ``tetra_volume_cayley_menger`` or
    ``ivm_tetra_volume_cayley_menger``.

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
    embedding = tuple(tuple(row) for row in embedding)
    pts = (p0, p1, p2, p3)
    d2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            diff = np.asarray(to_xyz(pts[i].sub(pts[j]), embedding))
            val = float(diff @ diff)
            d2[i, j] = val
            d2[j, i] = val
    return d2


def tetra_circumradius(d2: np.ndarray) -> float:
    """Circumscribed sphere radius of a tetrahedron from squared distances.

    Uses the Cayley–Menger relation
    ``R^2 = -det(d2) / (2 * det(CM))`` on dimensionless squared distances,
    then restores length units. There is no dimensional determinant cutoff:
    uniform scaling cannot turn a nondegenerate tetrahedron into a zero.

    For degenerate tetrahedra (zero volume) the circumradius is undefined
    and this function returns 0.0.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices.

    Returns
    - float: Non-negative circumradius in Euclidean units.
    """
    normalized, scale = _normalized_distances(d2)
    det_CM = float(np.linalg.det(_cayley_menger_matrix(normalized)))
    if det_CM <= 0.0:
        return 0.0
    det_minor = float(np.linalg.det(normalized))
    R2 = -det_minor / (2.0 * det_CM)
    return _restore_units(sqrt(R2), scale, 1)


def tetra_inradius(d2: np.ndarray) -> float:
    """Inscribed sphere radius of a tetrahedron from squared distances.

    Uses the relation r = 3V / A where V is the volume and A is the total
    surface area.  Surface area is computed by summing the areas of the four
    triangular faces using Heron's formula. Volume and face areas are
    evaluated with dimensionless squared distances before restoring length
    units; dimensional volumes or areas need not fit in float64.

    For degenerate tetrahedra (zero volume) returns 0.0.

    Parameters
    - d2: 4x4 ndarray of squared distances between vertices.

    Returns
    - float: Non-negative inradius in Euclidean units.
    """
    normalized, scale = _normalized_distances(d2)
    V = _normalized_volume(normalized)
    if V <= 0.0:
        return 0.0

    # Compute surface area via Heron's formula for each face
    # Faces are (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    total_area = 0.0
    for i, j, k in faces:
        a = np.sqrt(max(0.0, normalized[i, j]))
        b = np.sqrt(max(0.0, normalized[i, k]))
        c = np.sqrt(max(0.0, normalized[j, k]))
        s = (a + b + c) / 2.0
        area2 = s * (s - a) * (s - b) * (s - c)
        total_area += np.sqrt(max(0.0, area2))

    # Note: if V > 0, at least one face must have positive area,
    # so total_area > 0 is guaranteed here.
    return _restore_units(float(3.0 * V / total_area), scale, 1)
