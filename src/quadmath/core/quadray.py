from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass
from typing import Iterable, Tuple
from quadmath.core.linalg_utils import bareiss_determinant_int


@dataclass(frozen=True)
class Quadray:
    """Quadray vector with non-negative components and at least one zero (Fuller.4D).

    Utilities to support examples discussed in the paper:
    - normalization via adding/subtracting (k,k,k,k)
    - vector conversion with a configurable 3x4 embedding matrix (slice to Coxeter.4D/XYZ)
    - integer tetrahedron volume via determinant in IVM units
    """

    a: int
    b: int
    c: int
    d: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (a,b,c,d)."""
        return (self.a, self.b, self.c, self.d)

    def normalize(self) -> "Quadray":
        """Translate by -(k,k,k,k) so at least one component is zero.

        This selects the canonical representative on the equivalence class
        q ~ q + t(1,1,1,1), t ∈ Z.
        """
        k = min(self.a, self.b, self.c, self.d)
        return Quadray(self.a - k, self.b - k, self.c - k, self.d - k)

    def add(self, other: "Quadray") -> "Quadray":
        """Component-wise addition."""
        return Quadray(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def sub(self, other: "Quadray") -> "Quadray":
        """Component-wise subtraction."""
        return Quadray(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)


def to_xyz(q: Quadray, embedding: Iterable[Iterable[float]]) -> Tuple[float, float, float]:
    """Map quadray to R^3 via a 3x4 embedding matrix (Fuller.4D -> Coxeter.4D slice).

    embedding rows r0,r1,r2; columns correspond to (a,b,c,d)
    """
    a, b, c, d = q.as_tuple()
    r0, r1, r2 = embedding
    x = r0[0] * a + r0[1] * b + r0[2] * c + r0[3] * d
    y = r1[0] * a + r1[1] * b + r1[2] * c + r1[3] * d
    z = r2[0] * a + r2[1] * b + r2[2] * c + r2[3] * d
    return (x, y, z)


def integer_tetra_volume(p0: Quadray, p1: Quadray, p2: Quadray, p3: Quadray) -> Fraction:
    """Compute the exact IVM tetra-volume of a lattice tetrahedron (Fuller.4D).

    V_ivm = |det[P1-P0, P2-P0, P3-P0]| / 4, computed with the exact integer
    Bareiss determinant on the (a-d, b-d, c-d) projection of the edge vectors
    and returned as an exact :class:`fractions.Fraction`.

    The unit IVM tetrahedron (origin plus three IVM neighbor moves, i.e.
    permutations of (2,1,1,0)) has determinant 4 and volume exactly 1.
    The primitive tetrahedron spanned by (0,0,0,0), (1,0,0,0), (0,1,0,0),
    (0,0,1,0) has determinant 1 and volume exactly 1/4.  General lattice
    tetrahedra may have non-integral volume, so no integrality is assumed.
    """
    v1 = p1.sub(p0)
    v2 = p2.sub(p0)
    v3 = p3.sub(p0)

    def project(q: Quadray) -> Tuple[int, int, int]:
        return (q.a - q.d, q.b - q.d, q.c - q.d)

    M = [list(project(v1)), list(project(v2)), list(project(v3))]
    det = bareiss_determinant_int(M)
    return Fraction(abs(det), 4)


def ace_tetravolume_5x5(p0: Quadray, p1: Quadray, p2: Quadray, p3: Quadray) -> Fraction:
    """Tom Ace 5x5 determinant as the exact IVM tetra-volume (Fuller.4D).

    V_ivm = |det(A)| / 4, with
    A = [[a b c d 1], ... for four vertices; last row [1 1 1 1 0]].
    Uses the exact integer Bareiss determinant and returns an exact
    :class:`fractions.Fraction` (unit IVM tetra = 1, primitive tetra = 1/4).

    For integer quadray vertices, |det(A)| always equals the magnitude of the
    3x3 projected determinant used by :func:`integer_tetra_volume`, so the two
    functions agree exactly on every input (verified on randomized cases).
    """
    A = [
        [p0.a, p0.b, p0.c, p0.d, 1],
        [p1.a, p1.b, p1.c, p1.d, 1],
        [p2.a, p2.b, p2.c, p2.d, 1],
        [p3.a, p3.b, p3.c, p3.d, 1],
        [1, 1, 1, 1, 0],
    ]
    det = bareiss_determinant_int(A)
    return Fraction(abs(det), 4)


DEFAULT_EMBEDDING: Tuple[Tuple[float, float, float, float], ...] = (
    # Simple symmetric embedding of A,B,C,D directions to tetrahedron vertices in R^3
    (1.0, -1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0, -1.0),
    (1.0, -1.0, 1.0, -1.0),
)


def _xyz_tuple(embedding: Iterable[Iterable[float]]) -> Tuple[Tuple[float, float, float, float], ...]:
    """Internal helper to enforce shape on an embedding."""
    r0, r1, r2 = embedding  # type: ignore[misc]
    return (tuple(r0), tuple(r1), tuple(r2))  # type: ignore[return-value]


def _to_xyz_array(q: Quadray, embedding: Iterable[Iterable[float]]) -> Tuple[float, float, float]:
    """Internal helper returning XYZ triple for vector ops."""
    return to_xyz(q, _xyz_tuple(embedding))


def magnitude(q: Quadray, embedding: Iterable[Iterable[float]]) -> float:
    """Return Euclidean magnitude ||q|| under the given embedding (vector norm).

    Parameters
    - q: Quadray vector to measure
    - embedding: 3x4 matrix mapping Fuller.4D to a Coxeter.4D/XYZ slice

    Returns
    - float: Euclidean norm of the embedded vector
    """
    x, y, z = _to_xyz_array(q, embedding)
    return float((x * x + y * y + z * z) ** 0.5)


def dot(q1: Quadray, q2: Quadray, embedding: Iterable[Iterable[float]]) -> float:
    """Return Euclidean dot product <q1,q2> under the given embedding.

    Parameters
    - q1, q2: Quadray vectors
    - embedding: 3x4 matrix mapping Fuller.4D to a Coxeter.4D/XYZ slice

    Returns
    - float: Dot product in the embedded Euclidean space
    """
    x1, y1, z1 = _to_xyz_array(q1, embedding)
    x2, y2, z2 = _to_xyz_array(q2, embedding)
    return float(x1 * x2 + y1 * y2 + z1 * z2)


def distance(q1: Quadray, q2: Quadray, embedding: Iterable[Iterable[float]]) -> float:
    """Euclidean distance between two quadray points under the given embedding.

    Parameters
    - q1, q2: Quadray points (Fuller.4D)
    - embedding: 3x4 matrix mapping Fuller.4D -> Coxeter.4D/XYZ slice

    Returns
    - float: Non-negative Euclidean distance in R^3
    """
    x1, y1, z1 = _to_xyz_array(q1, embedding)
    x2, y2, z2 = _to_xyz_array(q2, embedding)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def angle(q1: Quadray, q2: Quadray, q3: Quadray,
          embedding: Iterable[Iterable[float]]) -> float:
    """Angle at vertex q2 formed by rays q2->q1 and q2->q3 (radians).

    Uses the embedded dot-product formula:
        cos(theta) = <v1, v2> / (|v1| |v2|)
    where v1 = q1 - q2 and v2 = q3 - q2 in R^3.

    Parameters
    - q1, q2, q3: Quadray points; angle is measured at q2
    - embedding: 3x4 matrix mapping Fuller.4D -> Coxeter.4D/XYZ slice

    Returns
    - float: Angle in radians in [0, pi]

    Raises
    - ValueError: If q1==q2 or q3==q2 (degenerate angle)
    """
    ax, ay, az = _to_xyz_array(q1, embedding)
    bx, by, bz = _to_xyz_array(q2, embedding)
    cx, cy, cz = _to_xyz_array(q3, embedding)
    v1x, v1y, v1z = ax - bx, ay - by, az - bz
    v2x, v2y, v2z = cx - bx, cy - by, cz - bz
    len1 = (v1x * v1x + v1y * v1y + v1z * v1z) ** 0.5
    len2 = (v2x * v2x + v2y * v2y + v2z * v2z) ** 0.5
    if len1 == 0.0 or len2 == 0.0:
        raise ValueError("Degenerate angle: vertex coincides with an endpoint")
    cos_theta = (v1x * v2x + v1y * v2y + v1z * v2z) / (len1 * len2)
    # Clamp for numerical safety
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.acos(cos_theta))


def centroid(*quads: Quadray) -> Quadray:
    """Component-wise mean of quadray points, rounded to the nearest lattice point.

    Computes the arithmetic mean of each component (a, b, c, d) across all
    input quadrays, rounds to the nearest integer, and normalizes.

    Parameters
    - *quads: Two or more Quadray points

    Returns
    - Quadray: Normalized lattice-point centroid

    Raises
    - ValueError: If fewer than one quadray is provided
    """
    if len(quads) == 0:
        raise ValueError("At least one Quadray is required")
    n = len(quads)
    sa = sum(q.a for q in quads)
    sb = sum(q.b for q in quads)
    sc = sum(q.c for q in quads)
    sd = sum(q.d for q in quads)
    return Quadray(
        round(sa / n), round(sb / n), round(sc / n), round(sd / n)
    ).normalize()


def quadray_from_xyz(
    x: float, y: float, z: float,
    embedding: Iterable[Iterable[float]],
) -> Quadray:
    """Map an R^3 point back to the quadray lattice via pseudoinverse rounding.

    Computes the pseudoinverse of the 3x4 embedding matrix to find the
    min-norm real-valued quadray coordinates, rounds half up to integers, and
    normalizes. For points that lie exactly on the lattice this round-trips
    exactly (the rounded point stays in the preimage's (1,1,1,1)-coset). For
    general R^3 points the result is the nearest lattice point in quadray
    coordinates (component-wise), which is not always the nearest in embedded
    XYZ distance.

    Parameters
    - x, y, z: Cartesian coordinates (Coxeter.4D)
    - embedding: 3x4 matrix used in to_xyz (Fuller.4D -> Coxeter.4D)

    Returns
    - Quadray: Nearest normalized integer lattice point
    """
    import numpy as np
    rows = list(embedding)
    M = np.array([list(r) for r in rows], dtype=float)  # shape (3, 4)
    xyz = np.array([x, y, z], dtype=float)
    # Pseudoinverse: M^+ = M^T (M M^T)^{-1}
    q_real = M.T @ np.linalg.solve(M @ M.T, xyz)
    # Round half up (floor(v + 0.5)), NOT banker's rounding: when the exact
    # preimage q has sum(q) % 4 == 2, the projection lands exactly on
    # half-integer ties in all four components. Per-component round-half-even
    # breaks ties inconsistently and can leave the (1,1,1,1)-coset of q, so
    # the rounded point maps to a different XYZ location (edge-length errors).
    # floor(v + 0.5) provably stays in q's coset, making lattice round-trips
    # exact.
    q_int = [int(math.floor(v + 0.5)) for v in q_real]
    return Quadray(q_int[0], q_int[1], q_int[2], q_int[3]).normalize()


__all__ = [
    "Quadray",
    "to_xyz",
    "integer_tetra_volume",
    "ace_tetravolume_5x5",
    "DEFAULT_EMBEDDING",
    "magnitude",
    "dot",
    "distance",
    "angle",
    "centroid",
    "quadray_from_xyz",
]
