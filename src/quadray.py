from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
from linalg_utils import bareiss_determinant_int

from linalg_utils import bareiss_determinant_int


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


def integer_tetra_volume(p0: Quadray, p1: Quadray, p2: Quadray, p3: Quadray) -> int:
    """Compute integer tetra-volume using det[p1-p0, p2-p0, p3-p0] (Fuller.4D).

    Returns the determinant magnitude normalized to synergetics units
    (unit IVM tetra = 1). If the determinant is divisible by 4, divide by 4.
    """
    v1 = p1.sub(p0)
    v2 = p2.sub(p0)
    v3 = p3.sub(p0)

    def project(q: Quadray) -> Tuple[int, int, int]:
        return (q.a - q.d, q.b - q.d, q.c - q.d)

    M = [list(project(v1)), list(project(v2)), list(project(v3))]
    det = bareiss_determinant_int(M)
    abs_det = abs(det)
    return abs_det // 4 if abs_det % 4 == 0 else abs_det


def ace_tetravolume_5x5(p0: Quadray, p1: Quadray, p2: Quadray, p3: Quadray) -> int:
    """Tom Ace 5x5 determinant in IVM units (Fuller.4D).

    V_ivm = |det(A)| / 4, with
    A = [[a b c d 1], ... for four vertices; last row [1 1 1 1 0]].
    Uses exact integer Bareiss determinant. Returns an integer.
    """
    A = [
        [p0.a, p0.b, p0.c, p0.d, 1],
        [p1.a, p1.b, p1.c, p1.d, 1],
        [p2.a, p2.b, p2.c, p2.d, 1],
        [p3.a, p3.b, p3.c, p3.d, 1],
        [1, 1, 1, 1, 0],
    ]
    det = bareiss_determinant_int(A)
    v = abs(det)
    return v // 4


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


__all__ = [
    "Quadray",
    "to_xyz",
    "integer_tetra_volume",
    "ace_tetravolume_5x5",
    "DEFAULT_EMBEDDING",
    "magnitude",
    "dot",
]
