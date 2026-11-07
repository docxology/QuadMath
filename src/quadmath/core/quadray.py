from __future__ import annotations

import math
import numbers
from fractions import Fraction
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Tuple
from quadmath.core.linalg_utils import bareiss_determinant_int


@dataclass(frozen=True)
class Quadray:
    """Four-component quadray vector; ``normalize`` chooses its min-zero gauge.

    Utilities to support examples discussed in the paper:
    - normalization via adding/subtracting (k,k,k,k)
    - vector conversion with a configurable 3x4 embedding matrix (slice to Coxeter.4D/XYZ)
    - integer tetrahedron volume via determinant in IVM units
    """

    a: int
    b: int
    c: int
    d: int

    def __post_init__(self) -> None:
        """Store integral and rational scalar fields with unbounded Python integers."""
        for name in ("a", "b", "c", "d"):
            value = getattr(self, name)
            if type(value) is not int and isinstance(value, numbers.Integral):
                object.__setattr__(self, name, int(value))
            elif isinstance(value, Fraction):
                normalized = _exact_fraction(value)
                if normalized is not value:
                    object.__setattr__(self, name, normalized)

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


def _exact_fraction(value: float) -> Fraction:
    """Retain integral/rational values and exact binary floating coefficients."""
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        if type(value.numerator) is int and type(value.denominator) is int:
            return value
        return Fraction(int(value.numerator), int(value.denominator))
    if isinstance(value, numbers.Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    return Fraction(float(value))


def _row_sum_is_zero(row: Iterable[float]) -> bool:
    """Test exact gauge compatibility without overflow or cancellation."""
    return sum((_exact_fraction(value) for value in row), Fraction()) == 0


def _embed_row(row: Tuple[float, ...], values: Tuple[int, ...]) -> float:
    """Evaluate a row, retaining scaled integers too large for float coercion."""
    try:
        return row[0] * values[0] + row[1] * values[1] + row[2] * values[2] + row[3] * values[3]
    except OverflowError:
        return float(sum((_exact_fraction(weight) * _exact_fraction(value)
                          for weight, value in zip(row, values)), Fraction()))


def to_xyz(q: Quadray, embedding: Iterable[Iterable[float]]) -> Tuple[float, float, float]:
    """Map quadray to R^3 via a 3x4 embedding matrix (Fuller.4D -> Coxeter.4D slice).

    Rows correspond to X,Y,Z and columns to a,b,c,d. If every row sums to
    exactly zero, remove the common integer gauge before any floating-point
    multiplication. Arbitrarily large gauge aliases then have the same
    image without converting their irrelevant common offset to float.
    Other embeddings retain their ordinary, gauge-dependent linear map.
    """
    r0, r1, r2 = _xyz_tuple(embedding)
    if all(_row_sum_is_zero(row) for row in (r0, r1, r2)):
        q = q.normalize()
    values = q.as_tuple()
    return (_embed_row(r0, values), _embed_row(r1, values), _embed_row(r2, values))


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


#: The 12 IVM neighbor moves: distinct permutations of (2, 1, 1, 0), sorted.
#: Single source for the lattice modules and the core neighbor examples.
IVM_NEIGHBOR_MOVES: Tuple[Tuple[int, int, int, int], ...] = tuple(
    sorted(set(permutations((2, 1, 1, 0))))
)


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
    return to_xyz(q, embedding)


def magnitude(q: Quadray, embedding: Iterable[Iterable[float]]) -> float:
    """Return the Euclidean magnitude of ``q`` under the given embedding (vector norm).

    Parameters
    - q: Quadray vector to measure
    - embedding: 3x4 matrix mapping Fuller.4D to a Coxeter.4D/XYZ slice

    Returns
    - float: Euclidean norm of the embedded vector
    """
    x, y, z = _to_xyz_array(q, embedding)
    return math.hypot(x, y, z)


def dot(q1: Quadray, q2: Quadray, embedding: Iterable[Iterable[float]]) -> float:
    """Return Euclidean dot product <q1,q2> under the given embedding.

    Parameters
    - q1, q2: Quadray vectors
    - embedding: 3x4 matrix mapping Fuller.4D to a Coxeter.4D/XYZ slice

    Returns
    - float: Dot product in the embedded Euclidean space
    """
    embedding = _xyz_tuple(embedding)
    x1, y1, z1 = _to_xyz_array(q1, embedding)
    x2, y2, z2 = _to_xyz_array(q2, embedding)
    return float(x1 * x2 + y1 * y2 + z1 * z2)


def distance(q1: Quadray, q2: Quadray, embedding: Iterable[Iterable[float]]) -> float:
    """Euclidean distance between two quadray points under the given embedding.

    Parameters
    - q1, q2: Quadray points (Fuller.4D)
    - embedding: 3x4 matrix mapping Fuller.4D -> Coxeter.4D/XYZ slice

    Returns
    - float: Non-negative Euclidean distance in R^3. Integer displacement is
      formed before embedding, preserving large spatial translations.
    """
    dx, dy, dz = _to_xyz_array(q2.sub(q1), embedding)
    return math.hypot(dx, dy, dz)


def angle(q1: Quadray, q2: Quadray, q3: Quadray,
          embedding: Iterable[Iterable[float]]) -> float:
    """Angle at vertex q2 formed by rays q2->q1 and q2->q3 (radians).

    Uses the embedded dot-product formula:
        cos(theta) = <v1, v2> / (|v1| |v2|)
    where v1 = q1 - q2 and v2 = q3 - q2 are formed before embedding to
    preserve integer displacements under large spatial translations.

    Parameters
    - q1, q2, q3: Quadray points; angle is measured at q2
    - embedding: 3x4 matrix mapping Fuller.4D -> Coxeter.4D/XYZ slice

    Returns
    - float: Angle in radians in [0, pi]

    Raises
    - ValueError: If q1==q2 or q3==q2 (degenerate angle)
    """
    embedding = _xyz_tuple(embedding)
    v1x, v1y, v1z = _to_xyz_array(q1.sub(q2), embedding)
    v2x, v2y, v2z = _to_xyz_array(q3.sub(q2), embedding)
    scale1 = max(abs(v1x), abs(v1y), abs(v1z))
    scale2 = max(abs(v2x), abs(v2y), abs(v2z))
    if scale1 == 0.0 or scale2 == 0.0:
        raise ValueError("Degenerate angle: vertex coincides with an endpoint")
    v1x, v1y, v1z = v1x / scale1, v1y / scale1, v1z / scale1
    v2x, v2y, v2z = v2x / scale2, v2y / scale2, v2z / scale2
    len1 = math.hypot(v1x, v1y, v1z)
    len2 = math.hypot(v2x, v2y, v2z)
    cos_theta = (v1x * v2x + v1y * v2y + v1z * v2z) / (len1 * len2)
    # Clamp for numerical safety
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.acos(cos_theta))


def _round_half_up(v: float) -> int:
    """Round to the nearest integer, ties toward +infinity, without adding 0.5."""
    lower = math.floor(v)
    return lower + int(v - lower >= 0.5)


def centroid(*quads: Quadray) -> Quadray:
    """Round the component-wise mean of canonical representatives, exactly.

    Each input is normalized independently before integer accumulation, so
    replacing any point by q + t*(1,1,1,1) leaves the result unchanged.
    Component means are rounded half up using integer division, then the
    result is normalized. This is a discrete component-wise centroid, not
    necessarily the XYZ-nearest lattice point to the Euclidean centroid.
    No float conversion occurs, even for arbitrarily large integer inputs.

    Parameters
    - *quads: One or more integer Quadray points, in any gauges

    Returns
    - Quadray: Normalized lattice-point centroid

    Raises
    - ValueError: If fewer than one quadray is provided
    """
    if len(quads) == 0:
        raise ValueError("At least one Quadray is required")
    n = len(quads)
    totals = [0, 0, 0, 0]
    for q in quads:
        components = tuple(int(value) for value in q.as_tuple())
        minimum = min(components)
        for i, value in enumerate(components):
            totals[i] += value - minimum
    return Quadray(*((2 * total + n) // (2 * n) for total in totals)).normalize()


def centroid_exact(points: Iterable[Quadray]) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return the exact vertex mean in the canonical minimum-zero gauge.

    Parameters
    ----------
    points : Iterable[Quadray]
        One or more points, in arbitrary independent gauges.

    Returns
    -------
    tuple[Fraction, Fraction, Fraction, Fraction]
        Rational components, without lattice snapping or float conversion.

    Raises
    ------
    ValueError
        If the iterable is empty.
    """
    totals = [Fraction(0) for _ in range(4)]
    count = 0
    for point in points:
        components = tuple(_exact_fraction(value) for value in point.as_tuple())
        for i, value in enumerate(components):
            totals[i] += value
        count += 1
    if count == 0:
        raise ValueError("At least one Quadray is required")
    minimum = min(totals)
    a, b, c, d = ((total - minimum) / count for total in totals)
    return a, b, c, d


def quadray_from_xyz(
    x: float, y: float, z: float,
    embedding: Iterable[Iterable[float]],
) -> Quadray:
    """Map an R^3 point back by minimum-norm, component-wise half-up rounding.

    The embedding and XYZ are scaled together before solving, avoiding
    overflow/underflow from normal equations and their squared condition
    number. For gauge-compatible rank-three embeddings, the minimum-norm
    preimage is the unique solution with component sum zero.

    Integer-image reconstruction keeps half-integer ties in the same gauge
    class despite solve roundoff. A reconstructed class is accepted only
    when the actual forward map reproduces the input XYZ floats exactly;
    magnitude-only error estimates cannot certify off-lattice points.
    This is not XYZ-nearest projection; off-lattice points retain the
    component-wise rule. Float round-trips require finite intermediate
    coordinates and enough precision to distinguish the lattice points;
    use the exact rational inverse for strict image membership.

    Parameters
    - x, y, z: Cartesian coordinates (Coxeter.4D)
    - embedding: 3x4 matrix used in to_xyz (Fuller.4D -> Coxeter.4D)

    Returns
    - Quadray: Normalized integer point from component-wise half-up rounding
      (not necessarily the XYZ-nearest lattice point)
    """
    import numpy as np

    M = np.array([list(row) for row in embedding], dtype=float)
    scale = float(np.max(np.abs(M)))
    if scale == 0.0:
        raise np.linalg.LinAlgError("Singular embedding")
    scaled = M / scale
    xyz = np.array([x, y, z], dtype=float) / scale
    gauge_compatible = all(_row_sum_is_zero(row) for row in M)
    if gauge_compatible:
        # ker(M) = span(1,1,1,1): sum(q)=0 selects the minimum-norm
        # solution. Solve this square system directly, not M @ M.T.
        system = np.vstack((scaled, np.ones(4)))
        q_real = np.linalg.solve(system, np.append(xyz, 0.0))

        # Differences of lattice preimages are integers, regardless of the
        # shared fractional part (0, 1/4, 1/2 or 3/4 at Urner scales).
        minimum = float(np.min(q_real))
        candidate = Quadray(*(_round_half_up(float(v) - minimum) for v in q_real))
        # Replay the public forward arithmetic, rather than admitting a
        # magnitude-sized residual. A large common Cartesian component must
        # not hide a resolved off-lattice displacement in another component.
        if to_xyz(candidate, M) == (float(x), float(y), float(z)):
            return candidate
    else:
        # Preserve the minimum-norm behavior for full-rank embeddings whose
        # kernel is not the quadray gauge, without forming normal equations.
        q_real, _, rank, _ = np.linalg.lstsq(scaled, xyz, rcond=None)
        if rank < 3:
            raise np.linalg.LinAlgError("Singular embedding")
    return Quadray(*(_round_half_up(float(v)) for v in q_real)).normalize()


# --------------- Quaternion helpers ---------------
#
# Component order convention: quaternions are (w, x, y, z) sequences,
# scalar first: q = w + x*i + y*j + z*k.  This is an independent
# representation from the Quadray lattice components (a, b, c, d) and from
# the XYZ triples returned by to_xyz.

_QUAT_UNIT_TOL = 1e-9


def qmul(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float, float, float]:
    """Hamilton product of two quaternions.

    Component order convention: inputs and output are scalar-first
    (w, x, y, z), i.e. q = w + x*i + y*j + z*k.  This is independent of the
    Quadray lattice storage (a, b, c, d) and of the XYZ triples produced by
    to_xyz.

    Parameters
    - a, b: Quaternions as (w, x, y, z) sequences of four numbers

    Returns
    - Tuple[float, float, float, float]: Hamilton product a*b in (w, x, y, z) order
    """
    aw, ax, ay, az = (float(c) for c in a)
    bw, bx, by, bz = (float(c) for c in b)
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qconjugate(q: Iterable[float]) -> Tuple[float, float, float, float]:
    """Conjugate (w, -x, -y, -z) of a quaternion in (w, x, y, z) order.

    For a unit quaternion the conjugate is the multiplicative inverse, so
    q * qconjugate(q) == (1, 0, 0, 0) up to floating-point error.

    Parameters
    - q: Quaternion as (w, x, y, z) sequence of four numbers

    Returns
    - Tuple[float, float, float, float]: (w, -x, -y, -z)
    """
    w, x, y, z = (float(c) for c in q)
    return (w, -x, -y, -z)


def qrotate(q: Iterable[float], v_xyz: Iterable[float],
            angle: float) -> Tuple[float, float, float]:
    """Rotate a 3-vector by a unit quaternion via Rodrigues (v' = q v q*).

    The rotated vector is computed as q * (0, v) * qconjugate(q) using
    :func:`qmul` and :func:`qconjugate`.  ``q`` and ``-q`` encode the same
    rotation and are both accepted.  The rotation axis is not an input, so
    ``angle`` is validated only up to sign: its magnitude, folded into
    [0, pi] modulo 2*pi, must equal the rotation angle encoded by q,
    2*atan2(||(x, y, z)||, |w|), within 1e-9.  A mismatched (q, angle) pair
    raises instead of silently rotating by the wrong amount.

    Parameters
    - q: Unit quaternion (w, x, y, z) encoding the rotation
    - v_xyz: 3-vector (x, y, z) to rotate
    - angle: Rotation angle in radians that q is asserted to represent

    Returns
    - Tuple[float, float, float]: Rotated vector (x, y, z), same
      convention as to_xyz

    Raises
    - ValueError: If q is not a unit quaternion (|q| = 1 within 1e-9) or
      the rotation magnitude it encodes differs from |angle| (modulo 2*pi,
      tolerance 1e-9)
    """
    qw, qx, qy, qz = (float(c) for c in q)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if abs(n - 1.0) > _QUAT_UNIT_TOL:
        raise ValueError("q must be a unit quaternion (|q| = 1 within 1e-9)")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    two_pi = 2.0 * math.pi
    theta_q = 2.0 * math.atan2(math.sqrt(qx * qx + qy * qy + qz * qz), abs(qw))
    folded = abs(angle) % two_pi
    expected = min(folded, two_pi - folded)
    if abs(theta_q - expected) > _QUAT_UNIT_TOL:
        raise ValueError(
            "angle does not match the rotation encoded by q "
            "(compared modulo 2*pi within 1e-9)"
        )
    qn = (qw, qx, qy, qz)
    rotated = qmul(qmul(qn, (0.0, float(v_xyz[0]), float(v_xyz[1]), float(v_xyz[2]))),
                   qconjugate(qn))
    return (rotated[1], rotated[2], rotated[3])


def slerp(qa: Iterable[float], qb: Iterable[float],
          t: float) -> Tuple[float, float, float, float]:
    """Shortest-arc spherical linear interpolation between unit quaternions.

    If <qa, qb> < 0 the second quaternion is negated first (q and -q
    represent the same rotation), so the path always takes the shorter arc
    on the rotation sphere.  For exactly antipodal inputs (qb == -qa) the
    rotation axis is undefined; after sign alignment the endpoints coincide
    and the result is qa for every t.  Endpoints are exact: slerp(qa, qb, 0)
    returns qa and slerp(qa, qb, 1) returns the sign-aligned qb.

    Parameters
    - qa, qb: Unit quaternions (w, x, y, z)
    - t: Interpolation parameter in [0, 1]

    Returns
    - Tuple[float, float, float, float]: Interpolated unit quaternion (w, x, y, z)

    Raises
    - ValueError: If qa or qb is not a unit quaternion (|q| = 1 within
      1e-9), or t lies outside [0, 1]
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must lie in [0, 1]")
    qa4 = tuple(float(c) for c in qa)
    qb4 = tuple(float(c) for c in qb)
    na = math.sqrt(sum(c * c for c in qa4))
    nb = math.sqrt(sum(c * c for c in qb4))
    if abs(na - 1.0) > _QUAT_UNIT_TOL or abs(nb - 1.0) > _QUAT_UNIT_TOL:
        raise ValueError("qa and qb must be unit quaternions (|q| = 1 within 1e-9)")
    dot = sum(a * b for a, b in zip(qa4, qb4))
    if dot < 0.0:
        qb4 = tuple(-c for c in qb4)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 1.0 - 1e-12:
        # Nearly parallel (includes exactly antipodal after sign alignment):
        # fall back to normalized lerp to avoid dividing by sin(theta) ~ 0.
        blended = tuple((1.0 - t) * a + t * b for a, b in zip(qa4, qb4))
        norm = math.sqrt(sum(c * c for c in blended))
        return tuple(c / norm for c in blended)
    theta = math.acos(dot)
    wa = math.sin((1.0 - t) * theta) / math.sin(theta)
    wb = math.sin(t * theta) / math.sin(theta)
    return tuple(wa * a + wb * b for a, b in zip(qa4, qb4))


def rotate_about_axis(v_xyz: Iterable[float], axis_xyz: Iterable[float],
                      angle: float) -> Tuple[float, float, float]:
    """Rotate a 3-vector about an axis by an angle (axis-angle convenience).

    Builds the unit quaternion q = (cos(angle/2), sin(angle/2) * axis_hat)
    from the normalized axis and delegates to :func:`qrotate`.  The axis is
    normalized internally, so any non-zero scale of the direction is
    accepted.  Rotation is right-handed about the axis.  The angle is
    reduced to (-pi, pi] first — rotation by ``angle`` is identical to
    rotation by the reduced angle, and the reduction keeps
    :func:`qrotate`'s encoded-angle validation exact.

    Parameters
    - v_xyz: 3-vector (x, y, z) to rotate
    - axis_xyz: Non-zero rotation axis (any scale; normalized internally)
    - angle: Rotation angle in radians

    Returns
    - Tuple[float, float, float]: Rotated vector (x, y, z), same
      convention as to_xyz

    Raises
    - ValueError: If axis_xyz is the zero vector
    """
    ax, ay, az = (float(c) for c in axis_xyz)
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n == 0.0:
        raise ValueError("axis must be a non-zero vector")
    angle = math.atan2(math.sin(angle), math.cos(angle))
    half = 0.5 * angle
    sin_half = math.sin(half)
    q = (math.cos(half), sin_half * ax / n, sin_half * ay / n, sin_half * az / n)
    return qrotate(q, v_xyz, angle)


__all__ = [
    "Quadray",
    "to_xyz",
    "integer_tetra_volume",
    "ace_tetravolume_5x5",
    "DEFAULT_EMBEDDING",
    "IVM_NEIGHBOR_MOVES",
    "magnitude",
    "dot",
    "distance",
    "angle",
    "centroid",
    "centroid_exact",
    "quadray_from_xyz",
    "qmul",
    "qconjugate",
    "qrotate",
    "slerp",
    "rotate_about_axis",
]
