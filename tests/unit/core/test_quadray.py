from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from quadmath.core.quadray import (
    IVM_NEIGHBOR_MOVES,
    Quadray,
    integer_tetra_volume,
    to_xyz,
    DEFAULT_EMBEDDING,
    ace_tetravolume_5x5,
    magnitude,
    dot,
    distance,
    dot_exact,
    squared_norm_exact,
    squared_distance_exact,
    affine_combination_exact,
    signed_tetra_volume_exact,
    triangle_area_squared_exact,
    angle,
    centroid,
    centroid_exact,
    quadray_from_xyz,
    qmul,
    qconjugate,
    qrotate,
    slerp,
    rotate_about_axis,
)
import math


def test_exact_centroid_primitive_tetrahedron_preserves_fractional_mean():
    points = [Quadray(0, 0, 0, 0), Quadray(1, 0, 0, 0),
              Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0)]
    result = centroid_exact(iter(points))
    assert result == (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(0))
    assert all(isinstance(value, Fraction) for value in result)
    assert centroid(*points) == Quadray(0, 0, 0, 0)


@pytest.mark.parametrize("scale", (1, 10**400 + 1))
def test_exact_centroid_canonicalizes_mean_and_independent_gauges(scale):
    rows = [(scale, 0, 0, 0), (0, scale, scale, scale)]
    gauges = [-(10**450), 10**500]
    shifted = [Quadray(*(value + gauge for value in row))
               for row, gauge in zip(rows, gauges)]
    assert centroid_exact(shifted) == (Fraction(0),) * 4
    assert centroid_exact([Quadray(scale, 0, 0, 0), Quadray(0, 0, 0, 0)]) == (
        Fraction(scale, 2), Fraction(0), Fraction(0), Fraction(0))


def test_exact_centroid_accepts_rational_components_without_truncation():
    points = [Quadray(Fraction(1, 3), 0, 0, 0), Quadray(0, 0, 0, 0)]
    assert centroid_exact(points) == (Fraction(1, 6), Fraction(0), Fraction(0), Fraction(0))


@pytest.mark.parametrize("numerator", [1, 10**400 + 1])
def test_exact_centroid_preserves_symbolic_rational_components(numerator):
    from sympy import Rational

    points = [Quadray(Rational(numerator, 3), 0, 0, 0), Quadray(0, 0, 0, 0)]
    assert centroid_exact(points) == (Fraction(numerator, 6), Fraction(0), Fraction(0), Fraction(0))


def test_exact_centroid_rejects_empty_iterable():
    with pytest.raises(ValueError):
        centroid_exact(iter(()))


def test_normalize():
    q = Quadray(3, 5, 3, 5)
    qn = q.normalize()
    assert min(qn.as_tuple()) == 0


def test_volume_unit_tetra():
    # Unit IVM tetrahedron: origin plus three (2,1,1,0)-type neighbor moves
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 1, 1, 0)
    p2 = Quadray(1, 2, 1, 0)
    p3 = Quadray(1, 1, 2, 0)
    v = integer_tetra_volume(p0, p1, p2, p3)
    assert v == 1 and isinstance(v, Fraction)


def test_volume_primitive_tetra_is_one_quarter():
    # Regression for the old divide-by-4-only-if-divisible heuristic, which
    # reported 1 for this tetrahedron (true exact IVM volume is 1/4)
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    assert integer_tetra_volume(p0, p1, p2, p3) == Fraction(1, 4)


def test_ace_tetravolume_matches_integer_tetra_volume():
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 1, 0, 1)
    p2 = Quadray(2, 1, 1, 0)
    p3 = Quadray(2, 0, 1, 1)
    # These form a unit IVM tetrahedron from a common origin
    v1 = integer_tetra_volume(p0, p1, p2, p3)
    v2 = ace_tetravolume_5x5(p0, p1, p2, p3)
    assert v1 == 1 and v2 == 1 and v1 == v2


def test_ace_and_integer_agree_on_non_divisible_determinant():
    # Old behavior disagreed here: integer heuristic returned 1, Ace floored to 0
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    assert ace_tetravolume_5x5(p0, p1, p2, p3) == integer_tetra_volume(p0, p1, p2, p3) == Fraction(1, 4)


def test_volume_exact_scaled_tetra():
    # Determinant 8 => exact volume 8/4 = 2 (no divisibility special-casing)
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 0, 0, 0)
    p2 = Quadray(0, 2, 0, 0)
    p3 = Quadray(0, 0, 2, 0)
    assert integer_tetra_volume(p0, p1, p2, p3) == 2


def test_to_xyz_known_image():
    q = Quadray(2, 1, 1, 0)
    x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
    # Exact image under the default embedding (integral input, +/-1 rows)
    assert (x, y, z) == (0.0, 2.0, 2.0)


@pytest.mark.parametrize("gauge", (10**20, -(10**20), 10**1100, -(10**1100)))
def test_forward_embedding_and_metrics_ignore_arbitrary_size_gauges(gauge):
    q = Quadray(1, 0, 0, 0)
    alias = Quadray(gauge + 1, gauge, gauge, gauge)
    origin_alias = Quadray(-gauge, -gauge, -gauge, -gauge)
    other_alias = Quadray(2 * gauge, 2 * gauge + 1, 2 * gauge, 2 * gauge)
    assert to_xyz(alias, DEFAULT_EMBEDDING) == (1.0, 1.0, 1.0)
    assert to_xyz(origin_alias, DEFAULT_EMBEDDING) == (0.0, 0.0, 0.0)
    assert magnitude(alias, DEFAULT_EMBEDDING) == pytest.approx(math.sqrt(3))
    assert distance(q, alias, DEFAULT_EMBEDDING) == 0.0
    assert distance(alias, other_alias, DEFAULT_EMBEDDING) == pytest.approx(math.sqrt(8))
    assert dot(q, alias, DEFAULT_EMBEDDING) == 3.0
    assert dot(alias, other_alias, DEFAULT_EMBEDDING) == -1.0
    assert angle(alias, origin_alias, other_alias, DEFAULT_EMBEDDING) == pytest.approx(
        math.acos(-1 / 3)
    )
    with pytest.raises(ValueError):
        angle(q, alias, other_alias, DEFAULT_EMBEDDING)


def test_forward_gauge_removal_supports_general_zero_sum_rows():
    embedding = ((2.0, 1.0, -4.0, 1.0), (3.0, -2.0, 1.0, -2.0), (0.0, 1.0, 2.0, -3.0))
    gauge = 10**1100
    alias = Quadray(gauge + 1, gauge, gauge, gauge)
    assert to_xyz(alias, embedding) == (2.0, 3.0, 0.0)


def test_forward_gauge_removal_preserves_exact_fraction_embeddings():
    embedding = (
        (Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(-3, 10)),
        (Fraction(1, 3), Fraction(-1, 3), 0, 0),
        (0, 0, Fraction(1, 7), Fraction(-1, 7)),
    )
    gauge = 10**1100
    alias = Quadray(gauge + 1, gauge, gauge, gauge)
    assert to_xyz(alias, embedding) == (Fraction(1, 10), Fraction(1, 3), Fraction(0))


def test_forward_integer_embedding_rounds_floating_coordinate_combinations():
    embedding = ((1, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    point = Quadray(0.1, 0.2, 0.0, 0.0)
    expected = float(Fraction(0.1) + Fraction(0.2))
    assert to_xyz(point, embedding) == (expected, 0.0, 0.0)


def test_forward_embedding_preserves_genuinely_gauge_dependent_maps():
    embedding = ((1.0, -1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, -1.0))
    assert to_xyz(Quadray(2, 1, 1, 1), embedding) == (1.0, 1.0, 0.0)
    assert to_xyz(Quadray(1, 0, 0, 0), embedding) == (1.0, 0.0, 0.0)
    # Row sums must be checked before any float conversion, too.
    large = 2**60
    integer_embedding = ((large + 1, -large, 0, 0), (0, 1, -1, 0), (0, 0, 1, -1))
    assert to_xyz(Quadray(1, 1, 1, 1), integer_embedding) == (1, 0, 0)


def test_vector_magnitude_and_dot():
    q1 = Quadray(1, 0, 0, 0)
    q2 = Quadray(0, 1, 0, 0)
    m1 = magnitude(q1, DEFAULT_EMBEDDING)
    m2 = magnitude(q2, DEFAULT_EMBEDDING)
    # Quadray basis vectors embed at length sqrt(3) (tetrahedral basis)
    assert math.isclose(m1, math.sqrt(3), rel_tol=1e-12)
    assert math.isclose(m2, math.sqrt(3), rel_tol=1e-12)
    # Distinct quadray basis vectors have inner product -1 (not orthogonal)
    d = dot(q1, q2, DEFAULT_EMBEDDING)
    assert d == -1.0


# --------------- New method tests ---------------


def test_distance_same_point():
    q = Quadray(1, 0, 0, 0)
    assert distance(q, q, DEFAULT_EMBEDDING) == 0.0


def test_distance_positive():
    q1 = Quadray(1, 0, 0, 0)
    q2 = Quadray(0, 1, 0, 0)
    d = distance(q1, q2, DEFAULT_EMBEDDING)
    assert d > 0.0
    # distance should equal |q1 - q2| via embedding
    x1, y1, z1 = to_xyz(q1, DEFAULT_EMBEDDING)
    x2, y2, z2 = to_xyz(q2, DEFAULT_EMBEDDING)
    expected = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** 0.5
    assert abs(d - expected) < 1e-12


def test_distance_symmetric():
    q1 = Quadray(2, 1, 0, 0)
    q2 = Quadray(0, 0, 1, 2)
    assert abs(distance(q1, q2, DEFAULT_EMBEDDING) - distance(q2, q1, DEFAULT_EMBEDDING)) < 1e-12


def test_angle_tetrahedral_between_basis_vectors():
    # Angle at the origin between two quadray basis vectors
    origin = Quadray(0, 0, 0, 0)
    q1 = Quadray(1, 0, 0, 0)
    q2 = Quadray(0, 1, 0, 0)
    a = angle(q1, origin, q2, DEFAULT_EMBEDDING)
    # cos = -1/3 => the tetrahedral bond angle ~109.47 degrees
    assert math.isclose(a, math.acos(-1.0 / 3.0), rel_tol=1e-12)


def test_angle_degenerate():
    q = Quadray(1, 0, 0, 0)
    try:
        angle(q, q, Quadray(0, 1, 0, 0), DEFAULT_EMBEDDING)
        assert False
    except ValueError:
        assert True


def test_angle_straight_line():
    # Opposite directions should give angle close to pi
    origin = Quadray(0, 0, 0, 0)
    q1 = Quadray(2, 0, 0, 0)
    # q2 in opposite direction: map q1 to xyz, negate, find closest quadray
    x1, y1, z1 = to_xyz(q1, DEFAULT_EMBEDDING)
    q2 = quadray_from_xyz(-x1, -y1, -z1, DEFAULT_EMBEDDING)
    a = angle(q1, origin, q2, DEFAULT_EMBEDDING)
    assert a > 2.5  # Should be close to pi


def test_centroid_single_point():
    q = Quadray(2, 1, 0, 0)
    c = centroid(q)
    assert c == q.normalize()


def test_centroid_multiple_points():
    q1 = Quadray(2, 0, 0, 0)
    q2 = Quadray(0, 2, 0, 0)
    c = centroid(q1, q2)
    assert c == Quadray(1, 1, 0, 0)


def test_centroid_empty():
    try:
        centroid()
        assert False
    except ValueError:
        assert True


def test_quadray_from_xyz_roundtrip():
    q_original = Quadray(2, 1, 1, 0)
    x, y, z = to_xyz(q_original, DEFAULT_EMBEDDING)
    q_recovered = quadray_from_xyz(x, y, z, DEFAULT_EMBEDDING)
    # Should recover the same point (normalized)
    assert q_recovered == q_original.normalize()



def test_quadray_from_xyz_roundtrip_half_integer_ties():
    # Regression: preimages with sum(q) % 4 == 2 project onto exact
    # half-integer ties; per-component banker's rounding used to leave the
    # (1,1,1,1)-coset and return a point with a different XYZ image.
    for q_original in (Quadray(1, 1, 0, 0), Quadray(0, 0, 1, 1),
                       Quadray(0, 1, 1, 0), Quadray(0, 0, 3, 3)):
        x, y, z = to_xyz(q_original, DEFAULT_EMBEDDING)
        q_back = quadray_from_xyz(x, y, z, DEFAULT_EMBEDDING)
        # The recovered quadray must map back to the same XYZ point
        assert to_xyz(q_back, DEFAULT_EMBEDDING) == (x, y, z)


@pytest.mark.parametrize(
    "scale",
    (0.1, 1 / 3, -0.1, 1e-300, -1e-300, 1e307, -1e307, 1e-320, -1e-320),
)
def test_quadray_from_xyz_scaled_signed_permutations_all_residues(scale):
    embedding = tuple(tuple(scale * entry for entry in row) for row in DEFAULT_EMBEDDING)
    for residue in range(4):
        for components in sorted(set(permutations((0, 1, 2, 3 + residue)))):
            for sign in (-1, 1):
                signed = tuple(sign * value for value in components)
                q = Quadray(*signed)
                expected = Quadray(*(value - min(signed) for value in signed))
                xyz = to_xyz(q, embedding)
                assert quadray_from_xyz(*xyz, embedding) == expected


@pytest.mark.parametrize("scale", (0.1, -0.1, 1e-200, 1e200))
def test_quadray_from_xyz_does_not_square_embedding_condition_number(scale):
    # The second row differs from the first by about 1e-9. Forming M M.T
    # erases that independent direction, although the embedding is rank three.
    delta = 2.0 ** -30
    rows = ((1, -1, -1, 1), (1, -1 + delta, -1 - delta, 1), (1, -1, 1, -1))
    embedding = tuple(tuple(scale * entry for entry in row) for row in rows)
    for q in (Quadray(1, 1, 0, 0), Quadray(2, 1, 0, 0), Quadray(0, 0, 1, 1)):
        assert quadray_from_xyz(*to_xyz(q, embedding), embedding) == q


def test_quadray_from_xyz_preserves_nongauge_minimum_norm_projection():
    embedding = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
    assert quadray_from_xyz(1.25, -0.5, 2.5, embedding) == Quadray(1, 0, 3, 0)


@pytest.mark.parametrize(
    "embedding",
    (
        ((0, 0, 0, 0),) * 3,
        ((1, -1, 0, 0), (2, -2, 0, 0), (0, 0, 0, 0)),
        ((1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)),
    ),
)
def test_quadray_from_xyz_rejects_singular_embeddings(embedding):
    import numpy as np

    with pytest.raises(np.linalg.LinAlgError):
        quadray_from_xyz(1.0, 2.0, 3.0, embedding)


@pytest.mark.parametrize(
    ("component", "expected"),
    (
        (0.5 - 1e-9, Quadray(0, 0, 0, 0)),
        (0.5, Quadray(1, 0, 0, 0)),
        (0.5 + 1e-9, Quadray(2, 1, 1, 0)),
    ),
)
def test_quadray_from_xyz_off_lattice_half_up_boundaries(component, expected):
    # Minimum-norm preimage is (component, 0, 0, -component).
    assert quadray_from_xyz(0.0, 2 * component, 2 * component, DEFAULT_EMBEDDING) == expected



def test_quadray_from_xyz_origin():
    q = quadray_from_xyz(0.0, 0.0, 0.0, DEFAULT_EMBEDDING)
    assert q == Quadray(0, 0, 0, 0)


def test_quadray_from_xyz_is_componentwise_not_xyz_nearest():
    """Chosen contract (SPEC.md S30): component-wise rounding, not XYZ-nearest."""
    xyz = (-0.25, 0.75, 0.75)

    def sq_dist(q: Quadray) -> float:
        return sum((p - t) ** 2 for p, t in zip(to_xyz(q, DEFAULT_EMBEDDING), xyz))

    got = quadray_from_xyz(*xyz, DEFAULT_EMBEDDING)
    assert got == Quadray(0, 0, 0, 0)
    assert sq_dist(got) == pytest.approx(19.0 / 16.0, abs=1e-12)
    assert sq_dist(Quadray(1, 1, 1, 0)) == pytest.approx(11.0 / 16.0, abs=1e-12)
    assert sq_dist(Quadray(1, 1, 1, 0)) < sq_dist(got)


@pytest.mark.parametrize("scale", (0.1, 1 / 3, -0.1, 1e-300, 1e300, 1e-320))
def test_quadray_from_xyz_scaled_off_lattice_contract(scale):
    embedding = tuple(tuple(scale * entry for entry in row) for row in DEFAULT_EMBEDDING)
    xyz = tuple(scale * value for value in (-0.25, 0.75, 0.75))
    assert quadray_from_xyz(*xyz, embedding) == Quadray(0, 0, 0, 0)


# --------------- Quaternion rotation tests ---------------


def test_qmul_associative():
    # Integer-valued components keep every intermediate exact in fp
    a = (1.0, 2.0, -1.0, 3.0)
    b = (0.0, 1.0, 1.0, 2.0)
    c = (2.0, -1.0, 0.0, 1.0)
    assert qmul(qmul(a, b), c) == qmul(a, qmul(b, c))


def test_qmul_identity():
    q = (0.5, -1.0, 2.0, 3.0)
    ident = (1.0, 0.0, 0.0, 0.0)
    assert qmul(q, ident) == q
    assert qmul(ident, q) == q


def test_qmul_conjugate_inverse():
    q = (0.5, 0.5, 0.5, 0.5)  # unit quaternion
    assert qmul(q, qconjugate(q)) == (1.0, 0.0, 0.0, 0.0)
    assert qmul(qconjugate(q), q) == (1.0, 0.0, 0.0, 0.0)


def test_qrotate_ninety_degrees_about_z():
    qz90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    x, y, z = qrotate(qz90, (1.0, 0.0, 0.0), math.pi / 2.0)
    assert abs(x - 0.0) < 1e-12
    assert abs(y - 1.0) < 1e-12
    assert abs(z - 0.0) < 1e-12


def test_qrotate_preserves_norm_and_angle():
    qz90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.0, 1.0, 1.0)
    r1 = qrotate(qz90, v1, math.pi / 2.0)
    r2 = qrotate(qz90, v2, math.pi / 2.0)
    n1 = math.sqrt(sum(c * c for c in r1))
    n2 = math.sqrt(sum(c * c for c in r2))
    assert abs(n1 - 1.0) < 1e-12
    assert abs(n2 - math.sqrt(2.0)) < 1e-12

    def cos_between(u, w):
        nu = math.sqrt(sum(c * c for c in u))
        nw = math.sqrt(sum(c * c for c in w))
        return (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (nu * nw)

    assert abs(cos_between(r1, r2) - cos_between(v1, v2)) < 1e-12


def test_qrotate_angle_mismatch_raises():
    qz90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    try:
        qrotate(qz90, (1.0, 0.0, 0.0), 1.0)  # not the pi/2 that qz90 encodes
        assert False
    except ValueError:
        assert True


def test_qrotate_non_unit_raises():
    try:
        qrotate((1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0), math.pi / 2.0)
        assert False
    except ValueError:
        assert True


def test_slerp_endpoints_exact():
    qa = (1.0, 0.0, 0.0, 0.0)
    qb = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    assert slerp(qa, qb, 0.0) == qa
    assert slerp(qa, qb, 1.0) == qb


def test_slerp_midpoint():
    qa = (1.0, 0.0, 0.0, 0.0)
    qb = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    mid = slerp(qa, qb, 0.5)
    expected = (math.cos(math.pi / 8.0), 0.0, 0.0, math.sin(math.pi / 8.0))
    for got, want in zip(mid, expected):
        assert abs(got - want) < 1e-12


def test_slerp_negative_dot_negates_second():
    qa = (1.0, 0.0, 0.0, 0.0)
    qb = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    # dot(qa, -qb) < 0: the negated path must land on the direct path
    assert slerp(qa, tuple(-c for c in qb), 0.3) == slerp(qa, qb, 0.3)


def test_slerp_exactly_antipodal_returns_endpoint():
    q = (0.5, 0.5, 0.5, 0.5)
    mid = slerp(q, tuple(-c for c in q), 0.5)
    # After sign alignment the endpoints coincide: result is q for every t
    for got, want in zip(mid, q):
        assert abs(got - want) < 1e-12


def test_slerp_non_unit_raises():
    try:
        slerp((1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.5)
        assert False
    except ValueError:
        assert True
    try:
        slerp((1.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0), 0.5)
        assert False
    except ValueError:
        assert True


def test_slerp_t_out_of_range_raises():
    qa = (1.0, 0.0, 0.0, 0.0)
    qb = (0.0, 1.0, 0.0, 0.0)
    try:
        slerp(qa, qb, -0.1)
        assert False
    except ValueError:
        assert True
    try:
        slerp(qa, qb, 1.1)
        assert False
    except ValueError:
        assert True


def test_rotate_about_axis_known_case():
    x, y, z = rotate_about_axis((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2.0)
    assert abs(x - 0.0) < 1e-12
    assert abs(y - 1.0) < 1e-12
    assert abs(z - 0.0) < 1e-12


def test_rotate_about_axis_normalizes_axis_and_matches_qrotate():
    angle = math.pi / 2.0
    v = (1.0, 0.0, 0.0)
    got = rotate_about_axis(v, (0.0, 0.0, 2.0), angle)  # unnormalized axis
    sin_half = math.sin(angle / 2.0)
    q = (math.cos(angle / 2.0), 0.0, 0.0, sin_half)
    assert got == qrotate(q, v, angle)


def test_rotate_about_axis_zero_axis_raises():
    try:
        rotate_about_axis((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), math.pi / 2.0)
        assert False
    except ValueError:
        assert True


def test_quaternion_helpers_deterministic():
    qz90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    v = (1.0, 2.0, 3.0)
    assert qrotate(qz90, v, math.pi / 2.0) == qrotate(qz90, v, math.pi / 2.0)
    assert slerp((1.0, 0.0, 0.0, 0.0), qz90, 0.25) == slerp((1.0, 0.0, 0.0, 0.0), qz90, 0.25)
    assert rotate_about_axis(v, (0.0, 1.0, 0.0), 0.7) == rotate_about_axis(v, (0.0, 1.0, 0.0), 0.7)


def test_dot_distance_angle_accept_one_shot_iterable_embedding():
    from quadmath.core.quadray import angle, distance

    q1, q2 = Quadray(2, 0, 0, 0), Quadray(0, 1, 0, 0)

    def rows():
        return (tuple(r) for r in DEFAULT_EMBEDDING)

    assert dot(q1, q2, rows()) == pytest.approx(dot(q1, q2, DEFAULT_EMBEDDING))
    assert distance(q1, q2, rows()) == pytest.approx(distance(q1, q2, DEFAULT_EMBEDDING))
    assert angle(q1, Quadray(0, 0, 0, 0), q2, rows()) == pytest.approx(angle(q1, Quadray(0, 0, 0, 0), q2, DEFAULT_EMBEDDING))


@pytest.mark.parametrize("gauges", ((0, 1), (-17, 5), (2**100, -(2**140))))
def test_centroid_rounds_half_up_under_independent_gauges(gauges):
    points = (Quadray(1, 0, 0, 0), Quadray(0, 0, 0, 0))
    shifted = tuple(
        Quadray(*(value + gauge for value in q.as_tuple()))
        for q, gauge in zip(points, gauges)
    )
    assert centroid(*points) == Quadray(1, 0, 0, 0)
    assert centroid(*shifted) == Quadray(1, 0, 0, 0)


def test_centroid_independent_gauges_with_three_distinct_points():
    points = (Quadray(5, 0, 1, 2), Quadray(0, 3, 2, 1), Quadray(2, 1, 0, 4))
    shifted = tuple(
        Quadray(*(value + gauge for value in q.as_tuple()))
        for q, gauge in zip(points, (2**200, -37, -(2**250)))
    )
    assert centroid(*points) == Quadray(1, 0, 0, 1)
    assert centroid(*shifted) == Quadray(1, 0, 0, 1)


@pytest.mark.parametrize("large", (2**53 + 1, 2**200 + 17, 2**1100 + 1))
def test_centroid_preserves_arbitrary_size_integer_identity_and_half_ties(large):
    q = Quadray(large, large + 2, 1, 0)
    assert centroid(q) == q
    assert centroid(q, q, q) == q
    assert centroid(Quadray(large, 1, 0, 0), Quadray(large + 1, 0, 0, 0)) == Quadray(
        large + 1, 1, 0, 0
    )


@pytest.mark.parametrize(
    ("points", "expected"),
    (
        ((Quadray(1, 0, 0, 0),) + (Quadray(0, 0, 0, 0),) * 2, Quadray(0, 0, 0, 0)),
        ((Quadray(1, 0, 0, 0),) * 2 + (Quadray(0, 0, 0, 0),), Quadray(1, 0, 0, 0)),
    ),
)
def test_centroid_exact_division_on_either_side_of_half(points, expected):
    assert centroid(*points) == expected


def test_qrotate_accepts_negated_quaternion_as_same_rotation():
    qz90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    neg = tuple(-c for c in qz90)
    v = (1.0, 2.0, 3.0)
    got = qrotate(neg, v, math.pi / 2.0)
    expected = qrotate(qz90, v, math.pi / 2.0)
    assert all(abs(g - e) < 1e-12 for g, e in zip(got, expected))


def test_ivm_neighbor_moves_are_the_twelve_sorted_permutations_of_2110():
    from itertools import permutations

    assert IVM_NEIGHBOR_MOVES == tuple(sorted(set(permutations((2, 1, 1, 0)))))
    assert len(IVM_NEIGHBOR_MOVES) == 12


def test_centroid_promotes_numpy_integers_before_normalization_and_accumulation():
    import numpy as np

    rows = (
        np.array([2**62, 0, 0, 0], dtype=np.int64),
        np.array([-2**63, 2**63 - 1, 0, -1], dtype=np.int64),
        np.array([2**64 - 1, 0, 1, 2], dtype=np.uint64),
    )
    for row in rows:
        minimum = min(int(value) for value in row)
        expected = Quadray(*(int(value) - minimum for value in row))
        assert centroid(Quadray(*row)) == expected


def test_large_resolved_off_lattice_displacement_keeps_component_rounding():
    large = 2**40
    # Exact minimum-norm components are
    # (large+511/1024, -large+511/1024, -511/1024, -511/1024).
    # A large X/Z must not hide the resolved Y residual of 1/256.
    assert quadray_from_xyz(2 * large, 511 / 256, 2 * large, DEFAULT_EMBEDDING) == Quadray(
        2 * large, 0, large, large
    )


@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
def test_exact_volumes_promote_fixed_width_components(dtype):
    length = 2**22
    points = [
        Quadray(*(dtype(value) for value in row))
        for row in (
            (0, 0, 0, 0), (length, 0, 0, 0),
            (0, length, 0, 0), (0, 0, length, 0),
        )
    ]
    expected = Fraction(2**66, 4)
    assert integer_tetra_volume(*points) == expected
    assert ace_tetravolume_5x5(*points) == expected


@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
def test_exact_volumes_promote_before_edge_and_projection_subtraction(dtype):
    limits = np.iinfo(dtype)
    low, high = int(limits.min), int(limits.max)
    for rows in (
        ((low, 0, 0, 0), (high, 0, 0, 0), (low, 1, 0, 0), (low, 0, 1, 0)),
        ((0, 0, 0, 0), (high, 0, 0, low), (0, 1, 0, 0), (0, 0, 1, 0)),
    ):
        points = [Quadray(*(dtype(value) for value in row)) for row in rows]
        expected = Fraction(high - low, 4)
        assert integer_tetra_volume(*points) == expected
        assert ace_tetravolume_5x5(*points) == expected


@pytest.mark.parametrize("dtype", (int, np.int64))
def test_relative_geometry_preserves_large_spatial_translations(dtype):
    offset = 2**60
    origin, a, b, c = (
        Quadray(*(dtype(value) for value in row))
        for row in (
            (offset, 0, 0, 0), (offset + 1, 0, 0, 0),
            (offset, 1, 0, 0), (offset, 0, 1, 0),
        )
    )
    assert distance(origin, a, DEFAULT_EMBEDDING) == pytest.approx(math.sqrt(3))
    assert angle(a, origin, b, DEFAULT_EMBEDDING) == pytest.approx(math.acos(-1 / 3))
    assert integer_tetra_volume(origin, a, b, c) == Fraction(1, 4)
    assert ace_tetravolume_5x5(origin, a, b, c) == Fraction(1, 4)


def test_relative_geometry_preserves_fractional_vector_components():
    origin = Quadray(0.25, 0.0, 0.0, 0.0)
    a = Quadray(0.75, 0.0, 0.0, 0.0)
    b = Quadray(0.25, 0.5, 0.0, 0.0)
    assert distance(origin, a, DEFAULT_EMBEDDING) == pytest.approx(math.sqrt(3) / 2)
    assert angle(a, origin, b, DEFAULT_EMBEDDING) == pytest.approx(math.acos(-1 / 3))


def test_geometry_accepts_single_use_embedding_rows_and_outer_iterable():
    def embedding():
        return (iter(row) for row in DEFAULT_EMBEDDING)

    origin = Quadray(0, 0, 0, 0)
    a = Quadray(1, 0, 0, 0)
    b = Quadray(0, 1, 0, 0)
    assert to_xyz(a, embedding()) == (1.0, 1.0, 1.0)
    assert distance(origin, a, embedding()) == pytest.approx(math.sqrt(3))
    assert angle(a, origin, b, embedding()) == pytest.approx(math.acos(-1 / 3))


@pytest.mark.parametrize("embedding", (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((1, 0, 0, 0, 0),) * 3,
    ((1, 0, 0, 0),) * 2,
    ((1, 0, 0, 0),) * 4,
    ((1, 0, 0, 0), (0, 1, 0), (0, 0, 1, 0)),
))
def test_embedding_geometry_rejects_wrong_shape_before_linear_combination(embedding):
    point = Quadray(0, 0, 0, 1)
    with pytest.raises(ValueError):
        to_xyz(point, (iter(row) for row in embedding))
    with pytest.raises(ValueError):
        squared_distance_exact(Quadray(0, 0, 0, 0), point,
                               (iter(row) for row in embedding))


def test_rank_deficient_forward_embedding_remains_a_valid_linear_map():
    rows = ((1, -1, 0, 0), (2, -2, 0, 0), (0, 0, 0, 0))
    point = Quadray(2, 1, 0, 0)
    assert to_xyz(point, (iter(row) for row in rows)) == (1, 2, 0)
    assert squared_norm_exact(point, (iter(row) for row in rows)) == 5


@pytest.mark.parametrize("scale", (1e-200, 1e200))
def test_norms_preserve_finite_values_when_squaring_would_underflow_or_overflow(scale):
    embedding = np.asarray(DEFAULT_EMBEDDING) * scale
    expected = math.sqrt(3) * scale
    assert magnitude(Quadray(1, 0, 0, 0), embedding) == pytest.approx(expected, abs=0)
    translation = 2**60
    actual = distance(Quadray(translation + 1, 0, 0, 0),
                      Quadray(translation, 0, 0, 0), embedding)
    assert actual == pytest.approx(expected, abs=0)


@pytest.mark.parametrize("scale", (1e-200, 1e200, 1e308))
def test_angle_is_unit_invariant_for_finite_embedded_components(scale):
    embedding = np.asarray(DEFAULT_EMBEDDING) * scale
    actual = angle(Quadray(1, 0, 0, 0), Quadray(0, 0, 0, 0),
                   Quadray(0, 1, 0, 0), embedding)
    assert actual == pytest.approx(math.acos(-1 / 3))


@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
def test_fixed_width_components_preserve_exact_normalization_addition_and_xyz(dtype):
    limits = np.iinfo(dtype)
    q = Quadray(dtype(limits.min), dtype(limits.max), dtype(limits.min), dtype(limits.min))
    span = int(limits.max) - int(limits.min)
    assert q.normalize().as_tuple() == (0, span, 0, 0)
    assert to_xyz(q, DEFAULT_EMBEDDING) == (-float(span), float(span), -float(span))
    assert Quadray(dtype(limits.max), 0, 0, 0).add(Quadray(1, 0, 0, 0)).a == int(limits.max) + 1


def test_scaled_integer_displacements_can_exceed_float_coercion_range():
    from quadmath.core.cayley_menger import squared_distances_from_quadrays, tetra_volume_cayley_menger

    length = 2**1023
    embedding = np.asarray(DEFAULT_EMBEDDING) * 2.0**-1023
    origin = Quadray(-length, 0, 0, 0)
    a = Quadray(length, 0, 0, 0)
    b = Quadray(-length, length, 0, 0)
    c = Quadray(-length, 0, length, 0)
    assert distance(origin, a, embedding) == pytest.approx(2 * math.sqrt(3))
    assert angle(a, origin, b, embedding) == pytest.approx(math.acos(-1 / 3))
    squared = squared_distances_from_quadrays(origin, a, b, c, embedding)
    assert tetra_volume_cayley_menger(squared) == pytest.approx(4 / 3)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_numpy_backed_fraction_components_preserve_addition_and_centroids(dtype):
    value = dtype(np.iinfo(dtype).max)
    point = Quadray(Fraction(value, 1), 0, 0, 0)
    assert point.add(point).a == Fraction(2 * int(value))
    assert centroid_exact([point, point]) == (Fraction(int(value)), Fraction(0), Fraction(0), Fraction(0))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_numpy_backed_fraction_denominators_do_not_wrap_in_centroids(dtype):
    denominator = dtype(2**62)
    point = Quadray(Fraction(1, denominator), 0, 0, 0)
    origin = Quadray(0, 0, 0, 0)
    assert centroid_exact([point, origin, origin, origin]) == (
        Fraction(1, 2**64), Fraction(0), Fraction(0), Fraction(0),
    )


def test_native_combinations_retain_resolved_cancellation_before_float_conversion():
    n = 2**60
    q, r = Quadray(n + 1, n, 0, 0), Quadray(1, 0, 1, 0)
    assert to_xyz(q, DEFAULT_EMBEDDING) == (1.0, float(2 * n + 1), 1.0)
    assert dot(q, r, DEFAULT_EMBEDDING) == 2.0
    assert dot_exact(q, r) == 2


def test_dot_cancels_unrepresentable_products_before_rounding():
    length = 2**600
    q = Quadray(length, 0, 0, 0)
    r = Quadray(length + 1, length, 0, 2 * length)
    assert dot_exact(q, r) == 3 * length
    assert dot(q, r, DEFAULT_EMBEDDING) == float(3 * length)
    assert dot(q, Quadray(length, length, 0, 2 * length), DEFAULT_EMBEDDING) == 0.0


def test_forward_combinations_rescale_before_native_float_coercion():
    q = Quadray(2**1100, 0, 0, 0)
    embedding = np.asarray(DEFAULT_EMBEDDING) * 2.0**-500
    assert to_xyz(q, embedding) == (2.0**600,) * 3
    assert magnitude(q, embedding) == pytest.approx(math.sqrt(3) * 2.0**600)
    # Each product overflows in ordinary float arithmetic, but the sum is 1.
    embedding = ((1e308, -1e308, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    assert to_xyz(Quadray(2, 2, 1, 0), embedding) == (1.0, 1.0, 0.0)


@pytest.mark.parametrize("sign", (1, -1))
def test_angle_retains_resolved_near_parallel_and_antiparallel_rays(sign):
    n = 2**40
    origin = Quadray(0, 0, 0, 0)
    u, v = Quadray(n, 0, 0, n), Quadray(sign * n + 1, 1, 0, sign * n)
    expected = math.atan2(1, sign * n)
    assert angle(u, origin, v, DEFAULT_EMBEDDING) == pytest.approx(
        expected, rel=1e-15, abs=0)
    assert angle(u, origin, v, DEFAULT_EMBEDDING) > 0
    assert angle(u, origin, v, DEFAULT_EMBEDDING) < math.pi


def test_exact_metric_is_a_rational_quotient_metric_not_raw_tuple_equality():
    q = Quadray(Fraction(1, 3), Fraction(2, 5), 0, 0)
    gauge = 10**450
    alias = Quadray(*(value + gauge for value in q.as_tuple()))
    assert q != alias
    assert squared_distance_exact(q, alias) == 0
    assert dot_exact(q, alias) == squared_norm_exact(q) == Fraction(41, 75)
    assert squared_distance_exact(q, Quadray(0, 0, 0, 0)) == squared_norm_exact(q)
    assert squared_norm_exact((gauge,) * 4) == 0


def test_exact_custom_metric_uses_stored_binary_coefficients_and_nongauge_maps():
    embedding = ((0.1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0))
    q, r = Quadray(3, 1, 0, 0), Quadray(1, 2, 0, 0)
    assert dot_exact(q, r, (iter(row) for row in embedding)) == 3 * Fraction(0.1) ** 2 + 8
    assert squared_norm_exact(Quadray(1, 1, 1, 1), embedding) == Fraction(0.1) ** 2 + 4
    assert squared_distance_exact(q, r, embedding) == 4 * Fraction(0.1) ** 2 + 4


@pytest.mark.parametrize("components", [(1, 2, 3), (1, 2, 3, 4, 5)])
def test_exact_metric_rejects_missing_and_extra_native_dimensions(components):
    with pytest.raises(ValueError):
        dot_exact(iter(components), Quadray(1, 0, 0, 0))


def test_affine_combination_supports_signed_rational_weights_without_snapping():
    points = [Quadray(0, 0, 0, 0), Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0)]
    weights = [Fraction(-1, 2), Fraction(1, 3), Fraction(7, 6)]
    expected = (Fraction(1, 3), Fraction(7, 6), Fraction(0), Fraction(0))
    assert affine_combination_exact(iter(points), iter(weights)) == expected
    shifted = [tuple(value + 10**400 * (i + 1) for value in point.as_tuple())
               for i, point in enumerate(points)]
    assert affine_combination_exact(shifted, weights) == expected
    midpoint = affine_combination_exact(points[:2], [Fraction(1, 2)] * 2)
    assert midpoint == (Fraction(1, 2), Fraction(0), Fraction(0), Fraction(0))
    assert all(isinstance(value, Fraction) for value in midpoint)


@pytest.mark.parametrize("points,weights", [
    ([], []), ([Quadray(0, 0, 0, 0)], []),
    ([], [1]), ([Quadray(0, 0, 0, 0)], [Fraction(1, 2)]),
])
def test_affine_combination_rejects_missing_or_nonaffine_weights(points, weights):
    with pytest.raises(ValueError):
        affine_combination_exact(points, weights)


def test_rational_signed_simplex_orientation_translation_and_independent_gauges():
    points = [(0, 0, 0, 0), (Fraction(1, 2), 0, 0, 0),
              (0, 1, 0, 0), (0, 0, 1, 0)]
    assert signed_tetra_volume_exact(*points) == Fraction(1, 8)
    assert signed_tetra_volume_exact(points[1], points[0], *points[2:]) == Fraction(-1, 8)
    translation = (10**400, -10**401, 7, 11)
    shifted = [tuple(value + offset + i * 10**450
                     for value, offset in zip(point, translation))
               for i, point in enumerate(points)]
    assert signed_tetra_volume_exact(*shifted) == Fraction(1, 8)
    assert signed_tetra_volume_exact(points[0], points[1], points[1], points[3]) == 0


def test_exact_native_simplex_keeps_sliver_height_lost_by_float_squared_edges():
    n = 2**30
    points = [Quadray(0, 0, 0, 0), Quadray(n, 0, 0, n),
              Quadray(2 * n + 1, n, 1, n), Quadray(n, n, 0, 0)]
    assert abs(signed_tetra_volume_exact(*points)) == Fraction(n * n, 2)
    assert abs(signed_tetra_volume_exact(*points)) == integer_tetra_volume(*points)
    assert squared_distance_exact(points[1], points[2]) == 4 * n * n + 4


def test_triangle_squared_area_is_exact_gram_measure_with_custom_units():
    origin, a, b = (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)
    assert triangle_area_squared_exact(origin, a, b) == 2
    assert triangle_area_squared_exact(origin, a, (2, 0, 0, 0)) == 0
    embedding = tuple(tuple(Fraction(float(v)) / 2 for v in row) for row in DEFAULT_EMBEDDING)
    assert triangle_area_squared_exact(origin, a, b, embedding) == Fraction(1, 8)
    # A nongauge projection can collapse a nondegenerate native triangle.
    assert triangle_area_squared_exact(origin, a, b, ((1, 1, 0, 0), (0,) * 4, (0,) * 4)) == 0
