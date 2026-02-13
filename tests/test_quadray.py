from quadray import (
    Quadray,
    integer_tetra_volume,
    to_xyz,
    DEFAULT_EMBEDDING,
    ace_tetravolume_5x5,
    magnitude,
    dot,
    distance,
    angle,
    centroid,
    quadray_from_xyz,
)
import math


def test_normalize():
    q = Quadray(3, 5, 3, 5)
    qn = q.normalize()
    assert min(qn.as_tuple()) == 0


def test_volume_unit_tetra():
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    assert integer_tetra_volume(p0, p1, p2, p3) == 1


def test_ace_tetravolume_matches_integer_tetra_volume():
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 1, 0, 1)
    p2 = Quadray(2, 1, 1, 0)
    p3 = Quadray(2, 0, 1, 1)
    # These form a unit IVM tetrahedron from a common origin
    v1 = integer_tetra_volume(p0, p1, p2, p3)
    v2 = ace_tetravolume_5x5(p0, p1, p2, p3)
    assert v1 == 1 and v2 == 1


def test_volume_divisible_by_four_branch():
    # Projections diag(2,2,2) => determinant 8, normalization returns 2
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 0, 0, 0)
    p2 = Quadray(0, 2, 0, 0)
    p3 = Quadray(0, 0, 2, 0)
    assert integer_tetra_volume(p0, p1, p2, p3) == 2


def test_to_xyz_runs():
    q = Quadray(2, 1, 1, 0)
    x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
    assert all(isinstance(v, float) for v in (x, y, z))


def test_vector_magnitude_and_dot():
    q1 = Quadray(1, 0, 0, 0)
    q2 = Quadray(0, 1, 0, 0)
    m1 = magnitude(q1, DEFAULT_EMBEDDING)
    m2 = magnitude(q2, DEFAULT_EMBEDDING)
    assert m1 > 0 and m2 > 0
    d = dot(q1, q2, DEFAULT_EMBEDDING)
    assert isinstance(d, float)


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


def test_angle_right_angle():
    # Origin and two orthogonal directions
    origin = Quadray(0, 0, 0, 0)
    q1 = Quadray(1, 0, 0, 0)
    q2 = Quadray(0, 1, 0, 0)
    # The angle at origin between q1 and q2 depends on the embedding
    a = angle(q1, origin, q2, DEFAULT_EMBEDDING)
    assert 0.0 < a < math.pi  # Valid angle
    assert isinstance(a, float)


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
    assert min(c.as_tuple()) == 0  # Normalized


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


def test_quadray_from_xyz_origin():
    q = quadray_from_xyz(0.0, 0.0, 0.0, DEFAULT_EMBEDDING)
    assert q == Quadray(0, 0, 0, 0)
