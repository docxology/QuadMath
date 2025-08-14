from quadray import (
    Quadray,
    integer_tetra_volume,
    to_xyz,
    DEFAULT_EMBEDDING,
    ace_tetravolume_5x5,
    magnitude,
    dot,
)


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
