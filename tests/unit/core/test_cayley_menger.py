import numpy as np
import pytest

from quadmath.core.cayley_menger import (
    tetra_volume_cayley_menger,
    ivm_tetra_volume_cayley_menger,
    squared_distances_from_quadrays,
    tetra_circumradius,
    tetra_inradius,
)
from quadmath.core.quadray import Quadray, DEFAULT_EMBEDDING


def test_tetra_volume_cayley_menger_regular_tetra_unit_edge():
    d2 = np.ones((4, 4)) - np.eye(4)
    V = tetra_volume_cayley_menger(d2)
    assert np.isclose(V, np.sqrt(2.0) / 12.0, rtol=1e-6)


def test_tetra_volume_shape_error():
    try:
        tetra_volume_cayley_menger(np.ones((3, 3)))
        assert False
    except ValueError:
        assert True


def test_tetra_volume_degenerate():
    # Collinear points make determinant <= 0; volume should be 0.0
    d2 = np.zeros((4, 4))
    V = tetra_volume_cayley_menger(d2)
    assert V == 0.0


def test_cayley_menger_matches_integer_volume_for_simple_case():
    # Build squared distances between points in an embedding where edge length=1 implies V=1
    # For this synthetic check, use combinatorial distances: edges between distinct unit axes = 1
    d2 = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    V_xyz = tetra_volume_cayley_menger(d2)  # XYZ volume of regular tetra with edge 1
    # In XYZ units, V = sqrt(2)/12 for a unit-edge regular tetrahedron. The
    # quadray volume functions in quadmath/core/quadray.py measure a different object:
    # exact IVM tetravolume |det|/4 of lattice tetrahedra (unit IVM tetra = 1).
    assert np.isclose(V_xyz, np.sqrt(2.0) / 12.0, rtol=1e-6)


def test_ivm_tetra_volume_cayley_menger_regular_tetra_unit_edge():
    d2 = np.ones((4, 4)) - np.eye(4)
    V_ivm = ivm_tetra_volume_cayley_menger(d2)
    # For unit-edge regular tetra: V_xyz = sqrt(2)/12, S3 = sqrt(9/8) => V_ivm = 1/8
    assert np.isclose(V_ivm, 1.0 / 8.0, rtol=1e-6)


# --------------- New method tests ---------------


def test_squared_distances_from_quadrays_symmetric():
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    d2 = squared_distances_from_quadrays(p0, p1, p2, p3, DEFAULT_EMBEDDING)
    assert d2.shape == (4, 4)
    # Symmetric
    assert np.allclose(d2, d2.T)
    # Diagonal is zero
    assert np.allclose(np.diag(d2), 0.0)
    # All off-diagonal positive
    for i in range(4):
        for j in range(4):
            if i != j:
                assert d2[i, j] > 0.0


def test_squared_distances_volume_pipeline():
    """Verify that squared_distances -> tetra_volume pipeline is consistent."""
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    d2 = squared_distances_from_quadrays(p0, p1, p2, p3, DEFAULT_EMBEDDING)
    V = tetra_volume_cayley_menger(d2)
    assert V == pytest.approx(2.0 / 3.0, rel=1e-13, abs=0.0)


def test_tetra_circumradius_regular():
    # Regular tetrahedron with unit edge: R = sqrt(6)/4 ≈ 0.6124
    d2 = np.ones((4, 4)) - np.eye(4)
    R = tetra_circumradius(d2)
    expected = np.sqrt(6.0) / 4.0
    assert np.isclose(R, expected, rtol=1e-4)


def test_tetra_circumradius_degenerate():
    d2 = np.zeros((4, 4))
    R = tetra_circumradius(d2)
    assert R == 0.0


def test_tetra_circumradius_shape_error():
    with pytest.raises(ValueError):
        tetra_circumradius(np.ones((3, 3)))


def test_tetra_inradius_regular():
    # Regular tetrahedron with unit edge: r = 1 / sqrt(24) ≈ 0.2041
    d2 = np.ones((4, 4)) - np.eye(4)
    r = tetra_inradius(d2)
    expected = 1.0 / np.sqrt(24.0)
    assert np.isclose(r, expected, rtol=1e-4)


def test_tetra_inradius_degenerate():
    d2 = np.zeros((4, 4))
    r = tetra_inradius(d2)
    assert r == 0.0


def test_tetra_inradius_shape_error():
    with pytest.raises(ValueError):
        tetra_inradius(np.ones((3, 3)))


def test_circumradius_greater_than_inradius():
    d2 = np.ones((4, 4)) - np.eye(4)
    R = tetra_circumradius(d2)
    r = tetra_inradius(d2)
    assert R > r > 0.0


@pytest.mark.parametrize("length_scale", [1e-100, 1e-60, 1e-6, 1.0, 1e60, 1e100, 1e103])
def test_regular_tetrahedron_scale_covariance(length_scale):
    d2 = (np.ones((4, 4)) - np.eye(4)) * length_scale * length_scale
    # Multiply in this order: scale**3 itself overflows at 1e103, although
    # both physical volumes below are finite and representable.
    expected_volume = (np.sqrt(2.0) / 12.0) * length_scale * length_scale * length_scale
    expected_ivm = 0.125 * length_scale * length_scale * length_scale
    assert tetra_volume_cayley_menger(d2) == pytest.approx(
        expected_volume, rel=2e-13, abs=0.0
    )
    assert ivm_tetra_volume_cayley_menger(d2) == pytest.approx(
        expected_ivm, rel=2e-13, abs=0.0
    )
    assert tetra_circumradius(d2) == pytest.approx(
        np.sqrt(6.0) / 4.0 * length_scale, rel=2e-13, abs=0.0
    )
    assert tetra_inradius(d2) == pytest.approx(
        length_scale / np.sqrt(24.0), rel=2e-13, abs=0.0
    )


@pytest.mark.parametrize("length_scale", [1e-100, 1e-6, 1.0, 1e100])
def test_irregular_tetrahedron_matches_independent_geometry_at_all_scales(length_scale):
    # Vertices (0,0,0), (1,0,0), (0,2,0), (0,0,3).
    # V=1, circumcenter=(1/2,1,3/2), total face area=9, inradius=1/3.
    d2 = np.array([
        [0.0, 1.0, 4.0, 9.0],
        [1.0, 0.0, 5.0, 10.0],
        [4.0, 5.0, 0.0, 13.0],
        [9.0, 10.0, 13.0, 0.0],
    ]) * length_scale * length_scale
    assert tetra_volume_cayley_menger(d2) == pytest.approx(
        length_scale * length_scale * length_scale, rel=2e-13, abs=0.0
    )
    assert tetra_circumradius(d2) == pytest.approx(
        np.sqrt(14.0) / 2.0 * length_scale, rel=2e-13, abs=0.0
    )
    assert tetra_inradius(d2) == pytest.approx(
        length_scale / 3.0, rel=2e-13, abs=0.0
    )


@pytest.mark.parametrize("length_scale", [1e-150, 1e150])
def test_radii_do_not_require_representable_dimensional_volume(length_scale):
    d2 = (np.ones((4, 4)) - np.eye(4)) * length_scale * length_scale
    # The dimensional volume really is outside float64's range; finite radii
    # must not depend on successfully representing it.
    expected_volume = 0.0 if length_scale < 1.0 else float("inf")
    assert tetra_volume_cayley_menger(d2) == expected_volume
    assert ivm_tetra_volume_cayley_menger(d2) == expected_volume
    assert tetra_circumradius(d2) == pytest.approx(
        np.sqrt(6.0) / 4.0 * length_scale, rel=2e-13, abs=0.0
    )
    assert tetra_inradius(d2) == pytest.approx(
        length_scale / np.sqrt(24.0), rel=2e-13, abs=0.0
    )


@pytest.mark.parametrize("length_scale", [1e-100, 1.0, 1e100])
def test_coplanar_square_remains_degenerate_at_all_scales(length_scale):
    # A physical, zero-volume tetrahedron with four noncoincident vertices.
    d2 = np.array([
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0, 0.0],
    ]) * length_scale * length_scale
    assert tetra_volume_cayley_menger(d2) == 0.0
    assert ivm_tetra_volume_cayley_menger(d2) == 0.0
    assert tetra_circumradius(d2) == 0.0
    assert tetra_inradius(d2) == 0.0


def test_ivm_volume_preserves_sphere_radius_one_convention():
    vertices = (
        Quadray(0, 0, 0, 0),
        Quadray(2, 1, 1, 0),
        Quadray(1, 2, 1, 0),
        Quadray(1, 1, 2, 0),
    )
    d2 = squared_distances_from_quadrays(
        *vertices, np.asarray(DEFAULT_EMBEDDING) / np.sqrt(2.0)
    )
    np.testing.assert_allclose(d2, 4.0 * (np.ones((4, 4)) - np.eye(4)),
                               rtol=1e-14, atol=0.0)
    assert ivm_tetra_volume_cayley_menger(d2) == pytest.approx(1.0, rel=1e-13)


@pytest.mark.parametrize("dtype", (int, np.int64))
@pytest.mark.parametrize("single_use_embedding", (False, True))
def test_pairwise_geometry_preserves_large_spatial_translations(dtype, single_use_embedding):
    offset = 2**60
    points = [
        Quadray(*(dtype(value) for value in row))
        for row in (
            (offset, 0, 0, 0), (offset + 1, 0, 0, 0),
            (offset, 1, 0, 0), (offset, 0, 1, 0),
        )
    ]
    embedding = (
        (iter(row) for row in DEFAULT_EMBEDDING)
        if single_use_embedding else DEFAULT_EMBEDDING
    )
    d2 = squared_distances_from_quadrays(*points, embedding)
    expected = np.array([
        [0.0, 3.0, 3.0, 3.0],
        [3.0, 0.0, 8.0, 8.0],
        [3.0, 8.0, 0.0, 8.0],
        [3.0, 8.0, 8.0, 0.0],
    ])
    np.testing.assert_array_equal(d2, expected)
    assert tetra_volume_cayley_menger(d2) == pytest.approx(2 / 3, rel=1e-13)
    assert tetra_circumradius(d2) == pytest.approx(tetra_circumradius(expected))
    assert tetra_inradius(d2) == pytest.approx(tetra_inradius(expected))


def test_pairwise_displacements_preserve_nearby_points_far_from_first_vertex():
    offset = 2**60
    d2 = squared_distances_from_quadrays(
        Quadray(0, 0, 0, 0), Quadray(offset, 0, 0, 0),
        Quadray(offset + 1, 0, 0, 0), Quadray(offset, 1, 0, 0),
        DEFAULT_EMBEDDING,
    )
    assert d2[1, 2] == 3.0
    assert d2[1, 3] == 3.0
    assert d2[2, 3] == 8.0


@pytest.mark.parametrize("scalar", (int, np.int64))
def test_integer_embedding_distances_do_not_square_in_fixed_width(scalar):
    length = 2**32
    points = (
        Quadray(0, 0, 0, 0), Quadray(length, 0, 0, 0),
        Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0),
    )
    embedding = tuple(tuple(scalar(v) for v in row) for row in DEFAULT_EMBEDDING)
    actual = squared_distances_from_quadrays(*points, embedding)
    # Independent integer Cartesian oracle; no NumPy arithmetic before float().
    images = [
        tuple(sum(int(weight) * value for weight, value in zip(row, point.as_tuple()))
              for row in embedding)
        for point in points
    ]
    expected = [
        [float(sum((a - b) ** 2 for a, b in zip(left, right))) for right in images]
        for left in images
    ]
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] == float(3 * 2**64)


def test_custom_rational_distances_round_only_after_exact_squaring():
    from fractions import Fraction

    n = 2**60
    embedding = (
        (n + 1, -n, 0, 0),
        (Fraction(1, 3), Fraction(1, 7), 0, 0),
        (0, 0, Fraction(5, 11), 0),
    )
    points = (
        Quadray(0, 0, 0, 0), Quadray(1, 1, 0, 0),
        Quadray(0, 0, 1, 0), Quadray(0, 0, 0, 1),
    )
    images = [
        tuple(sum(weight * value for weight, value in zip(row, point.as_tuple()))
              for row in embedding)
        for point in points
    ]
    expected = [
        [float(sum((a - b) ** 2 for a, b in zip(left, right))) for right in images]
        for left in images
    ]
    actual = squared_distances_from_quadrays(
        *points, (iter(row) for row in embedding)
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] == float(1 + Fraction(10, 21)**2)
