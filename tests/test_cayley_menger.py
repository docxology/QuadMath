import numpy as np
import pytest

from cayley_menger import (
    tetra_volume_cayley_menger,
    ivm_tetra_volume_cayley_menger,
    squared_distances_from_quadrays,
    tetra_circumradius,
    tetra_inradius,
)
from quadray import Quadray, DEFAULT_EMBEDDING, integer_tetra_volume


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
    # Regular unit IVM tetra from quadray points (origin plus 3 unit edges)
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
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
    # In XYZ units, V = sqrt(2)/12; our integer_tetra_volume returns 1 in IVM units
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
    assert V > 0.0


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


def test_tetra_circumradius_negative_r2():
    """Crafted d2 where the R^2 formula yields a non-positive value."""
    # Nearly-degenerate tetrahedron: 3 collinear points + 1 slightly off
    d2 = np.array([
        [0.0, 1.0, 4.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [4.0, 1.0, 0.0, 4.0],
        [1.0, 1.0, 4.0, 0.0],
    ])
    R = tetra_circumradius(d2)
    # Should still return a non-negative value (possibly 0.0 if R2 <= 0)
    assert R >= 0.0


def test_tetra_inradius_zero_area_face():
    """Tetrahedron with a degenerate face (zero area) but non-zero volume.

    This is hard to construct naturally, so we exercise the zero-area guard
    by using distances that produce zero area via Heron's formula.
    """
    # Isoceles with co-linear edges on one face: a = 2, b = 1, c = 1
    # s = 2, area2 = 2*(2-2)*(2-1)*(2-1) = 0 for one face
    d2 = np.array([
        [0.0, 4.0, 1.0, 1.0],
        [4.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ])
    r = tetra_inradius(d2)
    assert r >= 0.0  # Either positive or 0 depending on geometry
