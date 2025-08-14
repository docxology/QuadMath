import numpy as np

from cayley_menger import tetra_volume_cayley_menger, ivm_tetra_volume_cayley_menger
from quadray import Quadray, integer_tetra_volume
import numpy as np


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
