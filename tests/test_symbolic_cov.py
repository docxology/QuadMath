from __future__ import annotations

from sympy import Matrix

from symbolic import cayley_menger_volume_symbolic


def test_cayley_menger_volume_symbolic_invalid_shape_raises():
    # Not 4x4 -> should raise ValueError
    bad = Matrix([[0, 1], [1, 0]])
    try:
        cayley_menger_volume_symbolic(bad)
    except ValueError as e:
        assert "4x4" in str(e)
    else:
        raise AssertionError("Expected ValueError for non-4x4 input matrix")


