from __future__ import annotations

from sympy import Matrix, sqrt

from symbolic import (
    cayley_menger_volume_symbolic,
    convert_xyz_volume_to_ivm_symbolic,
)


def test_cayley_menger_volume_symbolic_unit_tetra():
    d2 = Matrix([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    V_xyz = cayley_menger_volume_symbolic(d2)
    assert str(V_xyz) == str(sqrt(2) / 12)
    V_ivm = convert_xyz_volume_to_ivm_symbolic(V_xyz)
    # S3 * sqrt(2)/12 simplifies to 1/8
    assert str(V_ivm.simplify()) in {"1/8", "0.125000000000000"}


