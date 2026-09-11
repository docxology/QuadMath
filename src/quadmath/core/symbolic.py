from __future__ import annotations

"""Symbolic (SymPy) helpers for quadray-related formalisms.

Public API:
- cayley_menger_volume_symbolic
- convert_xyz_volume_to_ivm_symbolic
"""

from typing import Any


def cayley_menger_volume_symbolic(d2: Any) -> Any:
    """Return symbolic Euclidean tetrahedron volume from squared distances.

    Parameters
    - d2: 4x4 SymPy Matrix of squared distances with zeros on the diagonal

    Returns
    - sympy expression: sqrt(det(CM)/288)
    """
    from sympy import Matrix, sqrt

    if getattr(d2, "shape", None) != (4, 4):  # basic shape sanity
        raise ValueError("d2 must be 4x4")
    # Construct CM directly from d2
    CM = Matrix(
        [
            [0, 1, 1, 1, 1],
            [1, 0, d2[0, 1], d2[0, 2], d2[0, 3]],
            [1, d2[1, 0], 0, d2[1, 2], d2[1, 3]],
            [1, d2[2, 0], d2[2, 1], 0, d2[2, 3]],
            [1, d2[3, 0], d2[3, 1], d2[3, 2], 0],
        ]
    )
    det = CM.det()
    return sqrt(det / 288)


def convert_xyz_volume_to_ivm_symbolic(V_xyz: Any) -> Any:
    """Convert a symbolic Euclidean volume to IVM tetravolume via S3.

    Parameters
    - V_xyz: sympy expression for Euclidean tetra volume

    Returns
    - sympy expression: S3 * V_xyz with S3 = sqrt(9/8)
    """
    from sympy import sqrt

    S3 = sqrt(9) / sqrt(8)
    return S3 * V_xyz


__all__ = [
    "cayley_menger_volume_symbolic",
    "convert_xyz_volume_to_ivm_symbolic",
]


