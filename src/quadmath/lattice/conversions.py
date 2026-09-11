"""XYZ <-> IVM conversions for quadray coordinates (Fuller.4D -> Coxeter.4D slice).

This module is the wave-1 conversions layer pinned by the forthcoming root
SPEC.md (wave 2 dispatches SPEC.md + tests/unit/lattice/test_spec_examples.py against
exactly this API).  Five public surfaces:

- :func:`urner_embedding` -- canonical symmetric 3x4 embedding matrix.
- :func:`quadray_to_xyz` -- quadray -> R^3, delegating to quadray.to_xyz.
- :func:`xyz_to_quadray_canonical` -- EXACT rational inverse (fractions.Fraction
  arithmetic, no tolerance, no rounding); returns the canonical (normalized,
  min-0) integer quadray for points in the embedding image.
- :func:`quadray_roundtrip` -- assertion wrapper proving the exact identity
  quadray_from_xyz(to_xyz(q)) == q.
- :func:`embedding_basis` -- the four embedding columns as a (4,3) matrix whose
  Gram matrix is G = 4*I4 - J4 at scale 1 (C_i . C_j = 4*delta_ij - 1).
"""
from __future__ import annotations

import numbers
from fractions import Fraction
from typing import Optional, Sequence, Tuple

import numpy as np

import quadmath.core.quadray as quadray
from quadmath.core.quadray import Quadray

__all__ = [
    "urner_embedding",
    "quadray_to_xyz",
    "xyz_to_quadray_canonical",
    "quadray_roundtrip",
    "embedding_basis",
]


def _validate_embedding(M: Optional[np.ndarray]) -> np.ndarray:
    """Validate a 3x4 embedding matrix and return it as an ndarray.

    Parameters
    - M: 3x4 embedding matrix (array-like), or None for
      :data:`quadray.DEFAULT_EMBEDDING`.

    Returns
    - np.ndarray: The validated (3, 4) embedding.

    Raises
    - ValueError: If M is not None and does not have shape (3, 4).
    """
    if M is None:
        return np.asarray(quadray.DEFAULT_EMBEDDING)
    arr = np.asarray(M)
    if arr.shape != (3, 4):
        raise ValueError(
            f"embedding must have shape (3, 4), got {arr.shape}"
        )
    return arr


def _embedding_rows(
    M: Optional[np.ndarray],
) -> Tuple[Tuple[float, float, float, float], ...]:
    """Validate a 3x4 embedding and return its rows as tuples of Python floats.

    Shared internal helper: :func:`quadray_to_xyz` delegates through it and
    :func:`embedding_basis` reuses it for shape validation.  M=None means
    :data:`quadray.DEFAULT_EMBEDDING`.

    Parameters
    - M: 3x4 embedding matrix, or None for the default embedding.

    Returns
    - Tuple of three rows, each a 4-tuple of Python floats.

    Raises
    - ValueError: If M does not have shape (3, 4).
    """
    arr = _validate_embedding(M)
    return tuple(
        (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        for row in arr
    )


def _exact_fraction(v: object) -> Fraction:
    """Convert one embedding/xyz entry to an exact Fraction.

    A Python float (or np.float64, a float subclass) becomes its exact binary
    rational; numpy integers convert through numbers.Integral; Fraction passes
    through unchanged.  np.float32 and every other type is rejected.

    Parameters
    - v: Entry of type int, float, Fraction, or numbers.Integral.

    Returns
    - Fraction: The exact rational value of v.

    Raises
    - TypeError: If v is not int, float, Fraction, or numbers.Integral.
    """
    if isinstance(v, numbers.Integral):
        return Fraction(int(v))
    if isinstance(v, Fraction):
        return v
    if isinstance(v, float):
        return Fraction(v)
    raise TypeError(
        "entries must be int, float, Fraction, or numbers.Integral, "
        f"got {type(v).__name__}"
    )


def urner_embedding(scale: float = 1.0) -> np.ndarray:
    """Return a 3x4 Urner-style symmetric embedding matrix (Fuller.4D -> Coxeter.4D slice).

    The rows map the four quadray axes (A,B,C,D) to the vertices of a regular
    tetrahedron in R^3. Scaling the matrix scales all resulting XYZ coordinates.

    Parameters
    - scale: Uniform scalar applied to the embedding (default 1.0).

    Returns
    - np.ndarray: A 3x4 matrix suitable for use with `quadray_to_xyz`.
    """
    M = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    return scale * M


def quadray_to_xyz(
    q: Quadray, M: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Map a `Quadray` to Cartesian XYZ via a 3x4 embedding matrix (Fuller.4D -> Coxeter.4D slice).

    Thin wrapper delegating to :func:`quadray.to_xyz`: columns of `M`
    correspond to the (a,b,c,d) axes, and rows to X,Y,Z.  M=None uses the
    default embedding, so `quadray_to_xyz(q)` is identical to
    `quadray.to_xyz(q, quadray.DEFAULT_EMBEDDING)`.

    Parameters
    - q: The input quadray coordinate with non-negative components.
    - M: 3x4 embedding matrix (e.g., from `urner_embedding`), or None for the
      default embedding.

    Returns
    - Tuple[float, float, float]: (x, y, z) in R^3.

    Raises
    - ValueError: If M does not have shape (3, 4).
    """
    return quadray.to_xyz(q, _embedding_rows(M))


def _det3(m: Sequence[Sequence[Fraction]]) -> Fraction:
    """Exact 3x3 determinant over Fractions.

    Parameters
    - m: Nested 3x3 tuple of Fractions.

    Returns
    - Fraction: The exact determinant.
    """
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = m
    return (
        m00 * (m11 * m22 - m12 * m21)
        - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20)
    )


def xyz_to_quadray_canonical(
    xyz: Sequence, M: Optional[np.ndarray] = None
) -> Quadray:
    """Recover the canonical integer quadray for an XYZ point in the embedding image, EXACTLY.

    Arithmetic is exact (:class:`fractions.Fraction`): a float entry is its
    exact binary rational, with NO rounding and NO tolerance.  Because every
    row of a valid embedding sums to exactly 0, the fiber over an image point
    xyz is {y + t*(1,1,1,1) : t in Z} where y = (x1, x2, x3, 0) is the
    particular preimage solved on columns 0-2; the min-0 representative of
    that fiber is unique, so the tie-break is deterministic (the same rule as
    :meth:`quadray.Quadray.normalize`).

    Rank lemma: with all row sums 0, C3 = -(C0+C1+C2) holds, so rank(M) = 3
    iff det[C0 C1 C2] != 0; the integrality of (x1, x2, x3) is then exactly
    the membership of xyz in the lattice image {M*q : q in Z^4} (every integer
    solution is y + t*(1,1,1,1), t in Z).

    Entry types: int, float, Fraction, numpy integer (np.int64 goes through
    numbers.Integral; np.float64 is a float subclass and is accepted).
    np.float32 and other types are REJECTED.  The same entry types apply to M.

    Parameters
    - xyz: Length-3 sequence of int | float | Fraction | numpy integer.
    - M: 3x4 embedding whose rows each sum to exactly 0, or None for the
      default embedding.

    Returns
    - Quadray: The canonical integer quadray (normalized, minimum component 0)
      whose image under M equals xyz exactly.

    Raises
    - ValueError: If xyz is not a length-3 sequence; M has the wrong shape;
      any row of M does not sum to exactly 0; the embedding has rank < 3
      (det of columns 0-2 == 0); or xyz is not in the embedding image
      (non-integral preimage).
    - TypeError: If an xyz or M entry is not int, float, Fraction, or
      numbers.Integral.
    """
    try:
        length = len(xyz)
    except TypeError:
        raise ValueError("xyz must be a length-3 sequence") from None
    if length != 3:
        raise ValueError(f"xyz must have length 3, got {length}")
    vec = [_exact_fraction(v) for v in xyz]
    arr = _validate_embedding(M)
    rows = [
        [_exact_fraction(arr[j][i]) for i in range(4)]
        for j in range(3)
    ]
    for j, row in enumerate(rows):
        row_sum = row[0] + row[1] + row[2] + row[3]
        if row_sum != 0:
            raise ValueError(
                f"embedding row {j} does not sum to exactly 0 "
                f"(got {row_sum}); normalization must preserve the image point"
            )
    # System C0*x1 + C1*x2 + C2*x3 = xyz (the particular preimage has 4th coord 0):
    # A[j][k] = C_k[j] = rows[j][k] is the top-left 3x3 block of the embedding.
    A = tuple(
        (row[0], row[1], row[2]) for row in rows
    )
    det = _det3(A)
    if det == 0:
        raise ValueError(
            "embedding has rank < 3 (determinant of columns 0-2 is 0); "
            "cannot solve for a unique preimage"
        )
    pre: list = []
    for k in range(3):
        replaced = tuple(
            tuple(vec[j] if kk == k else A[j][kk] for kk in range(3))
            for j in range(3)
        )
        pre.append(_det3(replaced) / det)
    if any(x.denominator != 1 for x in pre):
        raise ValueError(
            f"xyz {tuple(vec)} is not in the embedding image "
            f"(non-integral preimage {tuple(pre)})"
        )
    return Quadray(int(pre[0]), int(pre[1]), int(pre[2]), 0).normalize()


def quadray_roundtrip(q: Quadray, M: Optional[np.ndarray] = None) -> Quadray:
    """Assert the exact round-trip identity recon == q for q -> XYZ -> quadray.

    Both legs use the SAME embedding rows: the forward leg maps q to R^3
    with rows = :func:`_embedding_rows` of M (M=None ->
    :data:`quadray.DEFAULT_EMBEDDING`), and the inverse leg inverts that
    same image with the same rows via :func:`quadray.quadray_from_xyz`.

    recon == q holds EXACTLY for every normalized integer quadray at ANY
    Urner scale c != 0: pinv(cM)*(cM*q) = q - (sum(q)/4)*(1,1,1,1) is
    scale-independent (pinv(cM) = (1/c)*pinv(M)), and round-half-up
    preserves the (1,1,1,1)-coset (see quadray_from_xyz, src/quadmath/core/quadray.py).
    At c = 1/2 the image is the half-integer FCC picture of the same sites
    and round-trips through the same scaled embedding.

    Documented failure that raises AssertionError: q not normalized --
    quadray_from_xyz returns quadray.normalize(q), e.g. Quadray(2,2,2,1)
    comes back as Quadray(1,1,1,0).  Embeddings whose rows do not sum to
    exactly 0 are outside the Urner family and outside this guarantee.

    Parameters
    - q: Input quadray (normalized integer for the exact round-trip).
    - M: 3x4 embedding used by BOTH legs, or None for the default.

    Returns
    - Quadray: The round-tripped recon, equal to q exactly.

    Raises
    - ValueError: If M does not have shape (3, 4).
    - AssertionError: If the round-trip is not the identity (q not
      normalized).
    """
    forward_rows = _embedding_rows(M)
    x, y, z = quadray.to_xyz(q, forward_rows)
    recon = quadray.quadray_from_xyz(x, y, z, forward_rows)
    if recon != q:
        raise AssertionError(
            f"roundtrip is not the identity: {q} -> xyz -> {recon}"
        )
    return recon


def embedding_basis(M: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the four embedding COLUMNS as a (4,3) basis matrix (Fuller.4D axes).

    Row i of B is the embedding column C_i (i = 0..3) of the 3x4 matrix, so
    B is the (4,3) transpose-shaped basis whose Gram matrix is

        B @ B.T == 4*np.eye(4) - np.ones((4, 4))

    at scale 1: C_i . C_j = 4*delta_ij - 1 (diagonal 3, off-diagonal -1) --
    the identity :func:`lattice_search.squared_distance` relies on
    (d^2 = 4*sum(delta^2) - (sum delta)^2).  Scaling the embedding by c
    scales the Gram matrix by c^2.

    Parameters
    - M: 3x4 embedding matrix, or None for the default embedding.

    Returns
    - np.ndarray: Shape (4, 3) float matrix B whose rows are the columns of M.

    Raises
    - ValueError: If M does not have shape (3, 4).
    """
    rows = _embedding_rows(M)
    return np.array(
        [[rows[0][i], rows[1][i], rows[2][i]] for i in range(4)],
        dtype=float,
    )
