from __future__ import annotations

from typing import List


def bareiss_determinant_int(matrix: List[List[int]]) -> int:
    """Compute an exact integer determinant using the Bareiss algorithm.

    The Bareiss method is a fraction-free Gaussian elimination scheme that
    preserves integrality of intermediates. It is suitable for small to medium
    integer matrices where exactness matters.

    Parameters
    - matrix: Square list-of-lists of integers. Not mutated by this function.

    Returns
    - int: Exact determinant value (signed).
    """
    n = len(matrix)
    if n == 0:
        return 1
    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square")

    # Make a deep copy to avoid mutating caller data
    a = [row[:] for row in matrix]
    denom = 1
    sign = 1

    for k in range(n - 1):
        # Pivoting if necessary (simple partial pivot to avoid zero pivot)
        if a[k][k] == 0:
            pivot_row = None
            for r in range(k + 1, n):
                if a[r][k] != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                return 0
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign = -sign

        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                # Bareiss update with exact division
                num = a[i][j] * pivot - a[i][k] * a[k][j]
                if denom != 1:
                    # denom divides num exactly when inputs are integers
                    num //= denom
                a[i][j] = num
        denom = pivot

    det = a[n - 1][n - 1]
    return sign * det


def bareiss_rank(matrix: List[List[int]]) -> int:
    """Compute the exact integer rank of a matrix via Bareiss elimination.

    Performs fraction-free Gaussian elimination keeping track of the number
    of successful pivots. The rank equals the number of non-zero pivots.

    Parameters
    - matrix: Rectangular or square list-of-lists of integers.

    Returns
    - int: Rank of the matrix (0 for an empty matrix).
    """
    if not matrix:
        return 0
    m = len(matrix)
    n = len(matrix[0])
    # Work on a copy; pad to rectangular if needed
    a = [row[:] for row in matrix]

    rank = 0
    denom = 1

    for col in range(n):
        # Find a pivot in column col from row rank..m-1
        pivot_row = None
        for r in range(rank, m):
            if a[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue  # This column has no pivot; skip

        # Swap pivot row into position
        if pivot_row != rank:
            a[rank], a[pivot_row] = a[pivot_row], a[rank]

        pivot = a[rank][col]
        # Eliminate below
        for i in range(rank + 1, m):
            for j in range(col + 1, n):
                num = a[i][j] * pivot - a[i][col] * a[rank][j]
                if denom != 1:
                    num //= denom
                a[i][j] = num
            a[i][col] = 0

        denom = pivot
        rank += 1

    return rank


def integer_adjugate(matrix: List[List[int]]) -> List[List[int]]:
    """Compute the exact integer adjugate (classical adjoint) of a square matrix.

    The adjugate is the transpose of the cofactor matrix.  For an n×n integer
    matrix A, adj(A) satisfies A · adj(A) = det(A) · I.

    Uses ``bareiss_determinant_int`` for each (n-1)×(n-1) minor, so all
    arithmetic stays in exact integers.

    Parameters
    - matrix: Square list-of-lists of integers.

    Returns
    - List[List[int]]: The adjugate matrix (same size as input).

    Raises
    - ValueError: If the matrix is not square.
    """
    n = len(matrix)
    if n == 0:
        return []
    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square")

    adj: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Minor: delete row i, col j
            minor = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            cofactor = bareiss_determinant_int(minor)
            if (i + j) % 2 == 1:
                cofactor = -cofactor
            # Adjugate is the transpose of cofactor matrix
            adj[j][i] = cofactor

    return adj


__all__ = ["bareiss_determinant_int", "bareiss_rank", "integer_adjugate"]
