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


__all__ = ["bareiss_determinant_int"]


