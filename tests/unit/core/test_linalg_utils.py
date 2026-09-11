import pytest

from quadmath.core.linalg_utils import bareiss_determinant_int, bareiss_rank, integer_adjugate


def test_bareiss_determinant_int_basic():
    m = [[1, 2], [3, 4]]
    assert bareiss_determinant_int(m) == -2


def test_bareiss_determinant_int_identity():
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert bareiss_determinant_int(m) == 1


def test_bareiss_determinant_int_rectangular_error():
    try:
        bareiss_determinant_int([[1, 2, 3], [4, 5, 6]])
        assert False
    except ValueError:
        assert True


def test_bareiss_empty():
    assert bareiss_determinant_int([]) == 1


# --------------- bareiss_rank tests ---------------


def test_bareiss_rank_full_rank():
    m = [[1, 0], [0, 1]]
    assert bareiss_rank(m) == 2


def test_bareiss_rank_identity_3x3():
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert bareiss_rank(m) == 3


def test_bareiss_rank_deficient():
    # Row 2 = 2 * Row 1; rank is 1
    m = [[1, 2, 3], [2, 4, 6]]
    assert bareiss_rank(m) == 1


def test_bareiss_rank_zero_matrix():
    m = [[0, 0], [0, 0]]
    assert bareiss_rank(m) == 0


def test_bareiss_rank_rectangular():
    m = [[1, 0, 0], [0, 1, 0]]
    assert bareiss_rank(m) == 2


def test_bareiss_rank_empty():
    assert bareiss_rank([]) == 0


def test_bareiss_rank_column_matrix():
    m = [[1], [2], [3]]
    assert bareiss_rank(m) == 1


def test_bareiss_rank_pivot_swap():
    """Matrix where first row has zero in first col, requiring a pivot swap."""
    m = [[0, 1], [1, 0]]
    assert bareiss_rank(m) == 2


def test_bareiss_rank_multi_step_elimination():
    """3x3 full-rank matrix that exercises the denom!=1 scaling path."""
    m = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    assert bareiss_rank(m) == 3


# --------------- integer_adjugate tests ---------------


def test_integer_adjugate_identity():
    m = [[1, 0], [0, 1]]
    adj = integer_adjugate(m)
    assert adj == [[1, 0], [0, 1]]


def test_integer_adjugate_2x2():
    # adj([[a,b],[c,d]]) = [[d,-b],[-c,a]]
    m = [[1, 2], [3, 4]]
    adj = integer_adjugate(m)
    assert adj == [[4, -2], [-3, 1]]


def test_integer_adjugate_identity_property():
    # A * adj(A) = det(A) * I
    m = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
    adj = integer_adjugate(m)
    det_A = bareiss_determinant_int(m)
    n = len(m)
    # Compute A * adj(A)
    for i in range(n):
        for j in range(n):
            val = sum(m[i][k] * adj[k][j] for k in range(n))
            expected = det_A if i == j else 0
            assert val == expected


def test_integer_adjugate_non_square_error():
    with pytest.raises(ValueError):
        integer_adjugate([[1, 2, 3], [4, 5, 6]])


def test_integer_adjugate_empty():
    assert integer_adjugate([]) == []
