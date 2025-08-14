from linalg_utils import bareiss_determinant_int


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


