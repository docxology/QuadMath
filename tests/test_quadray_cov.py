from fractions import Fraction

from quadray import Quadray, ace_tetravolume_5x5, integer_tetra_volume


def test_ace_tetravolume_basic_nonzero():
    a = Quadray(1, 0, 0, 0)
    b = Quadray(0, 1, 0, 0)
    c = Quadray(0, 0, 1, 0)
    d = Quadray(0, 0, 0, 1)
    v = ace_tetravolume_5x5(a, b, c, d)
    # Tetra of the four basis vectors is a unit IVM tetrahedron
    assert v == 1 and isinstance(v, Fraction)
    # Exact agreement with the projected-determinant implementation
    assert v == integer_tetra_volume(a, b, c, d)

