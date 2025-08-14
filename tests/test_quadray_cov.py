from quadray import Quadray, ace_tetravolume_5x5


def test_ace_tetravolume_basic_nonzero():
    a = Quadray(1, 0, 0, 0)
    b = Quadray(0, 1, 0, 0)
    c = Quadray(0, 0, 1, 0)
    d = Quadray(0, 0, 0, 1)
    v = ace_tetravolume_5x5(a, b, c, d)
    assert v == 1

