from geometry import minkowski_interval


def test_minkowski_interval_zero_space():
    assert minkowski_interval(dt=1.0, dx=0.0, dy=0.0, dz=0.0, c=1.0) == -1.0
