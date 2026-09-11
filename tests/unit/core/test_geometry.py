import math
import pytest

from quadmath.core.geometry import minkowski_interval, lorentz_factor, proper_time, spacetime_classify


def test_minkowski_interval_zero_space():
    assert minkowski_interval(dt=1.0, dx=0.0, dy=0.0, dz=0.0, c=1.0) == -1.0


def test_minkowski_interval_spacelike():
    ds2 = minkowski_interval(dt=0.0, dx=1.0, dy=0.0, dz=0.0, c=1.0)
    assert ds2 == 1.0  # Spacelike


def test_minkowski_interval_lightlike():
    ds2 = minkowski_interval(dt=1.0, dx=1.0, dy=0.0, dz=0.0, c=1.0)
    assert abs(ds2) < 1e-12  # Lightlike


# --- Lorentz factor ---

def test_lorentz_factor_at_rest():
    gamma = lorentz_factor(0.0, c=1.0)
    assert gamma == 1.0


def test_lorentz_factor_half_c():
    gamma = lorentz_factor(0.5, c=1.0)
    expected = 1.0 / math.sqrt(1.0 - 0.25)
    assert abs(gamma - expected) < 1e-12


def test_lorentz_factor_high_speed():
    gamma = lorentz_factor(0.99, c=1.0)
    assert gamma > 7.0  # Should be ~7.09


def test_lorentz_factor_at_c():
    gamma = lorentz_factor(1.0, c=1.0)
    assert gamma == float("inf")


def test_lorentz_factor_superluminal():
    with pytest.raises(ValueError, match="Superluminal"):
        lorentz_factor(1.5, c=1.0)


def test_lorentz_factor_negative_c():
    with pytest.raises(ValueError, match="c must be positive"):
        lorentz_factor(0.5, c=-1.0)


# --- Proper time ---

def test_proper_time_timelike():
    # Pure time: dtau = dt
    dtau = proper_time(dt=1.0, dx=0.0, dy=0.0, dz=0.0, c=1.0)
    assert abs(dtau - 1.0) < 1e-12


def test_proper_time_spacelike():
    dtau = proper_time(dt=0.0, dx=1.0, dy=0.0, dz=0.0, c=1.0)
    assert dtau == 0.0  # No proper time for spacelike


def test_proper_time_moving_observer():
    # Observer moving at v = 0.6c laterally
    dtau = proper_time(dt=1.0, dx=0.6, dy=0.0, dz=0.0, c=1.0)
    expected = math.sqrt(1.0 - 0.36)  # sqrt(1 - v^2/c^2) * dt
    assert abs(dtau - expected) < 1e-12


# --- Spacetime classify ---

def test_spacetime_classify_timelike():
    assert spacetime_classify(-1.0) == "timelike"


def test_spacetime_classify_spacelike():
    assert spacetime_classify(1.0) == "spacelike"


def test_spacetime_classify_lightlike():
    assert spacetime_classify(0.0) == "lightlike"
    assert spacetime_classify(1e-14) == "lightlike"  # Within tolerance
