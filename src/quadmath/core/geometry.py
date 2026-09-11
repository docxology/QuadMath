from __future__ import annotations

import math


def minkowski_interval(dt: float, dx: float, dy: float, dz: float, c: float = 299792458.0) -> float:
    """Return the Minkowski interval squared ds^2 (Einstein.4D).

    Convention used: mostly-plus signature (-,+,+,+):
    ds^2 = -(c dt)^2 + dx^2 + dy^2 + dz^2

    This function lives in the Einstein.4D namespace (Minkowski spacetime),
    which is distinct from Euclidean E^4 (Coxeter.4D) and Quadray/IVM
    synergetics (Fuller.4D).

    Parameters
    - dt: Time difference.
    - dx, dy, dz: Spatial differences.
    - c: Speed of light (default SI units).

    Returns
    - float: The value of ds^2 under the chosen signature.
    """
    return - (c * c) * (dt * dt) + dx * dx + dy * dy + dz * dz


def lorentz_factor(v: float, c: float = 299792458.0) -> float:
    """Lorentz factor gamma = 1 / sqrt(1 - v^2/c^2) (Einstein.4D).

    Parameters
    - v: Velocity magnitude (same units as c).
    - c: Speed of light (default SI units).

    Returns
    - float: gamma >= 1.0

    Raises
    - ValueError: If |v| > c (superluminal) or c <= 0.
    """
    if c <= 0.0:
        raise ValueError("c must be positive")
    beta2 = (v * v) / (c * c)
    if beta2 > 1.0:
        raise ValueError("Superluminal velocity: |v| > c")
    if beta2 == 1.0:
        return float("inf")
    return 1.0 / math.sqrt(1.0 - beta2)


def proper_time(dt: float, dx: float, dy: float, dz: float, c: float = 299792458.0) -> float:
    """Proper time elapsed for a timelike interval (Einstein.4D).

    Computes dtau = sqrt(-ds^2) / c.  For spacelike or lightlike intervals
    the proper time is zero (no time passes along such paths).

    Parameters
    - dt: Coordinate time difference.
    - dx, dy, dz: Spatial coordinate differences.
    - c: Speed of light (default SI units).

    Returns
    - float: Non-negative proper time (same unit as dt for consistent units).
    """
    ds2 = minkowski_interval(dt, dx, dy, dz, c)
    if ds2 >= 0.0:
        # Spacelike or lightlike — no proper time
        return 0.0
    return math.sqrt(-ds2) / c


def spacetime_classify(ds2: float, tol: float = 1e-12) -> str:
    """Classify a Minkowski interval squared as timelike, spacelike, or lightlike.

    Uses the mostly-plus convention (-,+,+,+):
    - ds^2 < 0 => timelike (causal, particle can traverse)
    - ds^2 > 0 => spacelike (acausal, no particle can traverse)
    - ds^2 = 0 => lightlike (null, photon worldline)

    Parameters
    - ds2: The squared interval value.
    - tol: Tolerance for classifying as lightlike.

    Returns
    - str: One of "timelike", "spacelike", "lightlike".
    """
    if abs(ds2) < tol:
        return "lightlike"
    if ds2 < 0.0:
        return "timelike"
    return "spacelike"


__all__ = [
    "minkowski_interval",
    "lorentz_factor",
    "proper_time",
    "spacetime_classify",
]
