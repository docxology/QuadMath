from __future__ import annotations


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
