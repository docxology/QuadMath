"""Fast nearest-site queries on the IVM quadray lattice (Fuller.4D).

This module answers "which close-packed IVM sphere centers lie near a
given point?" using a precomputed ball index: the sites of the
omnidirectional close packing (see ``quadmath.lattice.omni_numbering``) grouped by
frequency shell, with all heavy work vectorized in NumPy (no scipy).

Geometry
--------
Under ``quadmath.core.quadray.DEFAULT_EMBEDDING`` the four basis columns satisfy
``C_i . C_j = 4*delta_ij - 1`` (the row Gram matrix is ``4I``), so for any
integer 4-vectors ``p, q`` the Euclidean squared distance is the exact
integer

    d2(p, q) = 4 * sum_j (p_j - q_j)^2 - (sum_j (p_j - q_j))^2,

which the test suite cross-validates against the float pipeline
``quadmath.core.quadray.distance``.  Every site ``s`` on frequency shell ``g`` of the
close packing satisfies ``d2(origin, s) >= 8g`` (verified by exhaustive
breadth-first enumeration through shell 8 in the tests); this bound
makes shell-ordered sweeps safe to truncate:

- ``within_radius`` enumerates exactly through shell
  ``floor((||center|| + R)^2 / 8)`` -- any site closer than ``R`` to the
  center must live on a shell at most that deep (triangle inequality
  plus the invariant above).
- ``nearest`` sweeps shells outward and stops once no unvisited shell
  can contain a site closer than the current k-th best distance.

Conventions
-----------
- Centers may be arbitrary real 4-vectors (quadray components); sites
  returned are canonical integer lattice sites (non-negative, at least
  one zero component).
- The center itself is a legitimate answer: querying a lattice site at
  radius >= 0 returns that site at distance 0 (its canonical
  representative under projective normalization).
- Radii are bounded by ``MAX_SHELL`` (the deepest precomputed shell);
  requests that would need a deeper enumeration raise ``ValueError``
  rather than silently truncating results.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from quadmath.lattice.omni_numbering import MAX_SHELL, Quadray, _build_through_shell

#: Slack (squared-distance units) used when truncating shell sweeps, so
#: that exact ties at the truncation boundary are never discarded.
_EPS: float = 1e-6

__all__ = [
    "MAX_SHELL",
    "squared_distance",
    "nearest",
    "within_radius",
]


def _validate_center(
    site: Union[Quadray, Tuple[int, int, int, int], List[int], np.ndarray]
) -> np.ndarray:
    """Validate and convert a query center to a length-4 numeric array.

    Parameters
    - site: Quadray vector or 4-sequence of real components.

    Returns
    - np.ndarray: (4,) float64 array of quadray components.

    Raises
    - TypeError: If the input is not numeric.
    - ValueError: If the shape is not (4,).
    """
    raw = site.as_tuple() if isinstance(site, Quadray) else tuple(site)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"center must have exactly 4 components, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("center components must be finite")
    return arr


def _validate_radius(R: float) -> float:
    """Validate a search radius.

    Parameters
    - R: Candidate radius.

    Returns
    - float: The radius as a float.

    Raises
    - TypeError: If ``R`` is not a real number.
    - ValueError: If ``R`` is negative or NaN.
    """
    if isinstance(R, bool) or not isinstance(R, (int, float, np.floating, np.integer)):
        raise TypeError("radius must be a real number")
    r = float(R)
    if np.isnan(r):
        raise ValueError("radius must not be NaN")
    if r < 0.0:
        raise ValueError("radius must be non-negative")
    return r


def squared_distance(p: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Exact lattice squared distances from ``p`` to rows of ``sites``.

    Uses the integer-exact identity
    ``d2 = 4*sum_j delta_j^2 - (sum_j delta_j)^2`` implied by the
    ``DEFAULT_EMBEDDING`` Gram matrix (``4I - J``); for float inputs the
    same identity is evaluated in float64.

    Parameters
    - p: (4,) query components.
    - sites: (n, 4) site components.

    Returns
    - np.ndarray: (n,) squared Euclidean distances.
    """
    delta = sites.astype(np.float64) - p
    return 4.0 * np.sum(delta * delta, axis=1) - np.sum(delta, axis=1) ** 2


def _max_shell_for(center_norm: float, R: float) -> int:
    """Deepest shell that must be enumerated for a query.

    Any site within ``R`` of the center satisfies
    ``d2(origin, s) >= 8g`` and ``||s|| <= ||center|| + R``, hence
    ``g <= (||center|| + R)^2 / 8``.  A small slack absorbs float
    rounding at the boundary.

    Parameters
    - center_norm: Euclidean norm of the query center.
    - R: Search radius.

    Returns
    - int: Required deepest shell index (>= 0).
    """
    bound = ((center_norm + R) ** 2 + _EPS) / 8.0
    return int(np.floor(bound)) + 1


def within_radius(
    site: Union[Quadray, Tuple[int, int, int, int], List[int], np.ndarray],
    R: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return all IVM lattice sites within Euclidean radius ``R`` of ``site``.

    The search enumerates the close packing exactly through the shell
    depth required by the triangle-inequality bound, then filters with
    exact squared distances.  Results are sorted by squared distance,
    with lexicographic ``(a, b, c, d)`` order as the deterministic
    tie-break.

    Parameters
    - site: Query center (Quadray or 4-sequence of real components).
    - R: Euclidean search radius (non-negative).

    Returns
    - tuple of (np.ndarray, np.ndarray):
        - sites: (m, 4) int64 array of canonical lattice sites.
        - d2: (m,) float64 squared distances, ``d2 <= R^2``, ascending.

    Raises
    - TypeError: If ``site`` or ``R`` has an invalid type.
    - ValueError: If ``R`` is negative/NaN or the required enumeration
      would exceed shell ``MAX_SHELL``.
    """
    center = _validate_center(site)
    r = _validate_radius(R)
    needed = _max_shell_for(float(np.sqrt(squared_distance(center, np.zeros((1, 4)))[0])), r)
    if needed > MAX_SHELL:
        raise ValueError(
            f"radius {r} at norm {np.linalg.norm(center):.3f} needs shell {needed} "
            f"> MAX_SHELL={MAX_SHELL}; reduce R or move the query near the origin"
        )
    sites, _, _ = _build_through_shell(needed)
    d2 = squared_distance(center, sites)
    keep = d2 <= r * r
    sel_sites = sites[keep]
    sel_d2 = d2[keep]
    order = np.lexsort((sel_sites[:, 3], sel_sites[:, 2], sel_sites[:, 1],
                        sel_sites[:, 0], sel_d2))
    return sel_sites[order], sel_d2[order]


def nearest(
    site: Union[Quadray, Tuple[int, int, int, int], List[int], np.ndarray],
    R: float,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` nearest IVM lattice sites within radius ``R``.

    Sweeps frequency shells outward from the center.  Each shell's
    squared distances are ranked with ``numpy.argsort``; the sweep stops
    early once ``k`` candidates are in hand and the shell invariant
    ``d2(origin, s) >= 8g`` guarantees no unvisited shell can improve
    the k-th best distance.  Ties are broken lexicographically by
    ``(a, b, c, d)``.

    Parameters
    - site: Query center (Quadray or 4-sequence of real components).
    - R: Euclidean search radius (non-negative).  Candidates farther
      than ``R`` are never returned, even if fewer than ``k`` exist.
    - k: Number of neighbors requested (>= 1).

    Returns
    - tuple of (np.ndarray, np.ndarray):
        - sites: (m, 4) int64 array with ``m = min(k, hits)`` rows.
        - d2: (m,) float64 squared distances, ascending.

    Raises
    - TypeError: If ``site`` or ``R`` has an invalid type.
    - ValueError: If ``R`` is negative/NaN, ``k < 1``, or the required
      enumeration would exceed shell ``MAX_SHELL``.
    """
    center = _validate_center(site)
    r = _validate_radius(R)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    center_norm = float(np.sqrt(squared_distance(center, np.zeros((1, 4)))[0]))
    needed = _max_shell_for(center_norm, r)
    if needed > MAX_SHELL:
        raise ValueError(
            f"radius {r} at norm {center_norm:.3f} needs shell {needed} "
            f"> MAX_SHELL={MAX_SHELL}; reduce R or move the query near the origin"
        )
    sites, _, offsets = _build_through_shell(needed)
    cand_sites: List[np.ndarray] = []
    cand_d2: List[np.ndarray] = []
    have = 0
    for g in range(needed + 1):
        shell = sites[offsets[g]:offsets[g + 1]]
        d2_shell = squared_distance(center, shell)
        in_radius = d2_shell <= r * r
        if np.any(in_radius):
            order = np.argsort(d2_shell[in_radius], kind="stable")
            cand_sites.append(shell[in_radius][order])
            cand_d2.append(d2_shell[in_radius][order])
            have += int(np.count_nonzero(in_radius))
            if have >= k:
                kth = np.partition(np.concatenate(cand_d2), k - 1)[k - 1]
                # Shell invariant: every shell-g site has d2(origin, s) >= 8g,
                # so d(center, s) >= sqrt(8g) - ||center||; once that lower
                # bound exceeds the current k-th best distance, later shells
                # cannot improve the answer.
                if 8.0 * (g + 1) > (np.sqrt(kth) + center_norm) ** 2 + _EPS:
                    break
    if not cand_sites:
        return np.empty((0, 4), dtype=np.int64), np.empty(0, dtype=np.float64)
    all_sites = np.concatenate(cand_sites, axis=0)
    all_d2 = np.concatenate(cand_d2)
    order = np.lexsort((all_sites[:, 3], all_sites[:, 2], all_sites[:, 1],
                        all_sites[:, 0], all_d2))
    take = min(k, all_d2.shape[0])
    return all_sites[order][:take], all_d2[order][:take]
