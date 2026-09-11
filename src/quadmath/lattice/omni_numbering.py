"""Omnidirectional close-packing numbering on the IVM quadray lattice (Fuller.4D).

In the isotropic vector matrix (IVM), equal spheres pack omnidirectionally
(closest packing, CCP).  Starting from a central sphere, the packing builds
up in consecutive **frequency shells**: the shell of frequency ``k`` (k >= 1)
carries exactly ``10k^2 + 2`` sphere centers, and the center is the lone
site of shell 0.  The first shell is the 12 neighbors -- the permutations of
``(2, 1, 1, 0)`` in quadray coordinates -- forming the cuboctahedron
(vector equilibrium); see Section 3 of the manuscript.

This module enumerates the shell sequence vectorized with NumPy and provides
bidirectional mappings between lattice sites and a canonical global index
(shell-major order: the center first, then each shell in lexicographic
``(a, b, c, d)`` order).

Conventions
-----------
- Sites are canonical quadray integer 4-tuples: non-negative components with
  at least one zero (projective normalization, cf. ``quadmath.core.quadray.Quadray.normalize``).
- The IVM site set reachable from the origin by the 12 neighbor moves
  (all distinct permutations of ``(2, 1, 1, 0)``, each move followed by
  projective normalization) is a strict subset of all normalized integer
  quadrays; membership is therefore determined by enumeration, not by a
  closed-form congruence.  The layer-by-layer expansion below reproduces the
  shell counts ``10k^2 + 2`` exactly (verified through frequency 6 in the
  test suite against an independent breadth-first reference).
"""
from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Tuple, Union

import numpy as np

from quadmath.core.quadray import Quadray

#: Base move of the 12 canonical IVM neighbors (vector equilibrium shell).
_IVM_MOVE_BASE: Tuple[int, int, int, int] = (2, 1, 1, 0)

#: The 12 canonical IVM neighbor moves as an (12, 4) integer array.
NEIGHBOR_MOVES: np.ndarray = np.array(
    sorted(set(permutations(_IVM_MOVE_BASE))), dtype=np.int64
)

#: Per-site bit width used to pack (a, b, c, d) rows into single int64 keys.
#: Four 16-bit fields fill an int64 exactly, so keys are injective for
#: components in [0, 65536); close-packed sites through any supported shell
#: have components <= 2*max_shell, far below that bound.
_KEY_BITS: int = 16

#: Cache of previously built through-shell enumerations.
_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

#: Deepest frequency shell the enumeration supports.  The cumulative site
#: count through shell 32 is about 146k rows -- small, cache-friendly, and
#: deep enough for every query radius ``lattice_search`` accepts.
MAX_SHELL: int = 32

__all__ = [
    "NEIGHBOR_MOVES",
    "MAX_SHELL",
    "shell_count",
    "cumulative_count",
    "generate_shell",
    "sites_through_shell",
    "site_index",
    "site_at_index",
]


def _validate_shells(arr: np.ndarray, name: str = "shell index") -> np.ndarray:
    """Validate an integer array of shell indices.

    Parameters
    - arr: Candidate array of shell indices.
    - name: Name used in the error message.

    Returns
    - np.ndarray: The validated integer array.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be integers, got dtype {arr.dtype}")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def shell_count(k: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Return the number of IVM sites on shell ``k`` of the close packing.

    The center shell (k = 0) holds one sphere; frequency ``k >= 1`` holds
    ``10k^2 + 2`` spheres on the k-th cuboctahedral (vector equilibrium)
    layer of the omnidirectional close packing.

    Parameters
    - k: Shell index or array of shell indices (non-negative integers).

    Returns
    - int or np.ndarray: Shell population(s), matching the shape of ``k``.
    """
    arr = _validate_shells(np.asarray(k))
    counts = np.where(arr == 0, 1, 10 * arr * arr + 2)
    if np.ndim(arr) == 0:
        return int(counts)
    return counts


def cumulative_count(k: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Return the total number of IVM sites through shell ``k`` (inclusive).

    This is the closed form of the cumulative (centered cuboctahedral)
    sequence ``1 + sum_{j=1}^{k} (10j^2 + 2)``:

    ``1 + 2k + 10 k (k+1) (2k+1) / 6``.

    Parameters
    - k: Shell index or array of shell indices (non-negative integers).

    Returns
    - int or np.ndarray: Cumulative site count(s) through shell ``k``.
    """
    arr = _validate_shells(np.asarray(k))
    totals = 1 + 2 * arr + 10 * arr * (arr + 1) * (2 * arr + 1) // 6
    if np.ndim(arr) == 0:
        return int(totals)
    return totals


def _normalize_rows(q: np.ndarray) -> np.ndarray:
    """Project rows to canonical quadray representatives by subtracting the row minimum.

    Parameters
    - q: (n, 4) integer array of quadray components.

    Returns
    - np.ndarray: (n, 4) integer array with each row's minimum component zero.
    """
    return q - q.min(axis=1, keepdims=True)


def _row_keys(q: np.ndarray) -> np.ndarray:
    """Pack (n, 4) non-negative integer rows into unique int64 keys.

    Parameters
    - q: (n, 4) integer array with non-negative entries (each < 2**_KEY_BITS).

    Returns
    - np.ndarray: (n,) int64 keys, injective for components below 2**_KEY_BITS.
    """
    key = np.zeros(q.shape[0], dtype=np.int64)
    for col in range(4):
        key = (key << _KEY_BITS) | q[:, col].astype(np.int64)
    return key


def _build_through_shell(max_shell: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the canonical site enumeration through shell ``max_shell``.

    Expands the packing layer by layer with vectorized NumPy operations:
    each layer is produced by adding all 12 neighbor moves to the previous
    frontier at once, re-normalizing rows, and removing sites seen on
    earlier shells.

    Parameters
    - max_shell: Highest frequency shell to enumerate (>= 0).

    Returns
    - tuple of (np.ndarray, np.ndarray, np.ndarray):
        - sites: (N, 4) int64 array in canonical order (shell-major,
          lexicographic within each shell).
        - shell_of: (N,) int64 shell index of each site.
        - offsets: (max_shell + 2,) int64 array; ``sites[shell_offsets[k]:shell_offsets[k+1]]``
          are the shell ``k`` sites in lexicographic order.
    """
    if max_shell < 0:
        raise ValueError("max_shell must be non-negative")
    if max_shell > MAX_SHELL:
        raise ValueError(f"max_shell must be <= MAX_SHELL={MAX_SHELL}, got {max_shell}")
    cached = _CACHE.get(max_shell)
    if cached is not None:
        return cached
    sites_list: List[np.ndarray] = [np.zeros((1, 4), dtype=np.int64)]
    shell_of_list: List[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    offsets: List[int] = [0, 1]
    seen_keys = _row_keys(sites_list[0])
    frontier = sites_list[0]
    for k in range(1, max_shell + 1):
        cand = (frontier[:, None, :] + NEIGHBOR_MOVES[None, :, :]).reshape(-1, 4)
        cand = np.unique(_normalize_rows(cand), axis=0)
        cand = cand[~np.isin(_row_keys(cand), seen_keys)]
        cand = cand[np.lexsort((cand[:, 3], cand[:, 2], cand[:, 1], cand[:, 0]))]
        seen_keys = np.concatenate([seen_keys, _row_keys(cand)])
        sites_list.append(cand)
        shell_of_list.append(np.full(cand.shape[0], k, dtype=np.int64))
        offsets.append(offsets[-1] + cand.shape[0])
        frontier = cand
    sites = np.concatenate(sites_list, axis=0)
    shell_of = np.concatenate(shell_of_list, axis=0)
    offsets_arr = np.array(offsets, dtype=np.int64)
    _CACHE.clear()
    _CACHE[max_shell] = (sites, shell_of, offsets_arr)
    return sites, shell_of, offsets_arr


def sites_through_shell(max_shell: int) -> np.ndarray:
    """Return all IVM sites through shell ``max_shell`` in canonical order.

    Parameters
    - max_shell: Highest frequency shell to include (>= 0).

    Returns
    - np.ndarray: (N, 4) int64 array of quadray components; row order is
      shell 0 (the center) first, then shells 1..max_shell, each shell in
      lexicographic (a, b, c, d) order.
    """
    sites, _, _ = _build_through_shell(max_shell)
    return sites


def generate_shell(k: int) -> np.ndarray:
    """Generate the sites of shell ``k`` of the omnidirectional close packing.

    Parameters
    - k: Shell index (>= 0); k = 0 returns only the center site.

    Returns
    - np.ndarray: (shell_count(k), 4) int64 array of quadray components in
      lexicographic (a, b, c, d) order.
    """
    sites, shell_of, _ = _build_through_shell(k)
    return sites[shell_of == k].copy()


def site_index(site: Union[Quadray, Tuple[int, int, int, int], List[int]],
               max_shell: int = 6) -> int:
    """Return the canonical global index of an IVM site, or -1 if absent.

    The global index enumerates ``sites_through_shell(max_shell)``: index 0
    is the center, then shell 1 sites in lexicographic order, and so on.
    The input is normalized (translated by -(k, k, k, k)) before lookup, so
    any representative of the projective class is accepted.

    Parameters
    - site: Quadray vector or integer 4-sequence.
    - max_shell: Depth of the enumeration to search (>= 0).

    Returns
    - int: Global index into ``sites_through_shell(max_shell)``, or -1 if
      the site is not an IVM site within that depth.
    """
    sites, _, _ = _build_through_shell(max_shell)
    q = np.asarray(
        site.as_tuple() if isinstance(site, Quadray) else tuple(site),
        dtype=np.int64,
    ).reshape(4)
    q = q - q.min()
    if int(q.max()) >= (1 << _KEY_BITS):
        return -1
    hits = np.flatnonzero(np.all(sites == q, axis=1))
    return int(hits[0]) if hits.size else -1


def site_at_index(index: int, max_shell: int = 6) -> Quadray:
    """Return the IVM site at a canonical global index.

    Inverse of ``site_index`` for sites present in the enumeration.

    Parameters
    - index: Global index in ``[0, cumulative_count(max_shell))``.
    - max_shell: Depth of the enumeration (>= 0).

    Returns
    - Quadray: The canonical site at the given index.

    Raises
    - IndexError: If ``index`` is out of range for the enumeration.
    """
    sites, _, _ = _build_through_shell(max_shell)
    if index < 0 or index >= sites.shape[0]:
        raise IndexError(f"site index {index} out of range for max_shell={max_shell}")
    row = sites[index]
    return Quadray(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
