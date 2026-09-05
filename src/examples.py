from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import List, Iterable

from quadray import Quadray, DEFAULT_EMBEDDING, to_xyz, integer_tetra_volume, ace_tetravolume_5x5
from nelder_mead_quadray import nelder_mead_quadray, SimplexState


def _ivm_neighbor_permutations() -> List[Quadray]:
    """Shared implementation: the 12 distinct permutations of (2,1,1,0), sorted.

    Single source of truth for `example_ivm_neighbors` and
    `example_cuboctahedron_neighbors`, which expose the same set under the two
    roles documented in the manuscript (neighbor-move set vs. shell positions).
    Sorted output keeps results deterministic regardless of set ordering.
    """
    return [Quadray(*perm) for perm in sorted(set(permutations((2, 1, 1, 0))))]


def example_ivm_neighbors() -> List[Quadray]:
    """Return the 12 nearest IVM neighbors as permutations of {2,1,1,0} (Fuller.4D).

    Same set as `example_cuboctahedron_neighbors`, in its "neighbor-move set
    around any lattice point" role; used throughout the paper to illustrate
    local neighborhoods in the quadray lattice. Both public names share one
    implementation for deterministic, single-source behavior.
    """
    return _ivm_neighbor_permutations()


def example_volume() -> Fraction:
    """Return the exact IVM tetravolume of the primitive lattice tetrahedron.

    The tetrahedron spanned by (0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)
    has projected determinant 1, hence exact IVM volume Fraction(1, 4).
    (The unit IVM tetrahedron — origin plus three (2,1,1,0)-type neighbors —
    has volume 1; this example intentionally illustrates the primitive cell.)
    """
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(1, 0, 0, 0)
    p2 = Quadray(0, 1, 0, 0)
    p3 = Quadray(0, 0, 1, 0)
    return integer_tetra_volume(p0, p1, p2, p3)


def example_optimize() -> SimplexState:
    """Run Nelder–Mead over integer quadrays for a simple convex objective (Fuller.4D).

    Returns
    - SimplexState: Final state containing ordered vertices, values, volume, and history.
    """
    # Simple convex bowl objective over integer quadrays
    def f(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        return (x - 1.0) ** 2 + (y + 0.5) ** 2 + (z - 0.25) ** 2

    initial = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(1, 1, 0, 0)]
    state = nelder_mead_quadray(f, initial)
    return state


def example_cuboctahedron_neighbors() -> List[Quadray]:
    """Return twelve-around-one IVM neighbors (vector equilibrium shell).

    Same set as `example_ivm_neighbors` — the 12 distinct permutations of
    (2,1,1,0) — exposed under its "shell positions around one lattice point"
    role, as documented in the manuscript's cuboctahedron sections. Both names
    are kept as documented public API; the implementation is shared.
    """
    return _ivm_neighbor_permutations()


def example_cuboctahedron_vertices_xyz() -> List[tuple[float, float, float]]:
    """Return XYZ coordinates for the twelve-around-one neighbors.

    Uses the default symmetric embedding to map Quadray to R^3, suitable for
    plotting or distance analysis.
    """
    neighbors = example_cuboctahedron_neighbors()
    return [to_xyz(q, DEFAULT_EMBEDDING) for q in neighbors]


def example_partition_tetra_volume(
    mu: Iterable[int], s: Iterable[int], a: Iterable[int], psi: Iterable[int]
) -> Fraction:
    """Construct a tetrahedron from the four-fold partition and return tetravolume (Fuller.4D).

    Parameters
    - mu: Internal state mapped to Quadray A-like emphasis
    - s:  Sensory state
    - a:  Active state
    - psi: External state

    Notes
    - Inputs are 4-tuples of nonnegative integers. The 5x5 determinant is
      invariant to adding (k,k,k,k) to each vertex, so explicit normalization
      is not required for volume.
    """
    p_mu = Quadray(*mu)
    p_s = Quadray(*s)
    p_a = Quadray(*a)
    p_psi = Quadray(*psi)
    return ace_tetravolume_5x5(p_mu, p_s, p_a, p_psi)
