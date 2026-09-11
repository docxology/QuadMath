"""Tests for omni_numbering: IVM close-packing shell enumeration.

Real numerics only: an independent breadth-first reference (itertools over
the 12 neighbor moves) cross-validates the vectorized enumeration. All
random draws use fixed seeds.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from quadmath.lattice.omni_numbering import (
    MAX_SHELL,
    _row_keys,
    NEIGHBOR_MOVES,
    Quadray,
    cumulative_count,
    generate_shell,
    shell_count,
    site_at_index,
    site_index,
    sites_through_shell,
)


def _bfs_shells(max_shell: int) -> dict:
    """Independent layer-BFS reference: site tuple -> shell index."""
    moves = sorted(set(permutations((2, 1, 1, 0))))

    def norm(q: tuple) -> tuple:
        m = min(q)
        return tuple(x - m for x in q)

    seen = {(0, 0, 0, 0): 0}
    frontier = [(0, 0, 0, 0)]
    for k in range(1, max_shell + 1):
        nxt = set()
        for q in frontier:
            for mv in moves:
                c = norm(tuple(q[i] + mv[i] for i in range(4)))
                if c not in seen:
                    seen[c] = k
                    nxt.add(c)
        frontier = sorted(nxt)
    return seen


# --------------- NEIGHBOR_MOVES ---------------


def test_neighbor_moves_shape_and_values():
    assert NEIGHBOR_MOVES.shape == (12, 4)
    expected = sorted(set(permutations((2, 1, 1, 0))))
    assert [tuple(row) for row in NEIGHBOR_MOVES] == expected


# --------------- shell_count / cumulative_count ---------------


def test_shell_count_formula():
    assert shell_count(0) == 1
    for k in range(1, 9):
        assert shell_count(k) == 10 * k * k + 2


def test_shell_count_scalar_returns_int():
    # Scalar queries return Python ints, not 0-d arrays.
    assert isinstance(shell_count(3), int)
    assert isinstance(cumulative_count(3), int)


def test_shell_count_array():
    out = shell_count(np.array([0, 1, 2, 3]))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 12, 42, 92]


def test_shell_count_rejects_negative():
    with pytest.raises(ValueError):
        shell_count(-1)
    with pytest.raises(ValueError):
        shell_count(np.array([0, -2]))


def test_shell_count_rejects_float():
    with pytest.raises(TypeError):
        shell_count(np.array([1.5]))


def test_cumulative_matches_direct_sum():
    for k in range(0, 7):
        direct = 1 + sum(10 * j * j + 2 for j in range(1, k + 1))
        assert cumulative_count(k) == direct


def test_cumulative_array():
    out = cumulative_count(np.array([0, 1, 2, 3]))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 13, 55, 147]


def test_cumulative_rejects_negative():
    with pytest.raises(ValueError):
        cumulative_count(-3)


def test_cumulative_rejects_float():
    with pytest.raises(TypeError):
        cumulative_count(np.array([2.0]))


# --------------- enumeration vs BFS reference ---------------


def test_sites_through_shell_matches_bfs():
    max_shell = 6
    ref = _bfs_shells(max_shell)
    sites = sites_through_shell(max_shell)
    assert sites.shape == (cumulative_count(max_shell), 4)
    assert {tuple(r) for r in sites} == set(ref)
    # Canonical order: center first, shells grouped, lexicographic within shell.
    assert tuple(sites[0]) == (0, 0, 0, 0)
    counts = [shell_count(k) for k in range(0, max_shell + 1)]
    edges = np.cumsum([0] + counts)
    for k in range(max_shell + 1):
        block = sites[edges[k]:edges[k + 1]]
        assert block.shape[0] == counts[k]
        keys = [tuple(r) for r in block]
        assert keys == sorted(keys)


def test_generate_shell_matches_bfs():
    ref = _bfs_shells(4)
    for k in range(0, 5):
        shell = generate_shell(k)
        assert shell.shape == (shell_count(k), 4)
        assert {tuple(r) for r in shell} == {q for q, g in ref.items() if g == k}


def test_generate_shell_returns_copy():
    shell = generate_shell(1)
    shell[0, 0] = 999
    assert generate_shell(1)[0, 0] != 999


def test_sites_through_shell_range_checks():
    with pytest.raises(ValueError):
        sites_through_shell(-1)
    with pytest.raises(ValueError):
        sites_through_shell(MAX_SHELL + 1)


def test_build_through_shell_cache_hit():
    a = sites_through_shell(3)
    b = sites_through_shell(3)
    assert a is b


def test_site_index_accepts_projective_representatives():
    expected = site_index((2, 1, 1, 0), 3)
    assert expected >= 0
    assert site_index((3, 2, 2, 1), 3) == expected
    assert site_index(Quadray(3, 2, 2, 1), 3) == expected
    # Negative components normalize via q - q.min() to the same site.
    assert site_index((1, 0, 0, -1), 3) == expected
    # (0, 1, 1, 2) is a DIFFERENT shell-1 site (another permutation), not
    # a projective variant of (2, 1, 1, 0).
    assert site_index((0, 1, 1, 2), 3) != expected
    assert site_index((0, 1, 1, 2), 3) == 1


# --------------- site_index / site_at_index ---------------


def test_site_index_roundtrip():
    max_shell = 4
    total = cumulative_count(max_shell)
    assert total == 309
    for idx in range(total):
        site = site_at_index(idx, max_shell)
        assert isinstance(site, Quadray)
        assert site_index(site, max_shell) == idx




def test_site_index_absent_returns_minus_one():
    assert site_index((1, 0, 0, 0), 4) == -1
    assert site_index((2, 0, 0, 0), 6) == -1


def test_site_index_overflow_returns_minus_one():
    # Components at/above 2**16 must not wrap the packed keys.
    assert site_index((65536, 0, 0, 0), 4) == -1
    assert site_index((1 << 17, 1 << 16, 0, 0), 4) == -1


def test_site_at_index_bounds():
    with pytest.raises(IndexError):
        site_at_index(-1, 3)
    with pytest.raises(IndexError):
        site_at_index(cumulative_count(3), 3)


def test_site_at_index_center():
    site = site_at_index(0, 2)
    assert site == Quadray(0, 0, 0, 0)


# --------------- fixed-seed deep invariants ---------------


def test_shell_distance_invariants_deep():
    # Every shell-g site satisfies the lower bound d2 >= 8g (the truncation
    # invariant lattice_search relies on), and the shell maximum is exactly
    # 8g^2 (verified through frequency 8).
    max_shell = 8
    sites = sites_through_shell(max_shell)
    counts = [shell_count(k) for k in range(max_shell + 1)]
    edges = np.cumsum([0] + counts)
    for g in range(1, max_shell + 1):
        rows = sites[edges[g]:edges[g + 1]].astype(np.float64)
        d2 = 4.0 * np.sum(rows * rows, axis=1) - np.sum(rows, axis=1) ** 2
        assert np.all(d2 >= 8.0 * g)
        assert d2.max() == 8.0 * g * g


def test_bfs_and_module_agree_through_shell_8():
    ref = _bfs_shells(8)
    sites = sites_through_shell(8)
    assert sites.shape[0] == len(ref) == cumulative_count(8)
    assert {tuple(r) for r in sites} == set(ref)


def test_row_keys_injective_across_full_component_range():
    # Regression for key packing: four 16-bit fields must fill an int64
    # exactly, so keys stay injective for every component in [0, 65536).
    # (A wider field silently truncates and collides at large components.)
    rng = np.random.default_rng(999)
    sample = rng.integers(0, 65536, size=(500, 4), dtype=np.int64)
    edge = np.array(
        [[0, 0, 0, 0], [65535, 65535, 65535, 65535], [65535, 0, 0, 0],
         [0, 65535, 0, 0], [0, 0, 0, 65535]],
        dtype=np.int64,
    )
    rows = np.unique(np.concatenate([sample, edge]), axis=0)
    keys = _row_keys(rows)
    assert np.unique(keys).size == rows.shape[0]
