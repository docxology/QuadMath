"""Tests for lattice_search: nearest-site queries on the IVM quadray lattice.

Real numerics only: brute-force references over the full enumeration
cross-validate every query path, and the fast squared-distance identity is
cross-checked against direct embedding coordinates. All random draws use
fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from quadmath.lattice.lattice_search import (
    _build_through_shell as _search_build,  # re-exported via omni_numbering
    _max_shell_for,
    nearest,
    squared_distance,
    within_radius,
)
from quadmath.lattice.omni_numbering import Quadray, cumulative_count, sites_through_shell
from quadmath.core.quadray import DEFAULT_EMBEDDING

_E = np.asarray(DEFAULT_EMBEDDING, dtype=np.float64)


def _brute_force_within(center: np.ndarray, R: float, max_shell: int):
    """Reference: filter the whole enumeration with the exact identity."""
    sites = sites_through_shell(max_shell)
    d2 = squared_distance(center.astype(np.float64), sites)
    keep = np.flatnonzero(d2 <= R * R)
    return sites[keep], d2[keep]


# --------------- squared_distance: identity vs direct embedding ---------------


def test_squared_distance_matches_direct_embedding():
    rng = np.random.default_rng(12345)
    sites = sites_through_shell(3)
    for _ in range(40):
        p = rng.normal(scale=2.0, size=4)
        row = sites[rng.integers(sites.shape[0])]
        got = float(squared_distance(p, row.reshape(1, 4))[0])
        direct = float(np.sum(((row.astype(np.float64) - p) @ _E.T) ** 2))
        assert abs(got - direct) < 1e-9


def test_squared_distance_integer_exact():
    sites = sites_through_shell(2)
    d2 = squared_distance(np.zeros(4), sites)
    # Shell-1 sites (perms of (2,1,1,0)) sit at exactly squared distance 8.
    assert np.all(d2[1:13] == 8.0)


# --------------- _max_shell_for bound ----------


def test_max_shell_for_zero_center():
    assert _max_shell_for(0.0, 0.0) >= 0


def test_max_shell_for_growth():
    assert _max_shell_for(0.0, 10.0) > _max_shell_for(0.0, 1.0)
    assert _max_shell_for(5.0, 2.0) > _max_shell_for(0.0, 2.0)


# --------------- within_radius ---------------


def test_within_radius_origin_small_shells():
    sites, d2 = within_radius((0, 0, 0, 0), 3.0)
    assert sites.shape == (13, 4)  # center + 12 shell-1 sites
    assert tuple(sites[0]) == (0, 0, 0, 0)
    assert np.all(d2 <= 9.0)
    assert np.all(np.diff(d2) >= 0)


def test_within_radius_matches_brute_force():
    rng = np.random.default_rng(777)
    for _ in range(15):
        center = rng.normal(scale=1.5, size=4)
        R = float(rng.uniform(0.5, 3.5))
        got_sites, got_d2 = within_radius(center, R)
        exp_sites, exp_d2 = _brute_force_within(center, R, 8)
        order = np.lexsort((exp_sites[:, 3], exp_sites[:, 2], exp_sites[:, 1],
                            exp_sites[:, 0], exp_d2))
        exp_sites = exp_sites[order]
        exp_d2 = exp_d2[order]
        assert got_sites.shape == exp_sites.shape
        assert np.allclose(got_d2, exp_d2)
        assert np.array_equal(got_sites, exp_sites)


def test_within_radius_accepts_quadray_and_normalizes():
    sites_a, d2_a = within_radius(Quadray(2, 1, 1, 0), 0.5)
    sites_b, d2_b = within_radius((5, 4, 4, 3), 0.5)  # same projective class
    assert np.array_equal(sites_a, sites_b)
    assert np.allclose(d2_a, d2_b)
    assert tuple(sites_a[0]) == (2, 1, 1, 0)
    assert d2_a[0] == 0.0


def test_within_radius_zero_radius_returns_site():
    sites, d2 = within_radius((2, 1, 1, 0), 0.0)
    assert sites.shape == (1, 4)
    assert tuple(sites[0]) == (2, 1, 1, 0)
    assert d2[0] == 0.0


def test_within_radius_empty_region():
    # (0.3, 0, 0, 0): nearest site is sqrt(0.27) ~ 0.52 away, so R = 0.1
    # contains no lattice site.
    sites, d2 = within_radius((0.3, 0.0, 0.0, 0.0), 0.1)
    assert sites.shape == (0, 4)
    assert d2.shape == (0,)


def test_within_radius_projective_center_is_origin():
    # (0.5, 0.5, 0.5, 0.5) is projectively the origin; at R = 3.0 the
    # answer must equal the origin query: center + 12 shell-1 sites.
    sites, d2 = within_radius((0.5, 0.5, 0.5, 0.5), 3.0)
    assert sites.shape == (13, 4)
    assert tuple(sites[0]) == (0, 0, 0, 0)
    assert d2[0] == 0.0
    assert np.all(d2[1:13] == 8.0)


def test_within_radius_rejects_bad_radius():
    with pytest.raises(TypeError):
        within_radius((0, 0, 0, 0), "big")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        within_radius((0, 0, 0, 0), -1.0)
    with pytest.raises(ValueError):
        within_radius((0, 0, 0, 0), float("nan"))


def test_within_radius_rejects_bad_center():
    with pytest.raises(ValueError):
        within_radius((1, 2, 3), 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        within_radius((1, 2, 3, float("nan")), 1.0)
    with pytest.raises(TypeError):
        within_radius(None, 1.0)  # type: ignore[arg-type]


def test_within_radius_caps_enumeration_depth():
    # norm((2,2,0,0)) = 4; R = 50 needs shell 366 > MAX_SHELL.
    with pytest.raises(ValueError, match="MAX_SHELL"):
        within_radius((2, 2, 0, 0), 50.0)




# --------------- nearest ---------------


def test_nearest_origin_k1():
    sites, d2 = nearest((0, 0, 0, 0), 1.0, k=1)
    assert sites.shape == (1, 4)
    assert tuple(sites[0]) == (0, 0, 0, 0)
    assert d2[0] == 0.0


def test_within_radius_r15_at_origin_count():
    # R = 15 at the origin: every shell g with max d2 = 8g^2 <= 225 is fully
    # inside (g <= 5); the count below is verified empirically.
    sites, d2 = within_radius((0, 0, 0, 0), 15.0)
    assert sites.shape[0] == 935
    assert np.all(d2 <= 225.0)


def test_nearest_origin_k13():
    sites, d2 = nearest((0, 0, 0, 0), 3.0, k=13)
    assert sites.shape == (13, 4)
    assert d2[0] == 0.0  # center first
    assert np.all(d2[1:13] == 8.0)
    assert {tuple(r) for r in sites} == set(map(tuple, sites_through_shell(1)))


def test_nearest_k_exceeds_hits():
    sites, _ = nearest((0, 0, 0, 0), 1.0, k=5)
    assert sites.shape == (1, 4)  # only the center is within radius 1
    assert tuple(sites[0]) == (0, 0, 0, 0)


def test_nearest_empty_region():
    sites, d2 = nearest((0.3, 0.0, 0.0, 0.0), 0.1, k=3)
    assert sites.shape == (0, 4)
    assert d2.shape == (0,)


def test_nearest_matches_brute_force():
    rng = np.random.default_rng(4242)
    for _ in range(12):
        center = rng.normal(scale=1.2, size=4)
        R = float(rng.uniform(1.0, 3.0))
        k = int(rng.integers(1, 8))
        got_sites, got_d2 = nearest(center, R, k=k)
        exp_sites, exp_d2 = _brute_force_within(center, R, 8)
        order = np.lexsort((exp_sites[:, 3], exp_sites[:, 2], exp_sites[:, 1],
                            exp_sites[:, 0], exp_d2))
        exp_sites = exp_sites[order][:k]
        exp_d2 = exp_d2[order][:k]
        assert np.array_equal(got_sites, exp_sites)
        assert np.allclose(got_d2, exp_d2)


def test_nearest_early_stop_full_shells():
    # k = 55 at R = sqrt(32 + eps) covers center + full shells 1-2
    # (1 + 12 + 42 = 55); at R = 5.0 only 30 of 42 shell-2 sites fit (43).
    sites, d2 = nearest((0, 0, 0, 0), 5.7, k=55)
    assert sites.shape[0] == 55
    assert d2.max() <= 32.0
    sites_r5, d2_r5 = nearest((0, 0, 0, 0), 5.0, k=55)
    assert sites_r5.shape[0] == 43
    assert d2_r5.max() <= 25.0


def test_nearest_rejects_bad_k():
    with pytest.raises(ValueError):
        nearest((0, 0, 0, 0), 2.0, k=0)
    with pytest.raises(ValueError):
        nearest((0, 0, 0, 0), 2.0, k=True)
    with pytest.raises(ValueError):
        nearest((0, 0, 0, 0), 2.0, k=1.5)  # type: ignore[arg-type]


def test_nearest_caps_enumeration_depth():
    with pytest.raises(ValueError, match="MAX_SHELL"):
        nearest((3, 3, 0, 0), 40.0, k=1)


# --------------- cross-function consistency ---------------


def test_nearest_and_within_radius_agree():
    rng = np.random.default_rng(31337)
    for _ in range(8):
        center = rng.normal(scale=1.0, size=4)
        R = float(rng.uniform(1.0, 3.0))
        w_sites, w_d2 = within_radius(center, R)
        if w_sites.shape[0] == 0:
            continue
        n_sites, n_d2 = nearest(center, R, k=w_sites.shape[0])
        assert np.array_equal(w_sites, n_sites)
        assert np.allclose(w_d2, n_d2)


def test_quadray_object_input():
    sites, d2 = within_radius(Quadray(0, 0, 0, 0), 3.0)
    assert sites.shape == (13, 4)
    assert d2[0] == 0.0


def test_reexport_consistency():
    # The _build_through_shell name visible through lattice_search is the
    # same function object as omni_numbering's.
    from quadmath.lattice.omni_numbering import _build_through_shell as _omni_build

    assert _search_build is _omni_build
