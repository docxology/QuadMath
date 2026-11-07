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
    nearest,
    squared_distance,
    within_radius,
)
from quadmath.lattice.omni_numbering import Quadray, sites_through_shell
from quadmath.core.quadray import DEFAULT_EMBEDDING

_E = np.asarray(DEFAULT_EMBEDDING, dtype=np.float64)


def _brute_force_within(center: np.ndarray, R: float, max_shell: int):
    """Reference: filter the whole enumeration in independent XYZ coordinates."""
    sites = sites_through_shell(max_shell)
    xyz = sites @ _E.T
    center_xyz = center @ _E.T
    d2 = np.sum((xyz - center_xyz) ** 2, axis=1)
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



@pytest.mark.parametrize("gauge", [2**40, 2**60, -(2**60), 2**100, -(2**100)])
def test_squared_distance_preserves_large_independent_integer_gauges(gauge):
    center = (2, 1, 1, 0)
    rows = [(0, 0, 0, 0), (2, 1, 1, 0), (1, 2, 1, 0), (0, 2, 4, 2)]
    p = np.asarray([v + gauge for v in center])
    sites = np.asarray([[v - gauge for v in row] for row in rows])
    expected = []
    for row in rows:
        a, b, c, d = (s - p for s, p in zip(row, center))
        xyz = (a - b - c + d, a + b - c - d, a - b + c - d)
        expected.append(sum(v * v for v in xyz))
    np.testing.assert_array_equal(squared_distance(p, sites), expected)


def test_squared_distance_large_float_gauge_cannot_erase_separation():
    p = np.full(4, float(2**40))
    sites = np.array([[2, 1, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(squared_distance(p, sites), [8.0, 0.0])


def test_squared_distance_subtracts_integer_positions_before_float_conversion():
    k = 2**60
    p = np.array([0, k, k, k], dtype=np.int64)
    sites = np.array([[0, k + 1, k, k]], dtype=np.int64)
    # The XYZ displacement is (-1, 1, -1), even though float(k+1) == float(k).
    np.testing.assert_array_equal(squared_distance(p, sites), [3.0])


@pytest.mark.parametrize("gauge", [2**40, 2**60, -(2**60), 2**100, -(2**100)])
@pytest.mark.parametrize("center", [(0, 0, 0, 0), (2, 1, 1, 0)])
def test_zero_radius_queries_remove_integer_gauge_before_float(gauge, center):
    shifted = tuple(v + gauge for v in center)
    for query in (within_radius, nearest):
        sites, d2 = query(shifted, 0.0)
        np.testing.assert_array_equal(sites, [center])
        np.testing.assert_array_equal(d2, [0.0])


def test_large_gauge_queries_match_independent_spatial_oracle():
    center = np.array([2.25, 1.0, 1.5, 0.0])
    gauge = float(2**40)
    radius = 3.5
    exp_sites, exp_d2 = _brute_force_within(center, radius, 8)
    order = np.lexsort((exp_sites[:, 3], exp_sites[:, 2], exp_sites[:, 1],
                       exp_sites[:, 0], exp_d2))
    exp_sites, exp_d2 = exp_sites[order], exp_d2[order]
    got_sites, got_d2 = within_radius(center + gauge, radius)
    np.testing.assert_array_equal(got_sites, exp_sites)
    np.testing.assert_allclose(got_d2, exp_d2, rtol=1e-14, atol=0.0)
    got_sites, got_d2 = nearest(center + gauge, radius, k=7)
    np.testing.assert_array_equal(got_sites, exp_sites[:7])
    np.testing.assert_allclose(got_d2, exp_d2[:7], rtol=1e-14, atol=0.0)


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
    # Individually finite inputs can have unrepresentable gauge-free deltas.
    with pytest.raises(ValueError, match="finite"):
        within_radius((-1e308, 1e308, 0.0, 0.0), 0.0)


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




def test_float32_gauge_subtraction_preserves_resolved_displacement():
    center = np.array([-2**24, 1, 0, 0], dtype=np.float32)
    sites = np.array([[-2**24, 0, 0, 0]], dtype=np.float32)
    # The original float32 arrays retain the unit B-axis displacement.
    # Subtracting their minima in float32 would erase it before widening.
    np.testing.assert_array_equal(squared_distance(center, sites), [3.0])
