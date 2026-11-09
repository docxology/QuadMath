"""Tests for static IVM field learning (quadmath/lattice/ivm_field.py).

All tests use real numerical examples on the IVM lattice with fixed RNG
seeds — no mocks, no ML frameworks.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from quadmath.lattice.ivm_field import (
    IVM_NEIGHBOR_STEPS,
    IVMField,
    TetrahedronFit,
    shell_ball_sites,
    fit_geometry,
    is_ivm_site,
    quadray_shell_norm,
    shell_cardinalities,
    shell_sites,
)
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, to_xyz

# 3x4 embedding as an array, for vectorized XYZ computation in tests.
E = np.array(DEFAULT_EMBEDDING, dtype=float)


def _xyz(q: Quadray) -> np.ndarray:
    """Embedded XYZ coordinates of a quadray (numpy vector)."""
    return E @ np.array(q.as_tuple(), dtype=float)


# --------------- shell enumeration ---------------


def test_shell_cardinalities_reproduce_cuboctahedral_numbers():
    assert shell_cardinalities(4) == [1, 12, 42, 92, 162]


def test_shell_cardinality_matches_10k_squared_plus_2():
    cards = shell_cardinalities(6)
    assert cards[0] == 1
    for k in range(1, 7):
        assert cards[k] == 10 * k * k + 2


def test_shell_sites_k0_is_origin():
    assert shell_sites(0) == [Quadray(0, 0, 0, 0)]


def test_shell_sites_k1_are_twelve_neighbors():
    sites = shell_sites(1)
    assert len(sites) == 12
    # Every shell-1 site has quadray norm 2 and is a permutation of (2,1,1,0)
    for q in sites:
        assert quadray_shell_norm(q) == 2
        assert sorted(q.as_tuple()) == [0, 1, 1, 2]
    # Deterministic ordering
    assert sites == sorted(sites, key=lambda q: q.as_tuple())


def test_shell_sites_all_ivm_sites():
    for q in shell_sites(3):
        assert is_ivm_site(q)


def test_shell_sites_negative_raises():
    with pytest.raises(ValueError):
        shell_sites(-1)


def test_ball_sites_negative_raises():
    with pytest.raises(ValueError):
        shell_ball_sites(-2)


def test_ball_sites_shell_partition():
    ball = shell_ball_sites(2)
    assert len(ball) == 1 + 12 + 42
    # Ordered by shell, lexicographic within shell
    norms = [quadray_shell_norm(q) for q in ball]
    assert norms == sorted(norms)
    assert len(set(ball)) == len(ball)


def test_shell_cardinalities_negative_raises():
    with pytest.raises(ValueError):
        shell_cardinalities(-2)


def test_is_ivm_site_membership():
    # (2,1,1,0) has sum 4 -> IVM site; (1,0,0,0) has sum 1 -> tetrahedral void;
    # (1,1,0,0) has sum 2 -> octahedral void.
    assert is_ivm_site(Quadray(2, 1, 1, 0))
    assert is_ivm_site(Quadray(0, 0, 0, 0))
    assert is_ivm_site(Quadray(4, 2, 2, 0))
    assert not is_ivm_site(Quadray(1, 0, 0, 0))
    assert not is_ivm_site(Quadray(1, 1, 0, 0))


def test_quadray_shell_norm_values():
    assert quadray_shell_norm(Quadray(0, 0, 0, 0)) == 0
    assert quadray_shell_norm(Quadray(2, 1, 1, 0)) == 2
    # Shell norm is a property of the projective class: (3,2,2,1) ~ (2,1,1,0)
    assert quadray_shell_norm(Quadray(3, 2, 2, 1)) == 2
    assert quadray_shell_norm(Quadray(4, 2, 2, 0)) == 4


def test_quadray_shell_norm_rejects_void_sites():
    with pytest.raises(ValueError):
        quadray_shell_norm(Quadray(1, 0, 0, 0))
    with pytest.raises(ValueError):
        quadray_shell_norm(Quadray(1, 1, 0, 0))


def test_neighbor_steps_are_twelve_normalized():
    assert len(IVM_NEIGHBOR_STEPS) == 12
    for step in IVM_NEIGHBOR_STEPS:
        assert min(step.as_tuple()) == 0
        assert quadray_shell_norm(step) == 2


# --------------- IVMField construction ---------------


def test_lattice_ball_sizes_and_adjacency():
    field = IVMField.lattice_ball(1)
    assert len(field.sites) == 13
    assert field.radius == 1
    # Origin has all 12 neighbors inside the ball; each shell-1 site has
    # 5 of its 12 neighbors inside (the rest step to shell 2)
    origin = field.site_index[Quadray(0, 0, 0, 0)]
    assert len(field.adjacency[origin]) == 12
    for i, _q in enumerate(field.sites):
        if i == origin:
            continue
        assert len(field.adjacency[i]) == 5
        assert i not in field.adjacency[i]  # no self-loops


def test_lattice_ball_negative_radius_raises():
    with pytest.raises(ValueError):
        IVMField.lattice_ball(-1)


def test_dataclass_defaults():
    field = IVMField(radius=0, sites=())
    assert field.site_index == {}
    assert field.adjacency == ()
    assert field.values.shape == (0,)


# --------------- predict / score ---------------


def test_predict_accepts_any_projective_representative():
    field = IVMField.lattice_ball(1)
    field.values[:] = 0.0
    idx = field.site_index[Quadray(2, 1, 1, 0)]
    field.values[idx] = 3.5
    # (3,2,2,1) normalizes to (2,1,1,0): same lattice site
    assert field.predict(Quadray(3, 2, 2, 1)) == 3.5
    assert field.predict(Quadray(2, 1, 1, 0)) == 3.5


def test_predict_outside_ball_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.predict(Quadray(4, 2, 2, 0))


def test_score_mse_matches_manual_computation():
    field = IVMField.lattice_ball(1)
    a = Quadray(2, 1, 1, 0)
    b = Quadray(1, 2, 0, 1)
    field.values[field.site_index[a]] = 1.0
    field.values[field.site_index[b]] = 3.0
    mse = field.score([a, b], [2.0, 1.0])
    assert abs(mse - ((1.0 - 2.0) ** 2 + (3.0 - 1.0) ** 2) / 2.0) < 1e-12


def test_score_length_mismatch_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.score([Quadray(2, 1, 1, 0)], [1.0, 2.0])


def test_score_empty_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.score([], [])


# --------------- learn ---------------


def test_learn_approximates_linear_field_with_weak_regularization():
    # Soft fitting can approximate this field with tiny lambda, a narrow
    # kernel, the full boundary AND additional interior observations.
    # This is not boundary-only hard harmonic extension.
    radius = 3
    truth = {
        q: 2.0 + 0.75 * _xyz(q)[0] - 0.5 * _xyz(q)[1] + 0.25 * _xyz(q)[2]
        for q in shell_ball_sites(radius)
    }
    field = IVMField.lattice_ball(radius)
    rng = np.random.default_rng(42)
    boundary = [q for q in field.sites if quadray_shell_norm(q) == 2 * radius]
    interior = [q for q in field.sites if q not in set(boundary)]
    obs = boundary + [q for q in interior if rng.random() < 0.6]
    returned = field.learn(obs, [truth[q] for q in obs], lam=1e-6, kernel_width=0.2)
    assert returned is field
    max_err = max(abs(field.predict(q) - truth[q]) for q in field.sites)
    assert max_err < 1e-5


def test_learn_denoises_noisy_observations():
    # Smooth weakly-curved field observed at half the sites with N(0, 0.3)
    # noise; the learned field must beat the raw observation MSE.
    radius = 3
    truth = {q: 0.5 - 0.01 * float(_xyz(q) @ _xyz(q)) for q in shell_ball_sites(radius)}
    rng = np.random.default_rng(7)
    obs = [q for q in shell_ball_sites(radius) if rng.random() < 0.5]
    noisy = {q: truth[q] + rng.normal(0.0, 0.3) for q in obs}
    raw_mse = float(np.mean([(noisy[q] - truth[q]) ** 2 for q in obs]))
    field = IVMField.lattice_ball(radius)
    field.learn(obs, [noisy[q] for q in obs], lam=0.05, kernel_width=1.0)
    learned_mse = float(np.mean([(field.predict(q) - truth[q]) ** 2 for q in shell_ball_sites(radius)]))
    assert learned_mse < raw_mse
    assert learned_mse < 0.05


@pytest.mark.parametrize("width", [0.15, 0.01, np.nextafter(0.0, 1.0)])
def test_learn_narrow_kernel_preserves_constant_with_floored_confidence(width):
    field = IVMField.lattice_ball(2)
    field.learn([field.sites[0]], [1.0], lam=0.1, kernel_width=width)
    np.testing.assert_array_equal(field.values, np.ones(len(field.sites)))


@pytest.mark.parametrize(
    "value", [0.0, np.nextafter(0.0, 1.0), np.finfo(float).max, -np.finfo(float).max]
)
def test_learn_preserves_constant_observations_at_float_extremes(value):
    field = IVMField.lattice_ball(1)
    field.values[:] = np.arange(len(field.sites)) + 1.0
    field.learn(field.sites[:2], [value, value], lam=0.1)
    np.testing.assert_array_equal(field.values, np.full(len(field.sites), value))


def test_learn_ill_conditioned_response_is_accurate_or_atomic():
    sites = (Quadray(0, 0, 0, 0), IVM_NEIGHBOR_STEPS[0])
    field = IVMField(
        radius=1, sites=sites, site_index={q: i for i, q in enumerate(sites)},
        adjacency=((1,), (0,)), values=np.array([7.0, 9.0]),
    )
    previous = field.values
    strength = 2**28
    # A finite positive-definite system can lose unit response mass in the
    # dense double solve. Accept an accurate fit or an atomic numerical rejection.
    try:
        field.learn(sites, [0.0, 1.0], lam=float(strength))
    except np.linalg.LinAlgError:
        assert field.values is previous
        np.testing.assert_array_equal(field.values, [7.0, 9.0])
    else:
        denominator = 1 + 2 * strength
        oracle = [Fraction(strength, denominator), Fraction(1 + strength, denominator)]
        np.testing.assert_allclose(
            field.values, [float(value) for value in oracle], rtol=2e-15, atol=0.0,
        )


@pytest.mark.parametrize("width", [0.01, np.nextafter(0.0, 1.0)])
@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_learn_nearest_observation_limit_keeps_absolute_confidence_floor(width, lam):
    field = IVMField.lattice_ball(1)
    # Every shell-1 site is one hop from the origin; exactly those also
    # adjacent to site 1 tie between the two observations.
    target = np.zeros(len(field.sites))
    target[list(field.adjacency[1])] = 1.0
    target[0], target[1] = 0.0, 2.0
    field.learn(field.sites[:2], [0.0, 2.0], lam=lam, kernel_width=width)
    if lam == 0.0:
        np.testing.assert_array_equal(field.values, target)
    else:
        confidence = np.full(len(field.sites), 1e-12)
        confidence[:2] = 1.0
        residual = [
            confidence[i] * (field.values[i] - target[i])
            + lam * sum(field.values[i] - field.values[j] for j in neighbors)
            for i, neighbors in enumerate(field.adjacency)
        ]
        np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2e-15)


def test_learn_extreme_nonconstant_targets_preserve_scale_and_soft_fit():
    ordinary = IVMField.lattice_ball(1)
    extreme = IVMField.lattice_ball(1)
    targets = np.array([1.0, -0.5, 0.25])
    indices = [0, 1, 7]
    scale = np.finfo(float).max
    ordinary.learn([ordinary.sites[i] for i in indices], targets, lam=0.1)
    extreme.learn([extreme.sites[i] for i in indices], targets * scale, lam=0.1)
    assert np.all(np.isfinite(extreme.values))
    np.testing.assert_allclose(extreme.values / scale, ordinary.values, atol=3e-15)


@pytest.mark.parametrize(
    "values",
    [[1e16, 1.0], [np.finfo(float).max, 1.0], [1.0, 1e-15], [1.0, 1e-17]],
)
def test_learn_zero_regularization_retains_mixed_magnitude_observations(values):
    field = IVMField.lattice_ball(1)
    field.learn(field.sites[:2], values, lam=0.0)
    np.testing.assert_array_equal(field.values[:2], values)


def test_learn_kernel_target_accumulates_cancelling_observations():
    field = IVMField.lattice_ball(1)
    # These three sites are all one hop from the unobserved origin. The data
    # term there is their arithmetic mean, including the small third value.
    values = [1e16, -1e16, 1.0]
    field.learn(field.sites[1:4], values, lam=0.0)
    expected = float(sum(Fraction(value) for value in values) / 3)
    assert field.values[0] == pytest.approx(expected, rel=2e-15, abs=0.0)


@pytest.mark.parametrize(
    "values, lam",
    [
        ([1e16, 1.0], 1e-20),
        ([1e16, 1.0], 1e-12),
        ([1.0, 1e-17], 1e-20),
        ([np.finfo(float).max, 1.0], 1e-310),
    ],
)
def test_learn_weak_coupling_matches_exact_edge_objective(values, lam):
    sites = (Quadray(0, 0, 0, 0), IVM_NEIGHBOR_STEPS[0])
    field = IVMField(
        radius=1, sites=sites, site_index={q: i for i, q in enumerate(sites)},
        adjacency=((1,), (0,)), values=np.zeros(2),
    )
    field.learn(sites, values, lam=lam)
    a, b = (Fraction(float(value)) for value in values)
    strength = Fraction(lam)
    # Exact stationary point of (u-a)^2 + (v-b)^2 + lam*(u-v)^2.
    oracle = [
        ((1 + strength) * a + strength * b) / (1 + 2 * strength),
        (strength * a + (1 + strength) * b) / (1 + 2 * strength),
    ]
    np.testing.assert_allclose(field.values, [float(value) for value in oracle], rtol=2e-15, atol=0.0)

    def objective(u, v):
        return (u - a) ** 2 + (v - b) ** 2 + strength * (u - v) ** 2

    u, v = (Fraction(float(value)) for value in field.values)
    # Fraction arithmetic sees changes hidden by the enormous baseline edge
    # energy, and tests the actual objective rather than another solve helper.
    perturbation = max(abs(v), Fraction(1)) / 10000
    assert objective(u, v) < objective(u, v - perturbation)
    assert objective(u, v) < objective(u, v + perturbation)


@pytest.mark.parametrize("lam", [0.0, 1e-20, 0.1])
@pytest.mark.parametrize("width", [0.01, np.nextafter(0.0, 1.0)])
def test_learn_independent_components_retain_small_nearest_and_tie_targets(lam, width):
    field = _disconnected_ivm_field()
    field.learn(
        [field.sites[i] for i in (0, 2, 3, 5)],
        [1e-17, 3e-17, 1e16, 7.0], lam=lam, kernel_width=width,
    )
    # The central chain target is the mean of its equidistant endpoints.
    # The separate edge has only the large observation, so both its target
    # and fitted value are constant, independent of the small chain.
    strength = Fraction(lam)
    left, right = Fraction(1e-17), Fraction(3e-17)
    midpoint = (left + right) / 2
    expected_left = (left + strength * midpoint) / (1 + strength)
    expected_right = (right + strength * midpoint) / (1 + strength)
    np.testing.assert_allclose(
        field.values, [float(expected_left), float(midpoint), float(expected_right), 1e16, 1e16, 7.0],
        rtol=2e-15, atol=0.0,
    )


def test_learn_rejects_unobserved_components_atomically():
    field = _disconnected_ivm_field()
    previous = field.values
    before = previous.copy()
    with pytest.raises(ValueError):
        field.learn(field.sites[:1], [1.0])
    assert field.values is previous
    np.testing.assert_array_equal(previous, before)


def test_learn_rejects_underflowed_response_before_committing():
    field = _disconnected_ivm_field()
    previous = field.values
    before = previous.copy()
    # Two-hop response is below float range, but multiplying it by the
    # maximum observation would produce a representable physical field.
    # A dense double response cannot represent that answer: reject, not zero.
    with pytest.raises(np.linalg.LinAlgError):
        field.learn(field.sites, [np.finfo(float).max, 0.0, 0.0, 1.0, 1.0, 1.0], lam=1e-200)
    assert field.values is previous
    np.testing.assert_array_equal(previous, before)


@pytest.mark.parametrize(
    "kwargs, values, error",
    [
        ({"lam": np.inf}, [1.0], ValueError),
        ({"kernel_width": np.inf}, [1.0], ValueError),
        ({}, [[1.0]], ValueError),
        ({}, [np.nan], ValueError),
        ({"lam": np.finfo(float).max}, [1.0], np.linalg.LinAlgError),
    ],
)
def test_learn_failure_preserves_previous_values_and_graph_ownership(kwargs, values, error):
    field = IVMField.lattice_ball(1)
    field.values[:] = np.arange(len(field.sites))
    previous = field.values
    before = previous.copy()
    sites, adjacency, index = field.sites, field.adjacency, field.site_index
    with pytest.raises(error):
        field.learn([field.sites[0]], values, **kwargs)
    assert field.values is previous
    np.testing.assert_array_equal(previous, before)
    assert field.sites is sites
    assert field.adjacency is adjacency
    assert field.site_index is index


def test_learn_lam_validation():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([Quadray(0, 0, 0, 0)], [1.0], lam=-1e-3)


def test_learn_kernel_width_validation():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([Quadray(0, 0, 0, 0)], [1.0], kernel_width=0.0)


def test_learn_requires_observations():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([], [])


def test_learn_length_mismatch_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([Quadray(0, 0, 0, 0)], [1.0, 2.0])


def test_learn_nonfinite_values_raise():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([Quadray(0, 0, 0, 0)], [float("nan")])


def test_learn_site_outside_ball_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([Quadray(4, 2, 2, 0)], [1.0])


def test_learn_laplacian_is_symmetric():
    field = IVMField.lattice_ball(2)
    lap = field._laplacian()
    assert np.allclose(lap, lap.T)
    assert np.allclose(lap.sum(axis=1), 0.0, atol=1e-12)


def test_learn_soft_boundary_values_shrink_under_regularization():
    field = IVMField.lattice_ball(1)
    boundary = shell_sites(1)
    values = np.array([_xyz(q)[0] for q in boundary])
    field.learn(boundary, values, lam=1.0)
    fitted = np.array([field.predict(q) for q in boundary])
    np.testing.assert_allclose(fitted, values / 4.0, atol=1e-14)
    assert np.max(np.abs(fitted - values)) == pytest.approx(1.5)


# --------------- hard-boundary harmonic extension ---------------


def test_harmonic_extension_recovers_linear_field_from_boundary_only():
    field = IVMField.lattice_ball(3)
    truth = np.array([2.0 + _xyz(q) @ [0.75, -0.5, 0.25] for q in field.sites])
    boundary_idx = [i for i, q in enumerate(field.sites) if quadray_shell_norm(q) == 6]
    boundary = [field.sites[i] for i in boundary_idx]
    field.harmonic_extension(boundary, truth[boundary_idx])
    np.testing.assert_array_equal(field.values[boundary_idx], truth[boundary_idx])
    np.testing.assert_allclose(field.values, truth, rtol=0.0, atol=2e-14)
    interior_idx = sorted(set(range(len(field.sites))) - set(boundary_idx))
    residual = [
        len(field.adjacency[i]) * field.values[i] - sum(field.values[j] for j in field.adjacency[i])
        for i in interior_idx
    ]
    # Bound the backward error of each Laplacian row, not an absolute residual
    # tied to one BLAS implementation's rounding at this particular field scale.
    row_magnitude = np.array([
        len(field.adjacency[i]) * abs(field.values[i])
        + sum(abs(field.values[j]) for j in field.adjacency[i])
        for i in interior_idx
    ])
    term_count = np.array([len(field.adjacency[i]) + 1 for i in interior_idx])
    gamma = term_count * np.finfo(float).eps / (1 - term_count * np.finfo(float).eps)
    assert np.all(np.abs(residual) <= gamma * row_magnitude)


def test_harmonic_extension_preserves_boundary_and_maximum_principle():
    field = IVMField.lattice_ball(2)
    boundary = [field.sites[0], field.sites[-1], field.sites[-2]]
    boundary_values = [-3.0, 7.0, 2.0]
    field.harmonic_extension(boundary, boundary_values)
    assert [field.predict(q) for q in boundary] == boundary_values
    assert np.min(field.values) >= -3.0 - 1e-13
    assert np.max(field.values) <= 7.0 + 1e-13
    boundary_set = set(boundary)
    for i, q in enumerate(field.sites):
        if q not in boundary_set:
            assert field.values[i] == pytest.approx(
                np.mean(field.values[list(field.adjacency[i])]), abs=1e-13
            )


def test_harmonic_extension_all_boundary_uses_site_order_not_input_order():
    field = IVMField.lattice_ball(1)
    boundary = list(reversed(field.sites))
    values = np.linspace(-2.0, 4.0, len(boundary))
    field.harmonic_extension(boundary, values)
    np.testing.assert_array_equal(field.values, values[::-1])


def test_harmonic_extension_isolated_site_and_gauge_representatives():
    field = IVMField.lattice_ball(0)
    field.harmonic_extension([Quadray(19, 19, 19, 19)], [3.25])
    assert field.predict(Quadray(0, 0, 0, 0)) == 3.25
    field = IVMField.lattice_ball(2)
    anchor = field.sites[-1]
    gauged = Quadray(*(c + 19 for c in anchor.as_tuple()))
    field.harmonic_extension([gauged], [-1.25])
    assert field.predict(anchor) == -1.25
    np.testing.assert_allclose(field.values, -1.25, rtol=0.0, atol=2e-14)


def _disconnected_ivm_field() -> IVMField:
    # Actual induced IVM graph: a three-site chain, an edge, and an isolated
    # site on one neighbor direction, separated by missing lattice sites.
    step = IVM_NEIGHBOR_STEPS[0].as_tuple()
    sites = tuple(Quadray(*(k * c for c in step)) for k in (0, 1, 2, 4, 5, 7))
    return IVMField(
        radius=7,
        sites=sites,
        site_index={q: i for i, q in enumerate(sites)},
        adjacency=((1,), (0, 2), (1,), (4,), (3,), ()),
        values=np.full(len(sites), 99.0),
    )


def test_harmonic_extension_anchors_every_disconnected_component():
    field = _disconnected_ivm_field()
    field.harmonic_extension([field.sites[i] for i in (0, 2, 3, 5)], [2.0, 6.0, -4.0, 8.0])
    np.testing.assert_array_equal(field.values, [2.0, 4.0, 6.0, -4.0, -4.0, 8.0])


@pytest.mark.parametrize(
    "value", [np.nextafter(0.0, 1.0), 1e308, -np.finfo(float).max]
)
def test_harmonic_extension_preserves_constant_boundary_at_float_extremes(value):
    field = IVMField.lattice_ball(1)
    boundary = shell_sites(1)
    field.harmonic_extension(boundary, [value] * len(boundary))
    np.testing.assert_array_equal(field.values, np.full(len(field.sites), value))


def test_harmonic_extension_scales_disconnected_components_independently():
    field = _disconnected_ivm_field()
    tiny = np.nextafter(0.0, 1.0)
    field.harmonic_extension(
        [field.sites[i] for i in (0, 3, 5)], [1e308, tiny, -7.0]
    )
    np.testing.assert_array_equal(
        field.values, [1e308, 1e308, 1e308, tiny, tiny, -7.0]
    )


@pytest.mark.parametrize("anchors", [(0, 2), (0, 2, 5)])
def test_harmonic_extension_rejects_unanchored_components_without_mutation(anchors):
    field = _disconnected_ivm_field()
    before = field.values.copy()
    with pytest.raises(ValueError):
        field.harmonic_extension([field.sites[i] for i in anchors], [1.0] * len(anchors))
    np.testing.assert_array_equal(field.values, before)


def test_harmonic_extension_requires_boundary_for_isolated_site():
    field = IVMField.lattice_ball(0)
    with pytest.raises(ValueError):
        field.harmonic_extension([], [])


@pytest.mark.parametrize("values", [[], [1.0, 2.0], [[1.0]]])
def test_harmonic_extension_rejects_mismatched_or_nonscalar_values(values):
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.harmonic_extension([field.sites[0]], values)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_harmonic_extension_rejects_nonfinite_values(value):
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.harmonic_extension([field.sites[0]], [value])


def test_harmonic_extension_rejects_sites_outside_field():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.harmonic_extension([Quadray(4, 2, 2, 0)], [1.0])


def test_harmonic_extension_rejects_gauge_duplicate_boundary_sites():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.harmonic_extension([Quadray(0, 0, 0, 0), Quadray(4, 4, 4, 4)], [1.0, 1.0])


def test_multi_source_distances_bfs_and_unreachable():
    adjacency = [[1], [0, 2], [1], []]  # 0-1-2 connected, 3 isolated
    dist = _distances(adjacency, [0])
    assert dist.tolist() == [0, 1, 2, -1]
    # A source that is already reached (duplicate source) is not re-seeded
    dist_dup = _distances([[1], [0]], [0, 0])
    assert dist_dup.tolist() == [0, 1]


def _distances(adjacency: list[list[int]], sources: list[int]) -> "np.ndarray":
    from quadmath.lattice.ivm_field import _multi_source_distances

    return _multi_source_distances(adjacency, sources)


# --------------- fit_geometry ---------------


def _rotation(axis: np.ndarray, angle: float) -> "np.ndarray":
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * cross + (1.0 - c) * (cross @ cross)


def test_fit_geometry_recovers_synthetic_tetrahedron():
    rng = np.random.default_rng(7)
    rot = _rotation(np.array([1.0, 0.0, 0.0]), 0.4) @ _rotation(np.array([0.0, 0.0, 1.0]), 0.7)
    g_true = 2.5 * rot
    basis4 = np.array([[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
    unit_tet = np.array([basis4 @ np.array(v, dtype=float) for v in
                         [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]])  # (4, 3)
    true_vertices = unit_tet @ g_true.T
    points: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(4):
        for _ in range(20):
            points.append(true_vertices[i] + rng.normal(0.0, 1e-7, 3))
            labels.append(i)
    fit = fit_geometry(np.array(points), labels)
    assert isinstance(fit, TetrahedronFit)
    assert fit.residual < 1e-6
    fitted_vertices = (fit.matrix @ unit_tet.T).T
    assert np.max(np.abs(fitted_vertices - true_vertices)) < 1e-6
    assert abs(fit.scale - 2.5) < 1e-6


def test_fit_geometry_reflected_tetrahedron_negative_determinant():
    rng = np.random.default_rng(3)
    g_true = 1.5 * np.diag([1.0, 1.0, -1.0])  # reflection: det < 0
    basis4 = np.array(DEFAULT_EMBEDDING, dtype=float)
    unit_tet = np.array([basis4 @ np.array(v, dtype=float) for v in
                         [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]])
    true_vertices = unit_tet @ g_true.T
    points: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(4):
        for _ in range(10):
            points.append(true_vertices[i] + rng.normal(0.0, 1e-7, 3))
            labels.append(i)
    fit = fit_geometry(np.array(points), labels)
    assert np.max(np.abs(fit.matrix @ unit_tet.T - true_vertices.T)) < 1e-6
    assert fit.scale < 0  # signed cube root of a negative determinant
    assert abs(fit.scale + 1.5) < 1e-6


def test_fit_geometry_scales_with_embedding():
    # Doubling the embedding scales the recovered matrix by one half
    rng = np.random.default_rng(5)
    embedding2 = (2.0 * np.array(DEFAULT_EMBEDDING, dtype=float)).tolist()
    basis4 = np.array(DEFAULT_EMBEDDING, dtype=float)
    unit_tet = np.array([basis4 @ np.array(v, dtype=float) for v in
                         [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]])
    g_true = 2.0 * _rotation(np.array([0.0, 1.0, 0.0]), 0.9)
    true_vertices = unit_tet @ g_true.T
    points = np.concatenate([true_vertices[i] + rng.normal(0.0, 1e-9, 3) for i in range(4)])
    points = points.reshape(4, 3)
    labels: list[int] = [0, 1, 2, 3]
    fit = fit_geometry(points, labels, embedding=embedding2)
    assert np.max(np.abs(fit.matrix - g_true / 2.0)) < 1e-8


def test_fit_geometry_rejects_bad_points():
    with pytest.raises(ValueError):
        fit_geometry(np.zeros(6), [0, 1, 2])
    with pytest.raises(ValueError):
        fit_geometry(np.zeros((2, 2)), [0, 1])
    with pytest.raises(ValueError):
        fit_geometry(np.zeros((0, 3)), [])


def test_fit_geometry_rejects_nonfinite_points():
    pts = np.zeros((4, 3))
    pts[1, 1] = np.inf
    with pytest.raises(ValueError):
        fit_geometry(pts, [0, 1, 2, 3])


def test_fit_geometry_rejects_label_mismatch():
    with pytest.raises(ValueError):
        fit_geometry(np.zeros((4, 3)), [0, 1, 2])


def test_fit_geometry_rejects_noninteger_labels():
    with pytest.raises(ValueError):
        fit_geometry(np.zeros((4, 3)), [0.5, 1.0, 2.0, 3.0])

def test_fit_geometry_requires_all_four_vertices():
    pts = np.zeros((4, 3))
    with pytest.raises(ValueError):
        fit_geometry(pts, [0, 0, 1, 2])


# --------------- cross-checks against quadray.py ---------------


def test_shell_one_sites_match_to_xyz_embedding():
    # The 12 shell-1 sites embed at equal magnitude (vector equilibrium)
    sites = shell_sites(1)
    mags = [float(np.linalg.norm(E @ np.array(q.as_tuple(), dtype=float))) for q in sites]
    assert np.allclose(mags, mags[0])
    # They embed exactly at the images of the (2,1,1,0) permutations
    expected = to_xyz(Quadray(2, 1, 1, 0), DEFAULT_EMBEDDING)
    assert any(np.allclose(E @ np.array(q.as_tuple(), dtype=float), expected) for q in sites)


def test_learn_rejects_nan_regularization_and_kernel_width():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError):
        field.learn([Quadray(0, 0, 0, 0)], [1.0], lam=float("nan"))
    with pytest.raises(ValueError):
        field.learn([Quadray(0, 0, 0, 0)], [1.0], kernel_width=float("nan"))


def test_learn_rejects_duplicate_observation_sites():
    field = IVMField.lattice_ball(1)
    origin = Quadray(0, 0, 0, 0)
    with pytest.raises(ValueError):
        field.learn([origin, origin], [1.0, 2.0])
