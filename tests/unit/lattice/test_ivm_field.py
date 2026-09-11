"""Tests for static IVM field learning (quadmath/lattice/ivm_field.py).

All tests use real numerical examples on the IVM lattice with fixed RNG
seeds — no mocks, no ML frameworks.
"""
from __future__ import annotations

import numpy as np
import pytest

from quadmath.lattice.ivm_field import (
    IVM_NEIGHBOR_STEPS,
    IVMField,
    TetrahedronFit,
    ball_sites,
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
    with pytest.raises(ValueError, match="non-negative"):
        shell_sites(-1)


def test_ball_sites_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ball_sites(-2)


def test_ball_sites_shell_partition():
    ball = ball_sites(2)
    assert len(ball) == 1 + 12 + 42
    # Ordered by shell, lexicographic within shell
    norms = [quadray_shell_norm(q) for q in ball]
    assert norms == sorted(norms)
    assert len(set(ball)) == len(ball)


def test_shell_cardinalities_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
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
    with pytest.raises(ValueError, match="not an IVM lattice site"):
        quadray_shell_norm(Quadray(1, 0, 0, 0))
    with pytest.raises(ValueError, match="not an IVM lattice site"):
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
    with pytest.raises(ValueError, match="non-negative"):
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
    with pytest.raises(ValueError, match="outside the lattice ball"):
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
    with pytest.raises(ValueError, match="same length"):
        field.score([Quadray(2, 1, 1, 0)], [1.0, 2.0])


def test_score_empty_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="at least one site"):
        field.score([], [])


# --------------- learn ---------------


def test_learn_recovers_harmonic_linear_field():
    # A linear field in the embedded XYZ coordinates is harmonic on the IVM
    # graph (the mean of the 12 neighbor steps is the zero vector), so with
    # the boundary fully observed and a narrow kernel the Laplacian-
    # regularized fit reproduces it essentially exactly.
    radius = 3
    truth = {
        q: 2.0 + 0.75 * _xyz(q)[0] - 0.5 * _xyz(q)[1] + 0.25 * _xyz(q)[2]
        for q in ball_sites(radius)
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
    truth = {q: 0.5 - 0.01 * float(_xyz(q) @ _xyz(q)) for q in ball_sites(radius)}
    rng = np.random.default_rng(7)
    obs = [q for q in ball_sites(radius) if rng.random() < 0.5]
    noisy = {q: truth[q] + rng.normal(0.0, 0.3) for q in obs}
    raw_mse = float(np.mean([(noisy[q] - truth[q]) ** 2 for q in obs]))
    field = IVMField.lattice_ball(radius)
    field.learn(obs, [noisy[q] for q in obs], lam=0.05, kernel_width=1.0)
    learned_mse = float(np.mean([(field.predict(q) - truth[q]) ** 2 for q in ball_sites(radius)]))
    assert learned_mse < raw_mse
    assert learned_mse < 0.05


def test_learn_narrow_kernel_floors_unobserved_confidence():
    # With a very narrow kernel every unobserved site's confidence is
    # clamped to the numerical floor; the fit still solves (Laplacian
    # propagation from the single observation at the origin).
    field = IVMField.lattice_ball(2)
    origin = Quadray(0, 0, 0, 0)
    field.learn([origin], [1.0], lam=0.1, kernel_width=0.15)
    assert abs(field.predict(origin) - 1.0) < 1e-9
    assert all(np.isfinite(field.values))


def test_learn_lam_validation():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        field.learn([Quadray(0, 0, 0, 0)], [1.0], lam=-1e-3)


def test_learn_kernel_width_validation():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="kernel_width must be positive"):
        field.learn([Quadray(0, 0, 0, 0)], [1.0], kernel_width=0.0)


def test_learn_requires_observations():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="at least one observation"):
        field.learn([], [])


def test_learn_length_mismatch_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="same length"):
        field.learn([Quadray(0, 0, 0, 0)], [1.0, 2.0])


def test_learn_nonfinite_values_raise():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="finite"):
        field.learn([Quadray(0, 0, 0, 0)], [float("nan")])


def test_learn_site_outside_ball_raises():
    field = IVMField.lattice_ball(1)
    with pytest.raises(ValueError, match="outside the lattice ball"):
        field.learn([Quadray(4, 2, 2, 0)], [1.0])


def test_learn_laplacian_is_symmetric():
    field = IVMField.lattice_ball(2)
    lap = field._laplacian()
    assert np.allclose(lap, lap.T)
    assert np.allclose(lap.sum(axis=1), 0.0, atol=1e-12)


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
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        fit_geometry(np.zeros(6), [0, 1, 2])
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        fit_geometry(np.zeros((2, 2)), [0, 1])
    with pytest.raises(ValueError, match="at least one point"):
        fit_geometry(np.zeros((0, 3)), [])


def test_fit_geometry_rejects_nonfinite_points():
    pts = np.zeros((4, 3))
    pts[1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_geometry(pts, [0, 1, 2, 3])


def test_fit_geometry_rejects_label_mismatch():
    with pytest.raises(ValueError, match="same length"):
        fit_geometry(np.zeros((4, 3)), [0, 1, 2])


def test_fit_geometry_rejects_noninteger_labels():
    with pytest.raises(ValueError, match="integers"):
        fit_geometry(np.zeros((4, 3)), [0.5, 1.0, 2.0, 3.0])

def test_fit_geometry_requires_all_four_vertices():
    pts = np.zeros((4, 3))
    with pytest.raises(ValueError, match="all four tetrahedron vertices"):
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
