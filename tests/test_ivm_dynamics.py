"""Tests for src/ivm_dynamics.py — real numerics, fixed seeds, no mocks.

Covers: lattice construction on the IVM move graph, the per-step
sum-of-squares monotonicity lemma for the heat update, range containment
for the majority update (plus an honesty case where rounding increases
the sum of squares), determinism of simulate, gradient-free coupling
identification via fit_trajectory, and the demo figure renderer.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # before any lazy matplotlib import

import numpy as np
import pytest

from ivm_dynamics import (
    DynamicsParams,
    ball_sites,
    fit_trajectory,
    heat_step,
    is_nonincreasing,
    make_lattice,
    majority_step,
    neighbor_shifts,
    render_dynamics_demo,
    simulate,
    site_radius_sq,
    step,
    sum_of_squares,
)
from quadray import Quadray

LATTICE = make_lattice(3)  # 27 sites; origin component = 12-around-one cluster
SEED = 7


# --------------- lattice geometry ---------------


def test_neighbor_shifts_are_the_12_ivm_moves():
    shifts = neighbor_shifts()
    assert len(shifts) == 12
    assert len({s.as_tuple() for s in shifts}) == 12
    for s in shifts:
        assert sorted(s.as_tuple()) == [0, 1, 1, 2]
        assert site_radius_sq(s) == 8  # close-packing distance


def test_site_radius_sq_values_and_shift_invariance():
    assert site_radius_sq(Quadray(0, 0, 0, 0)) == 0
    assert site_radius_sq(Quadray(1, 0, 0, 0)) == 3
    assert site_radius_sq(Quadray(2, 1, 1, 0)) == 8
    # (1,1,1,1) lies in the embedding kernel: radius is shift-invariant.
    assert site_radius_sq(Quadray(3, 2, 2, 1)) == site_radius_sq(Quadray(2, 1, 1, 0))


def test_ball_sites_radius_zero():
    sites = ball_sites(0)
    assert sites == [Quadray(0, 0, 0, 0)]


def test_ball_sites_radius_two_counts_shells():
    sites = ball_sites(2)
    assert len(sites) == 15  # origin + 8 vertex sites (r^2=3) + 6 axis (r^2=4)
    radii = [site_radius_sq(q) for q in sites]
    assert radii == sorted(radii)  # deterministic shell ordering
    for q in sites:
        components = q.as_tuple()
        assert min(components) == 0 and min(components) >= 0  # canonical


def test_ball_sites_radius_three():
    sites = ball_sites(3)
    assert len(sites) == 27
    assert Quadray(0, 0, 0, 0) in sites
    assert Quadray(2, 1, 1, 0) in sites
    assert ball_sites(3) == sites  # deterministic across calls


def test_ball_sites_negative_radius_raises():
    with pytest.raises(ValueError):
        ball_sites(-1)


def test_make_lattice_origin_component_is_twelve_around_one():
    lattice = LATTICE
    assert lattice.size == 27
    assert lattice.radius == 3
    origin = lattice.sites[0]
    assert origin == Quadray(0, 0, 0, 0)
    assert lattice.degrees[0] == 12  # origin touches all 12 close packers
    assert np.all(lattice.adjacency == lattice.adjacency.T)  # undirected
    assert set(np.unique(lattice.adjacency)) <= {0, 1}


def test_make_lattice_row_stochastic_average_and_symmetric_diffusion():
    lattice = LATTICE
    assert np.allclose(lattice.average.sum(axis=1), 1.0)  # incl. isolated rows
    assert np.allclose(lattice.diffusion, lattice.diffusion.T)
    eigvals = np.linalg.eigvalsh(lattice.diffusion)
    assert eigvals.min() >= -1.0 - 1e-12
    assert eigvals.max() <= 1.0 + 1e-12
    assert lattice.size == 27


def test_make_lattice_negative_radius_raises():
    with pytest.raises(ValueError):
        make_lattice(-1)


def test_lattice_size_property():
    assert make_lattice(1).size == 1


# --------------- observables ---------------


def test_sum_of_squares():
    assert sum_of_squares(np.array([3.0, 4.0])) == 25.0
    assert sum_of_squares(np.array([-2, 1], dtype=np.int64)) == 5.0


def test_is_nonincreasing_true_false_and_tolerance():
    assert is_nonincreasing([3.0, 2.0, 2.0])
    assert not is_nonincreasing([1.0, 2.0])
    assert is_nonincreasing([1.0, 1.0 + 5e-13])  # within default tolerance
    assert not is_nonincreasing([1.0, 1.0 + 5e-11])
    assert is_nonincreasing([])  # empty sequence trivially holds
    assert is_nonincreasing([1.0])  # single value


# --------------- heat update: proven monotonicity ---------------


def test_heat_step_alpha_zero_is_identity():
    single = make_lattice(1)
    assert np.array_equal(heat_step(np.array([1.0]), single, 0.0), np.array([1.0]))


def test_heat_step_hand_values_on_single_site_lattice():
    single = make_lattice(1)  # one isolated site: diffusion row is zero
    u = np.array([5.0])
    assert heat_step(u, single, 0.3)[0] == pytest.approx(3.5)  # (1-0.3)*5
    assert heat_step(u, single, 1.0)[0] == 0.0


def test_heat_step_isolated_sites_decay_exactly():
    sparse = make_lattice(2)  # origin + two tetrahedra (deg 3) + octahedron (deg 4)
    isolated = np.flatnonzero(sparse.degrees == 0)
    assert isolated.tolist() == [0]  # only the origin is isolated at radius 2
    u = np.ones(15)
    for _ in range(4):
        u = heat_step(u, sparse, 0.5)
    assert np.allclose(u[isolated], 0.5**4)  # uniform geometric decay
    # Connected pieces are regular (uniform degree per component), so the
    # symmetric normalization S preserves their constant state exactly:
    assert np.allclose(u[sparse.degrees > 0], 1.0, atol=1e-12)


def test_heat_step_validation():
    u = np.ones(LATTICE.size)
    with pytest.raises(ValueError):
        heat_step(u, LATTICE, -0.1)
    with pytest.raises(ValueError):
        heat_step(u, LATTICE, 1.1)
    with pytest.raises(ValueError):
        heat_step(np.ones(LATTICE.size + 1), LATTICE, 0.5)
    with pytest.raises(ValueError):
        heat_step(np.ones((2, LATTICE.size)), LATTICE, 0.5)


def test_heat_update_l2_nonincreasing_per_step():
    """The honest claim: heat averaging is non-increasing in sum-of-squares.

    Asserted exactly per step, across seeds and the full coupling range.
    """
    for seed in (0, 1, 2):
        for alpha in (0.0, 0.3, 0.7, 1.0):
            traj = simulate(
                30,
                DynamicsParams(kind="heat", alpha=alpha, seed=seed),
                lattice=LATTICE,
            )
            for t in range(len(traj.sos) - 1):
                assert traj.sos[t + 1] <= traj.sos[t] + 1e-12
            assert is_nonincreasing(traj.sos)


# --------------- majority update: range containment, honesty case ---------------


def test_majority_step_hand_computed_neighborhood_vote():
    lattice = LATTICE
    v = next(
        i for i, q in enumerate(lattice.sites) if site_radius_sq(q) == 8
    )
    assert lattice.degrees[v] == 5  # origin + 4 cuboctahedron neighbors
    u = np.zeros(lattice.size, dtype=np.int64)
    u[v] = 5
    out = majority_step(u, lattice, 1.0)
    origin = 0
    assert out[v] == 0  # mean of v's neighbors (all 0) rounds to 0
    assert out[origin] == 0  # mean 5/12 over the 12 neighbors rounds to 0
    vertex_neighbors = [
        j for j in np.flatnonzero(lattice.adjacency[v]) if site_radius_sq(lattice.sites[j]) == 8
    ]
    assert len(vertex_neighbors) == 4
    for j in vertex_neighbors:
        assert out[j] == 1  # mean (5 + 0 + 0 + 0 + 0)/5 = 1 exactly


def test_majority_step_range_containment_per_step():
    """Proven claim: max non-increasing, min non-decreasing under rounding."""
    for seed in (0, 1):
        for alpha in (0.0, 0.4, 1.0):
            traj = simulate(
                25,
                DynamicsParams(kind="majority", alpha=alpha, seed=seed),
                lattice=LATTICE,
            )
            for prev, nxt in zip(traj.fields, traj.fields[1:]):
                assert nxt.max() <= prev.max()
                assert nxt.min() >= prev.min()


def test_majority_step_sum_of_squares_can_increase():
    """Honesty contract: rounding is NOT sum-of-squares monotone.

    Origin 0 with its 12 close packers at 1: the averaging step moves the
    origin up to 1 and each packer to rint(4/5) = 1, raising the sum of
    squares from 12 to 13. No monotonicity claim is made for this update.
    """
    lattice = LATTICE
    u = np.zeros(lattice.size, dtype=np.int64)
    for i, q in enumerate(lattice.sites):
        if site_radius_sq(q) == 8:
            u[i] = 1
    assert sum_of_squares(u) == 12.0
    out = majority_step(u, lattice, 1.0)
    assert out[0] == 1  # origin adopts the unanimous neighbor value
    assert sum_of_squares(out) == 13.0  # strictly increased


def test_majority_step_validation():
    u = np.zeros(LATTICE.size, dtype=np.int64)
    with pytest.raises(ValueError):
        majority_step(u, LATTICE, -0.1)
    with pytest.raises(ValueError):
        majority_step(u, LATTICE, 1.5)
    with pytest.raises(ValueError):
        majority_step(np.zeros(LATTICE.size + 1, dtype=np.int64), LATTICE, 0.5)
    with pytest.raises(ValueError):
        majority_step(np.zeros((2, LATTICE.size)), LATTICE, 0.5)
    with pytest.raises(ValueError):
        majority_step(np.full(LATTICE.size, 0.5), LATTICE, 0.5)  # not integer-valued


def test_majority_step_alpha_zero_holds_state():
    u = np.array([3, -1, 0, 2, 5] + [0] * (LATTICE.size - 5), dtype=np.int64)
    assert np.array_equal(majority_step(u, LATTICE, 0.0), u)


# --------------- step dispatch ---------------


def test_step_dispatch_matches_direct_updates():
    u = np.arange(LATTICE.size, dtype=float) - 10.0
    heat_params = DynamicsParams(kind="heat", alpha=0.25)
    maj_params = DynamicsParams(kind="majority", alpha=0.75)
    assert np.array_equal(step(u, LATTICE, heat_params), heat_step(u, LATTICE, 0.25))
    u_int = np.rint(u).astype(np.int64)
    assert np.array_equal(step(u_int, LATTICE, maj_params), majority_step(u_int, LATTICE, 0.75))


def test_step_unknown_kind_raises():
    with pytest.raises(ValueError):
        step(np.zeros(LATTICE.size), LATTICE, DynamicsParams(kind="wave"))


# --------------- simulate: determinism and validation ---------------


def test_simulate_is_deterministic_given_seed():
    p_heat = DynamicsParams(kind="heat", alpha=0.4, seed=SEED)
    p_maj = DynamicsParams(kind="majority", alpha=0.6, seed=SEED)
    for params in (p_heat, p_maj):
        t1 = simulate(10, params, lattice=LATTICE)
        t2 = simulate(10, params, lattice=LATTICE)
        assert len(t1.fields) == len(t2.fields) == 11
        for f1, f2 in zip(t1.fields, t2.fields):
            assert np.array_equal(f1, f2)
        assert t1.sos == t2.sos


def test_simulate_seed_changes_the_draw():
    p1 = DynamicsParams(kind="heat", alpha=0.4, seed=0)
    p2 = DynamicsParams(kind="heat", alpha=0.4, seed=99)
    assert not np.array_equal(
        simulate(1, p1, lattice=LATTICE).fields[0],
        simulate(2, p2, lattice=LATTICE).fields[0],
    )


def test_simulate_default_lattice_and_seeded_majority_integers():
    traj = simulate(5, DynamicsParams(radius=1))
    assert traj.lattice.size == 1
    assert traj.lattice.radius == 1
    maj = simulate(3, DynamicsParams(kind="majority", radius=2, seed=1))
    assert maj.lattice.size == 15
    assert maj.fields[0].dtype.kind == "i"  # seeded integer initial field
    assert maj.fields[0].min() >= -3 and maj.fields[0].max() <= 3


def test_simulate_seeded_heat_floats():
    traj = simulate(2, DynamicsParams(kind="heat", radius=2, seed=0))
    assert traj.fields[0].dtype.kind == "f"
    assert traj.fields[0].min() >= -2.0 and traj.fields[0].max() <= 2.0


def test_simulate_horizon_zero_returns_initial_state_only():
    traj = simulate(0, DynamicsParams(alpha=0.5, seed=1), lattice=LATTICE)
    assert len(traj.fields) == 1
    assert len(traj.sos) == 1
    assert traj.sos[0] == sum_of_squares(traj.fields[0])


def test_simulate_negative_horizon_raises():
    with pytest.raises(ValueError):
        simulate(-1, DynamicsParams(seed=0), lattice=LATTICE)


def test_simulate_rejects_bad_initial_fields():
    with pytest.raises(ValueError):
        simulate(1, DynamicsParams(seed=0), lattice=LATTICE, u0=np.zeros(LATTICE.size + 1))
    with pytest.raises(ValueError):
        simulate(1, DynamicsParams(seed=0), lattice=LATTICE, u0=np.zeros((2, LATTICE.size)))
    with pytest.raises(ValueError):
        simulate(
            1,
            DynamicsParams(kind="majority", seed=0),
            lattice=LATTICE,
            u0=np.full(LATTICE.size, 0.5),
        )


# --------------- coupling learning ---------------


def _observed_heat(alpha: float, seed: int = 5, horizon: int = 12) -> np.ndarray:
    traj = simulate(
        horizon,
        DynamicsParams(kind="heat", alpha=alpha, seed=seed),
        lattice=LATTICE,
        u0=None,
    )
    return np.stack(traj.fields)


def test_fit_recovers_exact_grid_coupling():
    grid = np.linspace(0.0, 1.0, 11)
    alpha_true = float(grid[3])  # exact float identity with the grid candidate
    observed = _observed_heat(alpha_true)
    result = fit_trajectory(observed, grid, LATTICE)
    assert result.kind == "heat"
    assert result.best_alpha == alpha_true
    assert result.best_mse == 0.0  # deterministic re-simulation reproduces data
    assert result.mses[alpha_true] == 0.0


def test_fit_refinement_finds_offgrid_coupling():
    alpha_true = 0.37
    observed = _observed_heat(alpha_true)
    coarse = fit_trajectory(observed, [0.0, 0.25, 0.5, 0.75, 1.0], LATTICE, refine_rounds=0)
    refined = fit_trajectory(observed, [0.0, 0.25, 0.5, 0.75, 1.0], LATTICE, refine_rounds=3)
    assert abs(refined.best_alpha - alpha_true) <= 0.0125
    assert refined.best_mse < 1e-3
    assert refined.best_mse <= coarse.best_mse  # refinement never worsens


def test_fit_best_at_low_boundary_is_stable():
    observed = _observed_heat(0.0)
    result = fit_trajectory(observed, [0.0, 0.5, 1.0], LATTICE, refine_rounds=2)
    assert result.best_alpha == 0.0
    assert result.best_mse == 0.0


def test_fit_best_at_high_boundary_is_stable():
    observed = _observed_heat(1.0)
    result = fit_trajectory(observed, [0.0, 0.5, 1.0], LATTICE, refine_rounds=2)
    assert result.best_alpha == 1.0
    assert result.best_mse == 0.0


def test_fit_degenerate_single_candidate_grid():
    observed = _observed_heat(0.3)
    result = fit_trajectory(observed, [0.5], LATTICE, refine_rounds=3)
    assert result.best_alpha == 0.5  # refinement interval collapses -> break
    assert result.best_mse > 0.0


def test_fit_majority_kind_recovers_coupling():
    grid = np.linspace(0.0, 1.0, 11)
    alpha_true = float(grid[5])
    traj = simulate(
        15, DynamicsParams(kind="majority", alpha=alpha_true, seed=3), lattice=LATTICE
    )
    observed = np.stack(traj.fields)
    result = fit_trajectory(observed, grid, LATTICE, kind="majority")
    assert result.kind == "majority"
    assert result.best_alpha == alpha_true
    assert result.best_mse == 0.0


def test_fit_validation():
    observed = _observed_heat(0.3)
    with pytest.raises(ValueError):
        fit_trajectory(observed, [], LATTICE)
    with pytest.raises(ValueError):
        fit_trajectory(observed[0], [0.5], LATTICE)  # 1-D observed
    with pytest.raises(ValueError):
        fit_trajectory(observed[:1], [0.5], LATTICE)  # initial condition only
    with pytest.raises(ValueError):
        fit_trajectory(observed[:, :5], [0.5], LATTICE)  # wrong width
    with pytest.raises(ValueError):
        fit_trajectory(observed, [-0.1], LATTICE)
    with pytest.raises(ValueError):
        fit_trajectory(observed, [1.1], LATTICE)


# --------------- demo figure ---------------


def test_render_dynamics_demo_default_path():
    path = render_dynamics_demo()
    assert path.endswith("ivm_dynamics_demo.png")
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_render_dynamics_demo_explicit_path(tmp_path):
    target = tmp_path / "ivm_dynamics_demo.png"
    returned = render_dynamics_demo(str(target))
    assert returned == str(target)
    assert target.stat().st_size > 0