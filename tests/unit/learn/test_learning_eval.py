"""Tests for lattice training/testing methodology (quadmath/learn/learning_eval.py).

All tests use real numerical examples on the IVM lattice with fixed RNG
seeds — no mocks, no ML frameworks.  Expected values below were calibrated
against the deterministic pipeline (fixed seeds make the numbers exact).
"""
from __future__ import annotations

import numpy as np
import pytest

from quadmath.lattice.ivm_dynamics import DynamicsParams, make_lattice, simulate, step
from quadmath.lattice.ivm_field import IVMField, ball_sites
from quadmath.learn.learning_eval import (
    GradientDescentTrainer,
    RidgeSiteFit,
    cross_validate_field,
    enclosing_radius,
    kfold_site_splits,
    learning_curve,
    ridge_site_fit,
    three_way_split,
    trajectory_train_test,
)
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray

E = np.array(DEFAULT_EMBEDDING)

#: Candidate grid shared by the trajectory tests; contains the exact float
#: literal 0.3 used to generate the synthetic heat data.
GRID8 = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def _xyz(q: Quadray) -> np.ndarray:
    """Embedded XYZ coordinates of a quadray (numpy vector)."""
    return E @ np.array(q.as_tuple(), dtype=float)


def _harmonic_truth(sites):
    """Linear (harmonic on the IVM graph) reference field over `sites`."""
    return {
        q: 2.0 + 0.75 * _xyz(q)[0] - 0.5 * _xyz(q)[1] + 0.25 * _xyz(q)[2]
        for q in sites
    }


def _bowl_truth(sites, scale: float = 2.0, quad: float = 0.15):
    """Quadratic bowl reference field over `sites` (non-harmonic)."""
    return {q: scale - quad * float(_xyz(q) @ _xyz(q)) for q in sites}


# --------------- enclosing radius ---------------


def test_enclosing_radius_values():
    assert enclosing_radius([Quadray(0, 0, 0, 0)]) == 0
    assert enclosing_radius(ball_sites(1)) == 1
    assert enclosing_radius(ball_sites(2)) == 2
    # Projective representatives agree: (3,2,2,1) ~ (2,1,1,0)
    assert enclosing_radius([Quadray(3, 2, 2, 1)]) == 1


def test_enclosing_radius_empty_raises():
    with pytest.raises(ValueError, match="at least one site"):
        enclosing_radius([])


def test_enclosing_radius_rejects_void_sites():
    with pytest.raises(ValueError, match="not an IVM lattice site"):
        enclosing_radius([Quadray(1, 0, 0, 0)])


# --------------- k-fold site splits ---------------


def test_kfold_splits_are_disjoint_and_exhaustive():
    sites = ball_sites(2)
    splits = kfold_site_splits(sites, 5, seed=3)
    assert len(splits) == 5
    assert [len(test) for _, test in splits] == [11] * 5
    assert [len(train) for train, _ in splits] == [44] * 5
    site_set = set(sites)
    seen: set = set()
    for train, test in splits:
        assert set(test) <= site_set
        assert not (set(test) & set(train))
        assert set(train) | set(test) == site_set
        assert not (set(test) & seen)  # folds are pairwise disjoint
        seen |= set(test)
    assert seen == site_set


def test_kfold_splits_deterministic_and_seed_sensitive():
    sites = ball_sites(2)
    a = kfold_site_splits(sites, 5, seed=3)
    b = kfold_site_splits(sites, 5, seed=3)
    c = kfold_site_splits(sites, 5, seed=4)
    assert a == b
    assert any(ta != tc for (_, ta), (_, tc) in zip(a, c))


def test_kfold_uneven_fold_sizes():
    sites = ball_sites(1)  # 13 sites
    splits = kfold_site_splits(sites, 3, seed=0)
    assert [len(test) for _, test in splits] == [5, 4, 4]
    assert [len(train) for train, _ in splits] == [8, 9, 9]


def test_kfold_rejects_bad_k_and_duplicates():
    sites = ball_sites(2)
    with pytest.raises(ValueError, match="k must be at least 2"):
        kfold_site_splits(sites, 1, seed=0)
    with pytest.raises(ValueError, match="cannot exceed the number of observed sites"):
        kfold_site_splits(sites, 56, seed=0)
    with pytest.raises(ValueError, match="distinct after normalization"):
        kfold_site_splits([Quadray(2, 1, 1, 0), Quadray(3, 2, 2, 1)], 2, seed=0)


# --------------- cross-validation ---------------


def test_cross_validation_noise_free_selects_zero_lambda():
    # Exact observations of a harmonic field: the kernel-weighted
    # interpolator (lam = 0) generalizes best; any Laplacian smoothing only
    # biases the fit away from the data.
    sites = ball_sites(2)
    truth = _harmonic_truth(sites)
    rng = np.random.default_rng(5)
    obs = [q for q in sites if rng.random() < 0.8]
    res = cross_validate_field([truth[q] for q in obs], obs, 4, [0.0, 0.5, 5.0, 50.0], seed=9)
    assert res.best_lam == 0.0
    assert res.mean_mse[0] < res.mean_mse[1] < res.mean_mse[2] < res.mean_mse[3]
    assert res.fold_mse[0][0] < res.fold_mse[0][1]  # first fold, lam 0 vs 0.5
    assert all(s > 0.0 for s in res.std_mse)  # folds differ


def test_cross_validation_noisy_selects_positive_lambda():
    # Noisy observations of a smooth field: pure interpolation carries the
    # observation noise into the fit, and moderate regularization wins on
    # the held-out folds (calibrated: kw=0.5 kernel, sigma=0.5 noise).
    sites = ball_sites(2)
    truth = _bowl_truth(sites, scale=0.5, quad=0.01)
    rng = np.random.default_rng(11)
    obs = [q for q in sites if rng.random() < 0.8]
    noisy = {q: truth[q] + rng.normal(0.0, 0.5) for q in obs}
    res = cross_validate_field(
        [noisy[q] for q in obs], obs, 4, [0.0, 0.1, 1.0, 10.0, 100.0], seed=13,
        kernel_width=0.5,
    )
    assert res.best_lam == 1.0
    assert res.mean_mse[2] < 0.95 * res.mean_mse[0]
    assert res.mean_mse[2] <= res.mean_mse[-1]


def test_cross_validation_table_and_refit_consistency():
    sites = ball_sites(2)
    truth = _harmonic_truth(sites)
    rng = np.random.default_rng(5)
    obs = [q for q in sites if rng.random() < 0.8]
    values = [truth[q] for q in obs]
    grid = [0.0, 0.5, 5.0, 50.0]
    res = cross_validate_field(values, obs, 4, grid, seed=9)

    table = np.asarray(res.fold_mse, dtype=float)
    assert table.shape == (4, 4)
    assert res.lam_grid == (0.0, 0.5, 5.0, 50.0)
    assert res.mean_mse == tuple(float(m) for m in np.mean(table, axis=0))
    assert res.std_mse == tuple(float(s) for s in np.std(table, axis=0))
    assert res.best_lam == grid[int(np.argmin(res.mean_mse))]
    assert res.best_mean_mse == min(res.mean_mse)

    # The refit field is a fresh ball fit with the selected lam on ALL
    # observed sites — reproduce it by hand.
    refit = IVMField.lattice_ball(2)
    refit.learn(obs, values, lam=res.best_lam)
    assert res.refit_field.radius == 2
    assert res.refit_mse == refit.score(obs, values)
    assert np.all(np.isfinite(res.refit_field.values))


def test_cross_validation_radius_inference_and_explicit():
    sites = ball_sites(2)
    truth = _bowl_truth(sites, scale=0.5, quad=0.01)
    rng = np.random.default_rng(11)
    obs = [q for q in sites if rng.random() < 0.8]
    values = [truth[q] for q in obs]
    inferred = cross_validate_field(values, obs, 4, [0.0, 1.0], seed=13)
    assert inferred.refit_field.radius == enclosing_radius(obs) == 2
    explicit = cross_validate_field(values, obs, 4, [0.0, 1.0], seed=13, radius=3)
    assert explicit.refit_field.radius == 3
    # A ball too small for the sites is rejected by IVMField.learn.
    with pytest.raises(ValueError, match="outside the lattice ball"):
        cross_validate_field(values, obs, 4, [0.0, 1.0], seed=13, radius=1)


def test_cross_validation_inferred_radius_covers_scattered_sites():
    # Sites scattered across shells 0, 2, and 3 (a sparse, non-contiguous
    # observation set): the inferred ball must still contain every observed
    # site, so fitting and scoring never touch an out-of-ball site.
    sites = [Quadray(0, 0, 0, 0), ball_sites(2)[-1], ball_sites(3)[-1]]
    values = [1.0, 2.0, 3.0]
    res = cross_validate_field(values, sites, 2, [0.0, 1.0], seed=3)
    assert res.refit_field.radius == 3
    for q in sites:
        assert np.isfinite(res.refit_field.predict(q))


def test_cross_validation_rejects_invalid_inputs():
    sites = ball_sites(1)
    values = [1.0] * len(sites)
    with pytest.raises(ValueError, match="same length"):
        cross_validate_field(values[:-1], sites, 2, [0.0], seed=0)
    with pytest.raises(ValueError, match="at least one observed site"):
        cross_validate_field([], [], 2, [0.0], seed=0)
    with pytest.raises(ValueError, match="distinct after normalization"):
        cross_validate_field([1.0, 2.0], [Quadray(2, 1, 1, 0), Quadray(3, 2, 2, 1)], 2, [0.0], seed=0)
    with pytest.raises(ValueError, match="at least one candidate"):
        cross_validate_field(values, sites, 2, [], seed=0)
    with pytest.raises(ValueError, match="k must be at least 2"):
        cross_validate_field(values, sites, 1, [0.0], seed=0)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        cross_validate_field(values, sites, 2, [-1.0], seed=0)


# --------------- trajectory train/test ---------------


def test_trajectory_split_correctly_specified_is_exact():
    # Heat dynamics with alpha = 0.3 (in GRID8): the identified model
    # re-simulates the data exactly, so train, held-out multi-step, and
    # one-step-ahead errors are all exactly zero.
    lat = make_lattice(2)
    u0 = np.random.default_rng(19).uniform(-2.0, 2.0, size=lat.size)
    observed = np.stack(
        simulate(8, DynamicsParams(kind="heat", alpha=0.3), lattice=lat, u0=u0).fields
    )
    res = trajectory_train_test(observed, 0.55, lat, grid=GRID8)
    assert res.n_train_rows == 5  # round(0.55 * 9) = 5
    assert res.n_test_rows == 4
    assert res.best_alpha == 0.3
    assert res.train_mse == 0.0
    assert res.test_mse == 0.0
    assert res.one_step_mse == 0.0
    assert res.kind == "heat"


def test_trajectory_split_misspecified_blows_up():
    # Majority (integer) dynamics identified with the heat model: the fit
    # error over the early training rows is modest, but the held-out
    # multi-step continuation error compounds with horizon.
    lat = make_lattice(2)
    u0 = np.random.default_rng(19).integers(-3, 4, size=lat.size)
    observed = np.stack(
        simulate(8, DynamicsParams(kind="majority", alpha=0.3), lattice=lat, u0=u0).fields
    )
    res = trajectory_train_test(observed, 0.55, lat, kind="heat", grid=GRID8)
    assert res.n_train_rows == 5
    assert res.n_test_rows == 4
    assert res.train_mse > 0.0
    assert res.test_mse > 2.0 * res.train_mse
    assert res.test_mse > 5.0 * res.one_step_mse
    assert res.one_step_mse > 0.0


def test_trajectory_split_one_step_matches_manual():
    lat = make_lattice(2)
    u0 = np.random.default_rng(19).uniform(-2.0, 2.0, size=lat.size)
    observed = np.stack(
        simulate(8, DynamicsParams(kind="heat", alpha=0.3), lattice=lat, u0=u0).fields
    )
    res = trajectory_train_test(observed, 0.55, lat, grid=GRID8)
    params = DynamicsParams(kind="heat", alpha=res.best_alpha)
    errors = [
        (step(observed[t], lat, params) - observed[t + 1]) ** 2 for t in range(4, 8)
    ]
    assert abs(res.one_step_mse - float(np.mean(errors))) < 1e-15


def test_trajectory_split_clamps_extreme_fractions():
    lat = make_lattice(2)
    u0 = np.random.default_rng(19).uniform(-2.0, 2.0, size=lat.size)
    observed = np.stack(
        simulate(6, DynamicsParams(kind="heat", alpha=0.3), lattice=lat, u0=u0).fields
    )
    lo = trajectory_train_test(observed, 0.05, lat)
    hi = trajectory_train_test(observed, 0.95, lat)
    assert (lo.n_train_rows, lo.n_test_rows) == (2, 5)
    assert (hi.n_train_rows, hi.n_test_rows) == (6, 1)


def test_trajectory_split_rejects_invalid_inputs():
    lat = make_lattice(1)
    u0 = np.random.default_rng(3).uniform(-2.0, 2.0, size=lat.size)
    two_rows = np.stack([u0, u0 * 0.5])
    good = np.stack(simulate(2, DynamicsParams(kind="heat", alpha=0.3), lattice=lat, u0=u0).fields)
    with pytest.raises(ValueError, match=r"shape \(T\+1, N\)"):
        trajectory_train_test(u0, 0.5, lat, grid=GRID8)
    with pytest.raises(ValueError, match="at least two training snapshots"):
        trajectory_train_test(two_rows, 0.5, lat, grid=GRID8)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        trajectory_train_test(good, 0.0, lat, grid=GRID8)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        trajectory_train_test(good, 1.0, lat, grid=GRID8)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        trajectory_train_test(good, -0.5, lat, grid=GRID8)


# --------------- learning curve ---------------


def test_learning_curve_final_below_initial():
    # The classic data-coverage curve: held-out MSE falls as the observed
    # fraction grows (calibrated on the radius-2 ball, sigma=0.4 noise).
    sites = ball_sites(2)
    truth = _bowl_truth(sites)
    rng = np.random.default_rng(17)
    noisy = {q: truth[q] + rng.normal(0.0, 0.4) for q in sites}
    fracs = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
    res = learning_curve([noisy[q] for q in sites], sites, fracs, seed=17, lam=0.05)
    assert res.train_fracs == fracs
    assert res.train_sizes == (6, 14, 22, 30, 38, 47)
    assert res.test_mse[-1] < 0.5 * res.test_mse[0]
    assert max(res.train_mse) < 0.15  # pinned observations are fit closely
    assert min(res.test_mse) > 0.7  # generalization error stays nontrivial


def test_learning_curve_seed_changes_subsets():
    sites = ball_sites(2)
    truth = _bowl_truth(sites)
    rng = np.random.default_rng(17)
    noisy = {q: truth[q] + rng.normal(0.0, 0.4) for q in sites}
    fracs = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
    values = [noisy[q] for q in sites]
    a = learning_curve(values, sites, fracs, seed=11, lam=0.05, radius=2)
    b = learning_curve(values, sites, fracs, seed=23, lam=0.05, radius=2)
    assert a.train_sizes == b.train_sizes
    assert a.test_mse != b.test_mse


def test_learning_curve_extreme_fractions_clamp():
    sites = ball_sites(2)
    truth = _bowl_truth(sites)
    values = [truth[q] for q in sites]
    res = learning_curve(values, sites, (0.01, 0.99), seed=5, radius=2)
    assert res.train_sizes == (1, 54)


def test_learning_curve_rejects_invalid_inputs():
    sites = ball_sites(1)
    values = [1.0] * len(sites)
    with pytest.raises(ValueError, match="same length"):
        learning_curve(values[:-1], sites, (0.5,), seed=0)
    with pytest.raises(ValueError, match="at least one observed site"):
        learning_curve([], [], (0.5,), seed=0)
    with pytest.raises(ValueError, match="needs at least two observed sites"):
        learning_curve([1.0], [Quadray(0, 0, 0, 0)], (0.5,), seed=0)
    with pytest.raises(ValueError, match="train_fracs must be non-empty"):
        learning_curve(values, sites, (), seed=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        learning_curve(values, sites, (0.0,), seed=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        learning_curve(values, sites, (1.0, 0.5), seed=0)


# --------------- three-way split ---------------


def test_three_way_split_disjoint_and_exhaustive():
    train, val, test = three_way_split(50, 0.6, 0.2, seed=0)
    assert sorted(train + val + test) == list(range(50))
    assert len(set(train) | set(val) | set(test)) == 50
    # realized sizes follow the rounded fractions
    assert (len(train), len(val), len(test)) == (30, 10, 10)


def test_three_way_split_seed_reproducible_and_sensitive():
    a = three_way_split(20, seed=0)
    b = three_way_split(20, seed=0)
    c = three_way_split(20, seed=1)
    assert a == b
    assert a != c


def test_three_way_split_defaults_and_small_n_tail():
    assert three_way_split(10) == three_way_split(10, 0.6, 0.2, 0)
    # rounding may empty the trailing test block for tiny n
    train, val, test = three_way_split(3)
    assert (len(train), len(val), len(test)) == (2, 1, 0)


def test_three_way_split_rejects_invalid_fractions():
    with pytest.raises(ValueError, match="train_frac must be strictly positive"):
        three_way_split(10, 0.0, 0.2)
    with pytest.raises(ValueError, match="val_frac must be strictly positive"):
        three_way_split(10, 0.6, 0.0)
    with pytest.raises(ValueError, match="must stay below 1"):
        three_way_split(10, 0.6, 0.4)
    with pytest.raises(ValueError, match="must stay below 1"):
        three_way_split(10, 0.8, 0.5)


# --------------- ridge site fit ---------------


def test_ridge_site_fit_recovers_linear_relation():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 4.0 + rng.normal(scale=0.01, size=40)
    fit = ridge_site_fit(X, y, lam=1e-3)
    assert isinstance(fit, RidgeSiteFit)
    assert np.allclose(fit.coefficients, [1.5, -2.0, 0.5], atol=1e-2)
    assert abs(fit.intercept - 4.0) < 1e-2
    assert fit.train_mse < 1e-3


def test_ridge_site_fit_zero_lambda_matches_ols():
    # Noiseless, well-conditioned rows: lam = 0 must reproduce the exact
    # ordinary-least-squares solution (coefficients, intercept, train MSE).
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 2))
    y = X @ np.array([2.0, -1.0]) + 1.0
    fit = ridge_site_fit(X, y, lam=0.0)
    design = np.hstack([X, np.ones((X.shape[0], 1))])
    w_ols, *_ = np.linalg.lstsq(design, y, rcond=None)
    assert np.allclose(fit.coefficients, w_ols[:-1])
    assert np.isclose(fit.intercept, w_ols[-1])
    resid = design @ w_ols - y
    assert np.isclose(fit.train_mse, float(np.mean(resid ** 2)))
    # deterministic: identical inputs give identical results
    again = ridge_site_fit(X, y, lam=0.0)
    assert np.array_equal(fit.coefficients, again.coefficients)
    assert fit.intercept == again.intercept
    assert fit.train_mse == again.train_mse


def test_ridge_site_fit_rejects_invalid_inputs():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="lam must be non-negative"):
        ridge_site_fit(X, y, lam=-1e-3)
    with pytest.raises(ValueError, match="features must be 2-D"):
        ridge_site_fit(np.arange(10.0), y)
    with pytest.raises(ValueError, match="values must be 1-D"):
        ridge_site_fit(X, y.reshape(-1, 1))
    with pytest.raises(ValueError, match="rows but values has"):
        ridge_site_fit(X, y[:-1])
    with pytest.raises(ValueError, match="at least one sample row"):
        ridge_site_fit(np.zeros((0, 2)), np.zeros(0))


# --------------- gradient descent trainer ---------------


def test_gradient_descent_converges_on_linear_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([3.0, -1.0]) + 2.0
    trainer = GradientDescentTrainer(lr=0.1, max_iters=300, tol=1e-9).fit(X, y)
    assert trainer.converged_ is True
    assert trainer.loss_history[-1] < 1e-6
    losses = np.asarray(trainer.loss_history)
    assert np.all(np.diff(losses) <= 1e-12)  # non-increasing up to fp noise
    assert losses[0] > losses[-1]
    predictions = trainer.predict(X)
    assert predictions.shape == (60,)
    assert np.allclose(predictions, y, atol=1e-3)
    # generalizes to fresh rows through the stored standardization
    X_new = rng.normal(size=(10, 2))
    assert np.allclose(
        trainer.predict(X_new), X_new @ np.array([3.0, -1.0]) + 2.0, atol=1e-3
    )


def test_gradient_descent_handles_constant_column():
    # A constant feature column standardizes to zeros (std clamped to 1):
    # training must still converge and predict in original units.
    rng = np.random.default_rng(13)
    column = rng.normal(size=(30, 1))
    X = np.hstack([column, np.full((30, 1), 5.0)])
    y = 2.0 * column[:, 0] + 1.0
    trainer = GradientDescentTrainer(lr=0.2, max_iters=300).fit(X, y)
    assert trainer.converged_ is True
    assert trainer.std_[1] == 1.0
    assert np.allclose(trainer.predict(X), y, atol=1e-6)


def test_gradient_descent_max_iters_one_does_not_converge():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(20, 2))
    y = X @ np.array([1.0, 2.0])
    trainer = GradientDescentTrainer(lr=0.05, max_iters=1, tol=1e-9).fit(X, y)
    assert trainer.converged_ is False
    assert len(trainer.loss_history) == 1


def test_gradient_descent_rejects_invalid_inputs():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="lr must be strictly positive"):
        GradientDescentTrainer(lr=0.0)
    with pytest.raises(ValueError, match="lr must be strictly positive"):
        GradientDescentTrainer(lr=-0.5)
    with pytest.raises(ValueError, match="max_iters must be at least 1"):
        GradientDescentTrainer(max_iters=0)
    trainer = GradientDescentTrainer().fit(X, y)
    with pytest.raises(ValueError, match="features must be 2-D"):
        trainer.fit(np.arange(10.0), y)
    with pytest.raises(ValueError, match="target must be 1-D"):
        trainer.fit(X, y.reshape(-1, 1))
    with pytest.raises(ValueError, match="rows but target has"):
        trainer.fit(X, y[:-1])
    with pytest.raises(ValueError, match="at least one sample row"):
        trainer.fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError, match="predict requires a prior call to fit"):
        GradientDescentTrainer().predict(X)
    with pytest.raises(ValueError, match="features must be 2-D"):
        trainer.predict(np.arange(4.0))
    with pytest.raises(ValueError, match="feature columns"):
        trainer.predict(np.zeros((3, 3)))
