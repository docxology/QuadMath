"""Tests for the typed pipeline composition layer (src/pipeline.py).

All tests run against real IVM lattice data with fixed RNG seeds — no
mocks, no ML frameworks.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ivm_dynamics import DynamicsParams, Trajectory, is_nonincreasing
from ivm_field import IVMField, ball_sites
from pipeline import (
    FieldLearner,
    FieldModel,
    Fittable,
    LatticeBall,
    LatticeSource,
    Pipeline,
    Step,
    _sample_sites,
    dynamics_step,
    learn_step,
    sites_step,
)
from quadray import DEFAULT_EMBEDDING, Quadray, to_xyz


def _z(q: Quadray) -> float:
    """Embedded z coordinate — the pipeline's default synthetic target."""
    return to_xyz(q, DEFAULT_EMBEDDING)[2]


# --------------- structural protocol conformance ---------------


def test_ivmfield_conforms_to_fieldmodel_protocol():
    field = IVMField.lattice_ball(1)
    assert isinstance(field, FieldModel)
    assert field.predict(Quadray(0, 0, 0, 0)) == 0.0


def test_latticeball_conforms_to_latticesource_protocol():
    source = LatticeBall(1)
    assert isinstance(source, LatticeSource)
    assert len(source.sites()) == 13


def test_fieldlearner_conforms_to_fittable_protocol():
    learner = FieldLearner(radius=1, lam=0.1, seed=0)
    assert isinstance(learner, Fittable)


# --------------- Step ---------------


def test_step_runs_wrapped_function():
    step = Step(name="add_one", fn=lambda x: x + 1, detail="1")
    assert step.run(3) == 4
    assert step.describe() == "add_one(1)"


def test_step_describe_without_detail_omits_parentheses():
    step = Step(name="identity", fn=lambda x: x)
    assert step.describe() == "identity"
    assert step.run(7) == 7


# --------------- Pipeline ---------------


def test_empty_pipeline_raises():
    with pytest.raises(ValueError, match="at least one"):
        Pipeline()


def test_then_returns_new_pipeline_without_mutating_original():
    first, second = sites_step(1), learn_step(0.1, 0, radius=1)
    base = Pipeline(first)
    extended = base.then(second)
    assert base.steps == (first,)
    assert extended.steps == (first, second)
    assert extended is not base
    assert isinstance(base.run(None), tuple)  # original still runs unchanged
    with pytest.raises(FrozenInstanceError):
        base.steps = (second,)  # type: ignore[misc]


def test_run_threads_values_in_order():
    pipe = Pipeline(
        Step(name="add_one", fn=lambda x: x + 1),
        Step(name="double", fn=lambda x: 2 * x),
    )
    assert pipe.run(3) == 8


def test_describe_lists_steps_in_order():
    pipe = Pipeline(sites_step(1), learn_step(0.1, 0, radius=1))
    assert pipe.describe() == "sites(radius=1) -> learn(lam=0.1, seed=0, radius=1)"


# --------------- ready-made steps over real repo methods ---------------


def test_sites_step_enumerates_ivm_ball_ignoring_input():
    sites = sites_step(1).run(None)
    assert sites == tuple(ball_sites(1))
    assert len(sites) == 13
    assert all(isinstance(q, Quadray) for q in sites)
    assert sites_step(1).run("ignored input") == sites


def test_fieldlearner_fit_without_data_uses_ball_sites():
    learner = FieldLearner(radius=1, lam=0.0, seed=5)
    field = learner.fit()
    assert isinstance(field, IVMField)
    obs = _sample_sites(list(field.sites), 5)
    assert len(obs) == 4  # 13 ball sites -> max(1, 13 // 3)
    # With lam=0 observed sites are pinned exactly to their targets.
    for q in obs:
        assert learner.predict(q) == pytest.approx(_z(q), abs=1e-9)


def test_fieldlearner_fit_consumes_provided_sites_and_target():
    learner = FieldLearner(
        radius=1, lam=0.0, seed=3, target=lambda q: float(q.b)
    )
    learner.fit([Quadray(2, 1, 1, 0)])
    assert learner.predict(Quadray(2, 1, 1, 0)) == 1.0


def test_fieldlearner_empty_pool_raises():
    learner = FieldLearner(radius=1, lam=0.1, seed=0)
    with pytest.raises(ValueError, match="observation pool is empty"):
        learner.fit([])


def test_fieldlearner_predict_before_fit_raises():
    learner = FieldLearner(radius=1, lam=0.1, seed=0)
    with pytest.raises(RuntimeError, match="before fit"):
        learner.predict(Quadray(0, 0, 0, 0))


# --------------- end-to-end threading on real IVM data ---------------


def test_pipeline_sites_then_learn_produces_fitted_field():
    pipe = Pipeline(sites_step(2), learn_step(0.05, 7, radius=2))
    field = pipe.run(None)
    assert isinstance(field, IVMField)
    assert field.radius == 2
    assert len(field.sites) == 55
    assert np.isfinite(field.values).all()
    assert field.values.any()
    # The learned field beats the zero field against the synthetic z target.
    targets = [_z(q) for q in field.sites]
    zero_score = float(np.mean(np.square(targets)))
    assert field.score(list(field.sites), targets) < zero_score
    assert pipe.describe() == "sites(radius=2) -> learn(lam=0.05, seed=7, radius=2)"


def test_pipeline_sites_then_learn_default_radius():
    fitted = Pipeline(sites_step(3), learn_step(0.1, 7)).run(None)
    assert isinstance(fitted, IVMField)
    assert fitted.radius == 3
    assert len(fitted.sites) == 147
    assert np.isfinite(fitted.values).all()


def test_pipeline_run_is_deterministic():
    pipe = Pipeline(sites_step(2), learn_step(0.05, 7, radius=2))
    first = pipe.run(None)
    second = pipe.run(None)
    assert np.array_equal(first.values, second.values)


def test_dynamics_step_runs_deterministic_heat_simulation():
    params = DynamicsParams(kind="heat", alpha=0.35, seed=11, radius=3)
    dyn = dynamics_step(params, T=5)
    traj = dyn.run(None)
    assert isinstance(traj, Trajectory)
    assert len(traj.fields) == 6
    # Heat diffusion never increases the observable sum of squares.
    assert is_nonincreasing(traj.sos)
    assert dyn.describe() == "dynamics(kind=heat, alpha=0.35, T=5)"
    ignored = dynamics_step(params, T=0).run("unused input")
    assert len(ignored.fields) == 1