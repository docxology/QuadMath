"""Typed, composable pipeline layer over the QuadMath analytical methods.

A small structural-typing kit that composes the repo's existing analytical
surfaces — lattice enumeration (``quadmath.lattice.ivm_field.ball_sites``), Laplacian-
regularized field learning (:meth:`IVMField.learn`), and discrete lattice
dynamics (:func:`quadmath.lattice.ivm_dynamics.simulate`) — without modifying them.

Runtime-checkable protocols accept any object exposing the right methods,
so unmodified repo classes conform structurally:

- :class:`LatticeSource` — enumerates lattice sites (``sites()``).
- :class:`FieldModel` — predicts a scalar at a lattice site
  (``predict(site)``); the unmodified :class:`IVMField` conforms.
- :class:`Fittable` — fits a field model from data (``fit(data)``) and
  predicts (``predict(site)``); :class:`FieldLearner` conforms.

Composition is monoid-style and free of global state: a :class:`Pipeline`
is an immutable sequence of :class:`Step` objects, ``then`` returns a new
pipeline, and ``run`` threads each step's output into the next step's
input.  Any other repo function can join a chain by wrapping it in a
:class:`Step`.

Examples
--------
>>> from quadmath.pipeline import Pipeline, sites_step, learn_step
>>> pipe = Pipeline(sites_step(2), learn_step(0.05, 7, radius=2))
>>> field = pipe.run(None)          # fitted IVMField over the radius-2 ball
>>> pipe.describe()
'sites(radius=2) -> learn(lam=0.05, seed=7, radius=2)'
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from quadmath.lattice.ivm_dynamics import DynamicsParams, Trajectory, simulate
from quadmath.lattice.ivm_field import IVMField, ball_sites
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, to_xyz

__all__ = [
    "FieldLearner",
    "FieldModel",
    "Fittable",
    "LatticeBall",
    "LatticeSource",
    "Pipeline",
    "Step",
    "dynamics_step",
    "learn_step",
    "sites_step",
]


@runtime_checkable
class LatticeSource(Protocol):
    """Anything that can enumerate IVM lattice sites (structural)."""

    def sites(self) -> Sequence[Quadray]:
        """Return the lattice sites in canonical order."""
        ...


@runtime_checkable
class FieldModel(Protocol):
    """Anything that predicts a scalar field value at a lattice site."""

    def predict(self, site: Quadray) -> float:
        """Return the predicted field value at ``site``."""
        ...


@runtime_checkable
class Fittable(Protocol):
    """Anything that can fit a field model from data, then predict."""

    def fit(self, data: Optional[Sequence[Quadray]] = None) -> IVMField:
        """Fit a field model; ``data`` optionally supplies the site pool."""
        ...

    def predict(self, site: Quadray) -> float:
        """Return the fitted field value at ``site``."""
        ...


@dataclass(frozen=True)
class Step:
    """A named, typed callable unit of a :class:`Pipeline`.

    - name: stable identifier used by :meth:`describe`.
    - fn: pure callable threaded one value in, one value out.
    - detail: short parameter summary rendered by :meth:`describe`.
    """

    name: str
    fn: Callable[[Any], Any]
    detail: str = ""

    def run(self, data: Any) -> Any:
        """Apply the wrapped callable to ``data`` and return the result."""
        return self.fn(data)

    def describe(self) -> str:
        """Render the step as ``name(detail)``, or ``name`` when unparameterized."""
        return f"{self.name}({self.detail})" if self.detail else self.name


def _default_target(q: Quadray) -> float:
    """Synthetic observation target: the embedded z coordinate of ``q``."""
    return to_xyz(q, DEFAULT_EMBEDDING)[2]


def _sample_sites(pool: Sequence[Quadray], seed: int) -> List[Quadray]:
    """Deterministically sample about a third of ``pool`` as observation sites.

    Parameters
    - pool: candidate lattice sites (non-empty).
    - seed: RNG seed fixing the sample (determinism contract).

    Returns
    - List[Quadray]: sampled sites in pool order, at least one.
    """
    rng = np.random.default_rng(seed)
    size = max(1, len(pool) // 3)
    chosen = np.sort(rng.choice(len(pool), size=size, replace=False))
    return [pool[int(i)] for i in chosen]


@dataclass(frozen=True)
class LatticeBall:
    """Immutable view of an IVM lattice ball; satisfies :class:`LatticeSource`."""

    radius: int

    def sites(self) -> Tuple[Quadray, ...]:
        """Return the ball's sites in canonical (shell, lexicographic) order."""
        return tuple(ball_sites(self.radius))


@dataclass
class FieldLearner:
    """Laplacian-regularized field learner over an IVM ball; satisfies :class:`Fittable`.

    Wraps :meth:`IVMField.learn` without modifying it: ``fit`` builds the
    ball of :attr:`radius`, deterministically samples observations from
    ``data`` (or the whole ball when ``data`` is ``None``), targets them
    with ``target``, and returns the fitted :class:`IVMField`.  ``predict``
    delegates to the last fitted model.
    """

    radius: int
    lam: float
    seed: int
    kernel_width: float = 1.5
    target: Callable[[Quadray], float] = _default_target
    model: Optional[IVMField] = None

    def fit(self, data: Optional[Sequence[Quadray]] = None) -> IVMField:
        """Fit an :class:`IVMField` over this learner's ball.

        Parameters
        - data: optional pool of lattice sites to sample observations
          from; ``None`` samples the whole ball.  Sites outside the
          ball are rejected by :meth:`IVMField.learn`.

        Returns
        - IVMField: the fitted field (also cached on ``self.model``).

        Raises
        - ValueError: if ``data`` is an empty sequence, or an observation
          site lies outside the ball.
        """
        field = IVMField.lattice_ball(self.radius)
        pool = list(data) if data is not None else list(field.sites)
        if not pool:
            raise ValueError("observation pool is empty")
        obs = _sample_sites(pool, self.seed)
        values = [self.target(q) for q in obs]
        self.model = field.learn(
            obs, values, lam=self.lam, kernel_width=self.kernel_width
        )
        return self.model

    def predict(self, site: Quadray) -> float:
        """Return the fitted field value at ``site``.

        Raises
        - RuntimeError: if called before :meth:`fit`.
        """
        if self.model is None:
            raise RuntimeError("FieldLearner.predict called before fit()")
        return self.model.predict(site)


@dataclass(frozen=True, init=False)
class Pipeline:
    """Immutable sequence of :class:`Step` objects; monoid-style composition.

    ``then`` appends a step and returns a new pipeline, leaving ``self``
    untouched; ``run`` threads each step's output into the next input;
    ``describe`` renders the composed chain.  No global state.

    Raises
    - ValueError: if constructed with no steps.
    """

    steps: Tuple[Step, ...]

    def __init__(self, *steps: Step) -> None:
        if not steps:
            raise ValueError("a Pipeline requires at least one Step")
        object.__setattr__(self, "steps", tuple(steps))

    def then(self, step: Step) -> "Pipeline":
        """Return a new Pipeline with ``step`` appended; ``self`` is unchanged."""
        return Pipeline(*self.steps, step)

    def run(self, data: Any) -> Any:
        """Thread ``data`` through every step in order; return the last output."""
        for step in self.steps:
            data = step.run(data)
        return data

    def describe(self) -> str:
        """Render the composed chain as ``name(detail) -> name(detail)``."""
        return " -> ".join(step.describe() for step in self.steps)


def sites_step(radius: int) -> Step:
    """Step producing the IVM ball sites of shell ``radius`` (ignores input).

    Parameters
    - radius: non-negative shell radius; the ball holds
      ``1 + sum_{k=1..radius} (10k**2 + 2)`` sites.

    Returns
    - Step: emits a tuple of canonical :class:`Quadray` sites.
    """
    source = LatticeBall(radius)
    return Step(name="sites", fn=lambda _data: source.sites(),
                detail=f"radius={radius}")


def learn_step(lam: float, seed: int, radius: int = 3) -> Step:
    """Step fitting an :class:`IVMField` by Laplacian-regularized learning.

    Consumes a sequence of lattice sites (e.g. the output of
    :func:`sites_step`) as the observation pool, or the whole ball when
    the input is ``None``.  Observations target the embedded z coordinate
    of each sampled site; sampling is fixed by ``seed``.

    Parameters
    - lam: non-negative Laplacian regularization strength.
    - seed: RNG seed fixing the deterministic observation sample.
    - radius: shell radius of the field's lattice ball.

    Returns
    - Step: emitting the fitted :class:`IVMField`.
    """
    learner = FieldLearner(radius=radius, lam=lam, seed=seed)
    return Step(name="learn", fn=learner.fit,
                detail=f"lam={lam}, seed={seed}, radius={radius}")


def dynamics_step(params: DynamicsParams, T: int = 5) -> Step:
    """Step simulating the discrete IVM dynamics in ``params`` (ignores input).

    Parameters
    - params: dynamics parameters (kind, alpha, seed, radius).
    - T: number of update steps (non-negative).

    Returns
    - Step: emitting the :class:`ivm_dynamics.Trajectory` record.
    """

    def run_dynamics(_data: Any) -> Trajectory:
        return simulate(T, params)

    return Step(name="dynamics", fn=run_dynamics,
                detail=f"kind={params.kind}, alpha={params.alpha}, T={T}")