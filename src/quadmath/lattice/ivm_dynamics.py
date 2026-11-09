"""Discrete dynamical systems and coupling learning on the IVM lattice.

Scalar fields evolve over integer sites of the IVM lattice graph built
from canonical quadray representatives (Fuller.4D; see `quadmath/core/quadray.py`).
The module provides:

- lattice construction: `ball_sites`, `make_lattice` — the finite ball of
  canonical quadray sites with edges given by the 12-around-one IVM
  neighbor shifts (all permutations of (2, 1, 1, 0));
- updates: `heat_step` (symmetric-normalized energy contraction),
  `conservative_heat_step` (mass-preserving concentration transport), and
  `majority_step` (exact integral round-half-to-even neighbor averaging,
  range-preserving but NOT sum-of-squares monotone);
- simulation: `step`, `simulate(T, params)` — deterministic given the
  fixed seed in `DynamicsParams`;
- learning: `fit_trajectory(observed, grid, ...)` — gradient-free grid
  search with coordinate refinement minimizing trajectory MSE;
- observables: exact integral `sum_of_squares`, `is_nonincreasing`,
  `graph_laplacian` and `dirichlet_energy`; their graph identities and
  floating/integer boundaries are exercised by the lattice regressions.

Only numpy (plus `quadmath.core.quadray`) is imported at module level; matplotlib and
`quadmath.paths` are imported lazily inside `render_dynamics_demo`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quadmath.core.quadray import (
    DEFAULT_EMBEDDING, IVM_NEIGHBOR_MOVES, Quadray, _exact_fraction,
    _float_metric, squared_norm_exact, to_xyz,
)


def neighbor_shifts() -> List[Quadray]:
    """Return the 12 canonical IVM neighbor shifts as quadray deltas.

    The move set is all distinct permutations of (2, 1, 1, 0). Under
    `DEFAULT_EMBEDDING` each shift has integer squared radius 8, i.e. the
    close-packing (sphere-touching) distance of the 12-around-one cluster.
    """
    return [Quadray(*move) for move in IVM_NEIGHBOR_MOVES]


def site_radius_sq(q: Quadray) -> int:
    """Integer squared IVM radius of a quadray site (squared embedding norm).

    Parameters
    - q: quadray site (any representative; the value is invariant under
      (1,1,1,1) shifts because that vector lies in the embedding kernel).

    Returns
    - int: squared Euclidean norm of the embedded site.
    """
    return round(squared_norm_exact(q, DEFAULT_EMBEDDING))


def ball_sites(radius: int) -> List[Quadray]:
    """Enumerate canonical quadray sites with squared IVM radius <= radius**2.

    Sites are canonical quadray representatives: non-negative integer
    components with at least one zero, and component sum divisible by 4
    (the IVM close-packed sites; the other residue classes are tetrahedral
    and octahedral voids). For a canonical nonzero site the squared radius is
    at least twice the square of its largest component, so enumerating
    components in [0, radius] covers the whole ball.

    Parameters
    - radius: inclusive radius bound; must be non-negative.

    Returns
    - List[Quadray]: sites sorted by (squared radius, components) for
      deterministic ordering.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    r_cap = radius * radius
    found: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for a in range(radius + 1):
        for b in range(radius + 1):
            for c in range(radius + 1):
                for d in range(radius + 1):
                    q = Quadray(a, b, c, d)
                    if min(q.as_tuple()) != 0 or sum(q.as_tuple()) % 4 != 0:
                        continue  # canonical IVM sites only
                    r2 = site_radius_sq(q)
                    if r2 <= r_cap:
                        found.append((r2, q.as_tuple()))
    found.sort()
    return [Quadray(*t) for _, t in found]


@dataclass(frozen=True)
class IVMLattice:
    """Finite IVM lattice ball with neighbor adjacency and diffusion operators.

    - adjacency: (N, N) 0/1 matrix; edge iff one IVM neighbor shift maps
      one site onto the other (canonical renormalization does not move the
      embedded point).
    - degrees: (N,) neighbor counts; boundary/isolated sites may be 0.
    - diffusion: S = D^{-1/2} A D^{-1/2} with zero rows for isolated
      sites; symmetric with spectrum in [-1, 1] (basis of the heat lemma).
    - average: row-stochastic plain neighbor average P; identity row for
      isolated sites.
    """

    radius: int
    sites: Tuple[Quadray, ...]
    adjacency: np.ndarray
    degrees: np.ndarray
    diffusion: np.ndarray
    average: np.ndarray
    _edge_sources: np.ndarray = field(init=False, repr=False, compare=False)
    _edge_targets: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sources, targets = np.nonzero(self.adjacency)
        upper = sources < targets
        object.__setattr__(self, "_edge_sources", sources[upper])
        object.__setattr__(self, "_edge_targets", targets[upper])

    @property
    def size(self) -> int:
        """Number of lattice sites."""
        return len(self.sites)


def make_lattice(radius: int) -> IVMLattice:
    """Build the finite IVM lattice ball of the given radius.

    Only the FCC component-sum-zero-mod-four classes are included; the three
    void cosets are excluded. Radius 3 gives the origin and its twelve nearest
    neighbors. Operators also support explicitly constructed disconnected graphs.
    """
    sites = tuple(ball_sites(radius))
    n = len(sites)
    index = {q.as_tuple(): i for i, q in enumerate(sites)}
    adjacency = np.zeros((n, n), dtype=np.uint8)
    for i, q in enumerate(sites):
        for shift in neighbor_shifts():
            j = index.get(q.add(shift).normalize().as_tuple())
            if j is not None:
                adjacency[i, j] = 1
    degrees = adjacency.sum(axis=1).astype(np.int64)
    deg_safe = np.maximum(degrees, 1).astype(float)
    inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(deg_safe), 0.0)
    diffusion = adjacency.astype(float) * inv_sqrt[:, None] * inv_sqrt[None, :]
    average = adjacency.astype(float) / deg_safe[:, None]
    for i in np.flatnonzero(degrees == 0):
        average[i, i] = 1.0  # isolated sites hold their value
    return IVMLattice(
        radius=radius,
        sites=sites,
        adjacency=adjacency,
        degrees=degrees,
        diffusion=diffusion,
        average=average,
    )


def sum_of_squares(u: np.ndarray) -> float | int:
    """Return the flattened squared norm, exact for integral fields.

    Parameters
    - u: field values of any shape.

    Returns
    - float | int: sum of squared values; integer fields use unbounded
      Python arithmetic, while real-valued fields use floating arithmetic.
    """
    arr = np.asarray(u)
    if arr.dtype.kind in "biu" or (
        arr.dtype.kind == "O" and all(isinstance(value, Integral) for value in arr.flat)
    ):
        return sum(int(value) ** 2 for value in arr.flat)
    flattened = np.asarray(arr, dtype=float).ravel()
    return float(flattened @ flattened)


def is_nonincreasing(values: Sequence[float | int], tol: float = 1e-12) -> bool:
    """True iff no consecutive pair of values increases by more than tol.

    Parameters
    - values: sequence of observables (e.g. `Trajectory.sos`).
    - tol: tolerance absorbing floating-point round-off.

    Returns
    - bool: True iff values[t+1] <= values[t] + tol for all t.
    """
    for prev, nxt in zip(values, values[1:]):
        if not nxt - prev <= tol:
            return False
    return True


def _validated_field(u: np.ndarray, lattice: IVMLattice) -> np.ndarray:
    """Return the field as an array after shape validation."""
    arr = np.asarray(u)
    if arr.ndim != 1:
        raise ValueError("field must be one-dimensional")
    if arr.shape[0] != lattice.size:
        raise ValueError("field length must match lattice size")
    return arr


def _validated_alpha(alpha: float) -> float:
    """Validate and return the coupling parameter as a float."""
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("coupling alpha must lie in [0, 1]")
    return value


def graph_laplacian(u: np.ndarray, lattice: IVMLattice) -> np.ndarray:
    """Apply the unnormalized graph Laplacian L = D - A.

    Parameters
    - u: real field of length ``lattice.size``.
    - lattice: finite undirected IVM graph.

    Returns
    - np.ndarray: net outward edge differences; isolated sites return zero.
      Edge differences avoid cancellation between separately formed D u and A u.
    """
    arr = np.asarray(_validated_field(u, lattice), dtype=float)
    differences = arr[lattice._edge_sources] - arr[lattice._edge_targets]
    result = np.zeros(lattice.size, dtype=float)
    np.add.at(result, lattice._edge_sources, differences)
    np.add.at(result, lattice._edge_targets, -differences)
    return result


def dirichlet_energy(u: np.ndarray, lattice: IVMLattice) -> float:
    """Return E(u) = one half of the squared differences over undirected edges.

    Parameters
    - u: real field of length ``lattice.size``.
    - lattice: finite undirected IVM graph.

    Returns
    - float: ``u.T @ L @ u / 2``; zero precisely for fields constant on
      each connected component, up to floating-point arithmetic.
    """
    arr = np.asarray(_validated_field(u, lattice), dtype=float)
    differences = arr[lattice._edge_sources] - arr[lattice._edge_targets]
    return float((0.5 * differences) @ differences)


def conservative_heat_step(
    u: np.ndarray, lattice: IVMLattice, alpha: float
) -> np.ndarray:
    """Diffuse equal-volume concentrations without boundary mass leakage.

    The update is ``(I - alpha * L / d_max) u`` for ``L = D - A``.
    For alpha in [0, 1] it is symmetric and doubly stochastic: total mass,
    constants and isolated values are preserved, the range cannot expand,
    and both squared norm and Dirichlet energy are non-increasing.
    These are finite-graph identities, not a continuum error guarantee.

    Parameters
    - u: finite real field of length ``lattice.size``.
    - lattice: finite undirected IVM graph.
    - alpha: dimensionless coupling in [0, 1].

    Returns
    - np.ndarray: updated concentration field. Undirected edges are cached
      once per lattice; ordinary updates use O(N + E) work and O(N) workspace.
      Fluxes accumulate before the state changes. Overflow/underflow-risk
      fields use exact binary-rational weights and accumulation before
      converting the representable result back to floating point. For
      magnitude bound M, incident absolute flux sums are at most 2*M and
      adding the old state is bounded by 3*M before cancellation. The fast
      path therefore reserves headroom with M <= float_max/4.
    """
    arr = np.asarray(_validated_field(u, lattice), dtype=float)
    value = _validated_alpha(alpha)
    maximum_degree = int(lattice.degrees.max(initial=0))
    if maximum_degree == 0 or value == 0.0:
        return arr.copy()
    weight = value / maximum_degree
    limits = np.finfo(float)
    magnitudes = np.abs(arr)
    if (
        weight < limits.tiny
        or np.any(magnitudes > limits.max / 4)
        or np.any((magnitudes != 0) & (magnitudes < limits.tiny / weight))
    ):
        exact_weight = Fraction(value) / maximum_degree
        native = [Fraction(float(entry)) for entry in arr]
        weighted = [exact_weight * entry for entry in native]
        result = [
            entry * (1 - exact_weight * int(degree))
            for entry, degree in zip(native, lattice.degrees)
        ]
        for source, target in zip(lattice._edge_sources, lattice._edge_targets):
            result[source] += weighted[target]
            result[target] += weighted[source]
        return np.asarray([float(entry) for entry in result], dtype=float)
    fluxes = weight * (arr[lattice._edge_sources] - arr[lattice._edge_targets])
    changes = np.zeros(lattice.size, dtype=float)
    np.add.at(changes, lattice._edge_sources, -fluxes)
    np.add.at(changes, lattice._edge_targets, fluxes)
    return arr + changes


def heat_step(u: np.ndarray, lattice: IVMLattice, alpha: float) -> np.ndarray:
    """One heat-diffusion step u <- (1 - alpha) u + alpha S u.

    S = `lattice.diffusion` is the symmetric normalized adjacency with
    spectrum in [-1, 1], so the update operator (1 - alpha) I + alpha S has
    spectrum in [1 - 2 alpha, 1] subset [-1, 1] for alpha in [0, 1]; hence
    `sum_of_squares` is NON-INCREASING at every step, for any lattice and
    any state (lemma tested per step in the test suite). This guarantee is
    specific to the symmetric normalization S: plain row-average diffusion
    on truncated balls is NOT covered by the claim and is not used here.

    Parameters
    - u: 1-D field of length `lattice.size` (float values).
    - lattice: finite IVM lattice.
    - alpha: coupling in [0, 1] (0 = identity, 1 = pure diffusion).

    Returns
    - np.ndarray: updated float field.
    """
    arr = np.asarray(_validated_field(u, lattice), dtype=float)
    value = _validated_alpha(alpha)
    return arr + value * (lattice.diffusion @ arr - arr)


def majority_step(u: np.ndarray, lattice: IVMLattice, alpha: float) -> np.ndarray:
    """One rounded-averaging ("majority") step on integer states.

    u_next(s) = rint((1 - alpha) u(s) + alpha mean_{n in N(s)} u(n)), with
    identity rows for isolated sites and round-half-to-even ties. Every new
    value is a rounded convex combination of the site and its neighbors, so
    the field maximum is non-increasing and the minimum non-decreasing
    (range containment, tested per step). We deliberately do NOT claim
    sum-of-squares monotonicity here: rounding can increase it, and the
    test suite pins a demonstrating case.

    Parameters
    - u: 1-D integer-valued field of length `lattice.size`.
    - lattice: finite IVM lattice.
    - alpha: coupling in [0, 1] (0 = hold state, 1 = neighbor majority).

    Returns
    - np.ndarray: updated integer field; int64 when representable, otherwise
      an object array of unbounded Python integers. The binary value of the
      float coupling is applied exactly before round-half-to-even.
    """
    arr = _validated_field(u, lattice)
    value = _validated_alpha(alpha)
    try:
        integers = [int(entry) for entry in arr]
    except (ValueError, OverflowError) as exc:
        raise ValueError("majority update requires integer-valued states") from exc
    if any(integer != entry for integer, entry in zip(integers, arr)):
        raise ValueError("majority update requires integer-valued states")
    neighbor_sums = [0] * lattice.size
    for source, target in zip(lattice._edge_sources, lattice._edge_targets):
        neighbor_sums[source] += integers[target]
        neighbor_sums[target] += integers[source]
    numerator, denominator = value.as_integer_ratio()
    rounded = []
    for index, current in enumerate(integers):
        degree = int(lattice.degrees[index])
        if degree == 0:
            rounded.append(current)
        else:
            weighted_sum = (
                (denominator - numerator) * current * degree
                + numerator * neighbor_sums[index]
            )
            rounded.append(round(Fraction(weighted_sum, denominator * degree)))
    limits = np.iinfo(np.int64)
    dtype = np.int64 if all(limits.min <= entry <= limits.max for entry in rounded) else object
    return np.asarray(rounded, dtype=dtype)


@dataclass(frozen=True)
class DynamicsParams:
    """Parameters of a discrete IVM lattice dynamics run.

    - kind: "heat" (symmetric-normalized energy diffusion), "conservative"
      (mass-preserving concentration diffusion), or "majority" (exact
      integral rounded averaging).
    - alpha: coupling in [0, 1].
    - seed: RNG seed for synthetic initial fields (determinism contract).
    - radius: lattice ball radius used when no lattice is supplied.
    """

    kind: str = "heat"
    alpha: float = 0.35
    seed: int = 0
    radius: int = 3


def step(u: np.ndarray, lattice: IVMLattice, params: DynamicsParams) -> np.ndarray:
    """Apply one update of the dynamics in `params` to the field.

    Parameters
    - u: 1-D field of length `lattice.size`.
    - lattice: finite IVM lattice.
    - params: dynamics parameters (kind and coupling).

    Returns
    - np.ndarray: updated field (float for "heat", int for "majority").
    """
    if params.kind == "heat":
        return heat_step(u, lattice, params.alpha)
    if params.kind == "conservative":
        return conservative_heat_step(u, lattice, params.alpha)
    if params.kind == "majority":
        return majority_step(u, lattice, params.alpha)
    raise ValueError(f"unknown dynamics kind: {params.kind!r}")


@dataclass
class Trajectory:
    """Deterministic simulation record.

    - fields: snapshots for t = 0..T (fields[0] is the initial state).
    - sos: `sum_of_squares` of each snapshot.
    """

    lattice: IVMLattice
    params: DynamicsParams
    fields: List[np.ndarray]
    sos: List[float | int]


def simulate(
    T: int,
    params: DynamicsParams,
    lattice: Optional[IVMLattice] = None,
    u0: Optional[np.ndarray] = None,
) -> Trajectory:
    """Simulate T updates; deterministic given `params` (fixed seed).

    With u0=None the initial field is drawn from numpy's seeded generator:
    uniform(-2, 2) floats for real modes, integers in [-3, 3] for "majority".
    T = 0 returns the initial field alone; T < 0 raises.

    Parameters
    - T: number of update steps (non-negative).
    - params: dynamics parameters (kind, alpha, seed, radius).
    - lattice: optional prebuilt lattice; defaults to make_lattice(params.radius).
    - u0: optional initial field overriding the seeded draw.

    Returns
    - Trajectory: fields and sum-of-squares for t = 0..T.
    """
    if T < 0:
        raise ValueError("horizon T must be non-negative")
    if lattice is None:
        lattice = make_lattice(params.radius)
    if u0 is None:
        rng = np.random.default_rng(params.seed)
        if params.kind == "majority":
            u0 = rng.integers(-3, 4, size=lattice.size)
        else:
            u0 = rng.uniform(-2.0, 2.0, size=lattice.size)
    arr = _validated_field(np.asarray(u0), lattice).copy()
    fields: List[np.ndarray] = [arr]
    for _ in range(T):
        fields.append(step(fields[-1], lattice, params))
    return Trajectory(
        lattice=lattice,
        params=params,
        fields=fields,
        sos=[sum_of_squares(field) for field in fields],
    )


@dataclass(frozen=True)
class FitResult:
    """Outcome of gradient-free coupling identification.

    - best_alpha: coupling minimizing trajectory MSE over evaluated candidates.
    - best_mse: its mean squared error.
    - mses: every evaluated (candidate, mse) pair.
    - kind: dynamics kind used for re-simulation.
    """

    best_alpha: float
    best_mse: float
    mses: Dict[float, float]
    kind: str


def _trajectory_mse(
    predicted: np.ndarray, observed: np.ndarray, kind: str
) -> float:
    """Score exact count residuals or range-scaled real residuals.

    Majority observations may be integral floats or fractional noisy values.
    Subtraction, squaring and averaging precede the final floating boundary;
    an unrepresentably bad candidate scores infinity without aborting search.
    """
    if kind == "majority":
        reference = np.asarray(observed)
        differences = (
            _exact_fraction(actual) - _exact_fraction(expected)
            for actual, expected in zip(np.asarray(predicted).flat, reference.flat)
        )
        total = sum((error * error for error in differences), Fraction())
        return _float_metric(total / reference.size)
    differences = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    scale = float(np.max(np.abs(differences)))
    if scale == 0.0 or not np.isfinite(scale):
        return scale
    normalized = differences / scale
    return scale * (scale * float(np.mean(normalized * normalized)))


def fit_trajectory(
    observed: np.ndarray,
    grid: Sequence[float],
    lattice: IVMLattice,
    kind: str = "heat",
    refine_rounds: int = 2,
) -> FitResult:
    """Identify the coupling alpha from an observed trajectory (gradient-free).

    `observed` has shape (T+1, N): row 0 is the initial condition, rows
    1..T the observed snapshots. Each candidate alpha in `grid` is scored
    by re-simulating from the initial condition and averaging the squared
    error against the observed snapshots; the best candidate is then
    refined by re-gridding the interval between its grid neighbors
    `refine_rounds` times (grid/coordinate search, no gradients). Ties
    resolve to the smallest alpha. The result is a global optimum over the
    evaluated candidates only; no convergence beyond refinement resolution
    is claimed.

    Parameters
    - observed: (T+1, N) snapshot matrix including the initial condition.
    - grid: candidate couplings, each in [0, 1]; must be non-empty.
    - lattice: lattice the observed snapshots live on.
    - kind: "heat", "conservative" or "majority".
    - refine_rounds: number of interval re-gridding rounds (0 disables).

    Returns
    - FitResult: best coupling, its MSE, and the full MSE table.
    """
    grid_values = [float(g) for g in grid]
    if not grid_values:
        raise ValueError("grid must contain at least one candidate")
    obs = np.asarray(observed)
    if obs.ndim != 2 or obs.shape[0] < 2:
        raise ValueError("observed must have shape (T+1, N) with T >= 1")
    if obs.shape[1] != lattice.size:
        raise ValueError("observed width must match lattice size")
    for g in grid_values:
        if g < 0.0 or g > 1.0:
            raise ValueError("candidate couplings must lie in [0, 1]")
    horizon = obs.shape[0] - 1
    mses: Dict[float, float] = {}

    def _mse(alpha: float) -> float:
        if alpha not in mses:
            params = DynamicsParams(kind=kind, alpha=alpha)
            traj = simulate(horizon, params, lattice=lattice, u0=obs[0])
            mses[alpha] = _trajectory_mse(traj.fields[1:], obs[1:], kind)
        return mses[alpha]

    candidates = sorted(set(grid_values))
    best = min(candidates, key=lambda g: (_mse(g), g))
    for _ in range(refine_rounds):
        i = candidates.index(best)
        lo = candidates[max(i - 1, 0)]
        hi = candidates[min(i + 1, len(candidates) - 1)]
        if lo == hi:
            break
        candidates = sorted(set(candidates) | set(np.linspace(lo, hi, 9).tolist()))
        best = min(candidates, key=lambda g: (_mse(g), g))
    return FitResult(best_alpha=best, best_mse=mses[best], mses=dict(mses), kind=kind)


def render_dynamics_demo(
    output_path: Optional[str] = None,
    *,
    radius: int = 3,
    seed: int = 7,
    alpha: float = 0.35,
    horizon: int = 40,
    fit_alpha: float = 0.3,
    grid_count: int = 21,
    refine_rounds: int = 2,
) -> str:
    """Render the multi-snapshot IVM dynamics demo figure; return its path.

    Panels: heat-field snapshots at t = 0, horizon//2, horizon over the
    embedded lattice; sum-of-squares histories for heat and majority
    updates; the `fit_trajectory` MSE landscape with best and true coupling;
    and the final majority (integer) field. Deterministic via fixed seeds.
    matplotlib and `paths` are imported lazily so the core module stays
    numpy-only.

    Parameters
    - output_path: target PNG path; defaults to
      quadmath/output/figures/ivm_dynamics_demo.png.
    - radius, seed, alpha, horizon: simulation parameters.
    - fit_alpha: true coupling of the synthetic trajectory to identify.
    - grid_count, refine_rounds: `fit_trajectory` search settings.

    Returns
    - str: path of the written figure.
    """
    from quadmath.paths import get_figure_dir
    from quadmath.viz._common import figure_scope, set_axes_equal

    lattice = make_lattice(radius)
    heat = simulate(
        horizon, DynamicsParams(kind="heat", alpha=alpha, seed=seed), lattice
    )
    maj = simulate(
        horizon, DynamicsParams(kind="majority", alpha=alpha, seed=seed), lattice
    )
    observed = np.stack(
        simulate(
            horizon,
            DynamicsParams(kind="heat", alpha=fit_alpha, seed=seed),
            lattice,
            u0=heat.fields[0],
        ).fields
    )
    fit = fit_trajectory(
        observed, np.linspace(0.0, 1.0, grid_count), lattice, refine_rounds=refine_rounds
    )

    coords = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in lattice.sites])
    snapshot_times = (0, horizon // 2, horizon)
    values = [heat.fields[t] for t in snapshot_times]
    vmax = max(float(np.max(np.abs(v))) for v in values + [maj.fields[horizon]])
    with figure_scope(figsize=(13.5, 8.0)) as fig:
        for k, (t, field) in enumerate(zip(snapshot_times, values)):
            ax = fig.add_subplot(2, 3, k + 1, projection="3d")
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=field,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                s=28,
            )
            set_axes_equal(ax)
            ax.set_title(f"normalized heat field, t = {t}")
        ax = fig.add_subplot(2, 3, 4)
        ax.plot(range(horizon + 1), heat.sos, label="symmetric heat (L2 contraction)")
        ax.plot(range(horizon + 1), maj.sos, label="majority (range-preserving)")
        ax.set_xlabel("t")
        ax.set_ylabel("sum of squares")
        ax.legend()
        ax.set_title("energy histories")
        ax = fig.add_subplot(2, 3, 5)
        alphas = sorted(fit.mses)
        ax.semilogy(alphas, np.maximum([fit.mses[a] for a in alphas], 1e-18), "o-", ms=3)
        ax.axvline(fit.best_alpha, color="k", ls="--", lw=1.0, label=f"fit {fit.best_alpha:.4f}")
        ax.axvline(fit_alpha, color="g", ls=":", lw=1.0, label=f"true {fit_alpha}")
        ax.set_xlabel("coupling alpha")
        ax.set_ylabel("trajectory MSE")
        ax.legend()
        ax.set_title("grid-search identification")
        ax = fig.add_subplot(2, 3, 6, projection="3d")
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=maj.fields[horizon],
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            s=28,
        )
        set_axes_equal(ax)
        ax.set_title(f"majority field, t = {horizon}")
        fig.suptitle("IVM lattice dynamics: normalized heat, majority rounding, coupling learning")
        fig.tight_layout()
        if output_path is None:
            target = os.path.join(get_figure_dir(), "ivm_dynamics_demo.png")
        else:
            target = output_path
        fig.savefig(target, dpi=150, bbox_inches="tight")
        return target


__all__ = [
    "DynamicsParams",
    "FitResult",
    "IVMLattice",
    "Trajectory",
    "ball_sites",
    "conservative_heat_step",
    "dirichlet_energy",
    "fit_trajectory",
    "heat_step",
    "graph_laplacian",
    "is_nonincreasing",
    "make_lattice",
    "majority_step",
    "neighbor_shifts",
    "render_dynamics_demo",
    "simulate",
    "site_radius_sq",
    "step",
    "sum_of_squares",
]