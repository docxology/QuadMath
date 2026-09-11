"""Training/testing methodology for the IVM lattice learners.

Honest skill assessment for machine learning on a lattice requires
explicit held-out structure: spatial autocorrelation makes naive
resampling optimistic, and time-ordered dynamics data must never be
shuffled.  This module provides the four standard evaluation surfaces for
the learners of :mod:`quadmath.lattice.ivm_field` and :mod:`quadmath.lattice.ivm_dynamics`:

- :func:`kfold_site_splits` — deterministic seeded k-fold partition of
  observed lattice sites into disjoint, exhaustive train/test pairs.
- :func:`cross_validate_field` — k-fold cross-validation of the
  Laplacian-regularized field learner (:meth:`quadmath.lattice.ivm_field.IVMField.learn`):
  the regularization strength is chosen on held-out folds only and the
  final model is refit on all observed data (honest model selection).
- :func:`trajectory_train_test` — temporal (first-frac) train/test split
  for dynamics parameter identification with
  :func:`quadmath.lattice.ivm_dynamics.fit_trajectory`, scored by multi-step continuation
  MSE on the held-out window plus a teacher-forced one-step-ahead error.
- :func:`learning_curve` — held-out MSE as a function of the fraction of
  observed sites, the classic data-coverage curve on a lattice.

Everything is ``numpy`` only (no ML frameworks) and fully deterministic
for fixed seeds; the split randomness is confined to :func:`kfold_site_splits`
and :func:`learning_curve` (fold/subset assignment), never to the fits
themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quadmath.lattice.ivm_dynamics import DynamicsParams, IVMLattice, fit_trajectory, simulate, step
from quadmath.lattice.ivm_field import IVMField, quadray_shell_norm
from quadmath.core.quadray import Quadray

__all__ = [
    "CrossValidationResult",
    "LearningCurveResult",
    "TrajectorySplitResult",
    "cross_validate_field",
    "enclosing_radius",
    "kfold_site_splits",
    "learning_curve",
    "trajectory_train_test",
]


@dataclass(frozen=True)
class CrossValidationResult:
    """K-fold cross-validation table for the IVM field learner.

    - lam_grid: regularization candidates, as given.
    - fold_mse: held-out MSE per fold, one row (grid-length tuple) per fold.
    - mean_mse: per-candidate mean held-out MSE over folds.
    - std_mse: per-candidate population standard deviation over folds.
    - best_lam: candidate minimizing ``mean_mse`` (earliest on ties).
    - best_mean_mse: its mean held-out MSE.
    - refit_field: field refit with ``best_lam`` on ALL observed sites.
    - refit_mse: in-sample MSE of the refit over all observed sites.
    """

    lam_grid: Tuple[float, ...]
    fold_mse: Tuple[Tuple[float, ...], ...]
    mean_mse: Tuple[float, ...]
    std_mse: Tuple[float, ...]
    best_lam: float
    best_mean_mse: float
    refit_field: IVMField
    refit_mse: float


@dataclass(frozen=True)
class TrajectorySplitResult:
    """Outcome of a temporal train/test evaluation of dynamics identification.

    - n_train_rows: leading snapshot rows used for identification (>= 2).
    - n_test_rows: held-out trailing snapshot rows (>= 1).
    - best_alpha: coupling identified on the training prefix only.
    - train_mse: fit MSE over the training prefix (from fit_trajectory).
    - test_mse: multi-step continuation MSE over the held-out rows,
      re-simulating from the observed initial condition.
    - one_step_mse: teacher-forced one-step-ahead MSE over held-out
      transitions; the first transition starts from the last training row.
    - kind: dynamics kind used for identification and scoring.
    """

    n_train_rows: int
    n_test_rows: int
    best_alpha: float
    train_mse: float
    test_mse: float
    one_step_mse: float
    kind: str


@dataclass(frozen=True)
class LearningCurveResult:
    """Data-coverage curve for the IVM field learner.

    - train_fracs: requested training fractions, as given.
    - train_sizes: realized training-set sizes (nested prefixes of one
      seeded permutation of the observed sites).
    - train_mse: in-sample MSE of the fit on each training subset.
    - test_mse: held-out MSE on the complement of each training subset.
    """

    train_fracs: Tuple[float, ...]
    train_sizes: Tuple[int, ...]
    train_mse: Tuple[float, ...]
    test_mse: Tuple[float, ...]


def enclosing_radius(sites: Sequence[Quadray]) -> int:
    """Return the smallest IVM ball radius containing every given site.

    Parameters
    - sites: Lattice sites (any projective representatives; normalized
      internally by :func:`ivm_field.quadray_shell_norm`).

    Returns
    - int: Smallest non-negative ``R`` such that
      ``quadray_shell_norm(q) <= 2 * R`` for every site, i.e. every site
      lies inside the ball enumerated by ``IVMField.lattice_ball(R)``.

    Raises
    - ValueError: If ``sites`` is empty, or if a site is not an IVM
      lattice site (propagated from :func:`ivm_field.quadray_shell_norm`).
    """
    if len(sites) == 0:
        raise ValueError("at least one site is required")
    return max(quadray_shell_norm(q) for q in sites) // 2


def _normalized_distinct(sites: Sequence[Quadray]) -> List[Quadray]:
    """Normalize sites and reject duplicate lattice points."""
    normalized = [q.normalize() for q in sites]
    if len(set(normalized)) != len(normalized):
        raise ValueError("observed sites must be distinct after normalization")
    return normalized


def kfold_site_splits(
    observed_sites: Sequence[Quadray],
    k: int,
    seed: int,
) -> List[Tuple[List[Quadray], List[Quadray]]]:
    """Partition observed lattice sites into ``k`` seeded folds.

    The sites are normalized, checked for distinctness, shuffled by a
    seeded ``numpy.random.default_rng`` permutation, and cut into ``k``
    contiguous folds of near-equal size (``numpy.array_split`` layout:
    the first ``n % k`` folds receive one extra site).  Each split's test
    set is one fold; its train set is the complement.  Site order within
    each returned list follows the original (normalized) index order, so
    the splits are a deterministic function of ``(sites, k, seed)``.

    Parameters
    - observed_sites: Observed lattice sites (duplicates rejected).
    - k: Number of folds, ``2 <= k <= len(sites)``.
    - seed: RNG seed controlling the fold assignment permutation.

    Returns
    - list of (train_sites, test_sites) pairs, one per fold.

    Raises
    - ValueError: If sites are not distinct after normalization, ``k < 2``,
      or ``k`` exceeds the number of observed sites.
    """
    sites = _normalized_distinct(observed_sites)
    n = len(sites)
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed the number of observed sites ({n})")
    permutation = np.random.default_rng(seed).permutation(n)
    splits: List[Tuple[List[Quadray], List[Quadray]]] = []
    for fold in np.array_split(permutation, k):
        test_idx = sorted(int(i) for i in fold)
        train_idx = sorted(set(range(n)) - set(test_idx))
        splits.append(
            (
                [sites[i] for i in train_idx],
                [sites[i] for i in test_idx],
            )
        )
    return splits


def cross_validate_field(
    field_values: Sequence[float],
    sites: Sequence[Quadray],
    k: int,
    lam_grid: Sequence[float],
    seed: int,
    *,
    radius: Optional[int] = None,
    kernel_width: float = 1.5,
) -> CrossValidationResult:
    """Cross-validate the Laplacian-regularized field learner over k folds.

    For each seeded k-fold split (see :func:`kfold_site_splits`) and each
    candidate ``lam``, a fresh ``IVMField.lattice_ball`` is fit on the
    training sites only (via :meth:`ivm_field.IVMField.learn`) and scored
    by mean squared error on the held-out sites.  The returned
    ``best_lam`` is the candidate minimizing the mean held-out MSE —
    model selection uses validation folds only.  The returned
    ``refit_field`` is the final model: a fresh ball refit with
    ``best_lam`` on ALL observed sites, matching the deployment setting
    where every observation is available.

    Parameters
    - field_values: Observed scalar values, one per site.
    - sites: Observed lattice sites (duplicates rejected).
    - k: Number of folds, ``2 <= k <= len(sites)``.
    - lam_grid: Regularization candidates (non-empty); negatives are
      rejected by :meth:`ivm_field.IVMField.learn`.
    - seed: RNG seed for the fold assignment.
    - radius: Lattice ball radius; ``None`` infers the smallest enclosing
      ball (:func:`enclosing_radius`).
    - kernel_width: Positive kernel width ``tau`` (in graph hops).

    Returns
    - CrossValidationResult: per-fold and aggregated MSE table, the
      selected ``best_lam``, and the refit final field.

    Raises
    - ValueError: If lengths differ, sites are empty or not distinct after
      normalization, ``k`` is out of range (from :func:`kfold_site_splits`),
      or ``lam_grid`` is empty; further validation is propagated from
      :meth:`ivm_field.IVMField.learn` (negative ``lam``, non-positive
      ``kernel_width``, non-finite values, sites outside the ball).
    """
    if len(field_values) != len(sites):
        raise ValueError("sites and field_values must have the same length")
    if len(sites) == 0:
        raise ValueError("at least one observed site is required")
    normalized = _normalized_distinct(sites)
    grid = tuple(float(lam) for lam in lam_grid)
    if not grid:
        raise ValueError("lam_grid must contain at least one candidate")
    ball_radius = enclosing_radius(normalized) if radius is None else int(radius)
    values_by_site: Dict[Quadray, float] = dict(zip(normalized, (float(v) for v in field_values)))

    fold_mse: List[Tuple[float, ...]] = []
    for train_sites, test_sites in kfold_site_splits(normalized, k, seed):
        row: List[float] = []
        for lam in grid:
            ball = IVMField.lattice_ball(ball_radius)
            ball.learn(
                train_sites,
                [values_by_site[q] for q in train_sites],
                lam=lam,
                kernel_width=kernel_width,
            )
            row.append(ball.score(test_sites, [values_by_site[q] for q in test_sites]))
        fold_mse.append(tuple(row))

    table = np.asarray(fold_mse, dtype=float)
    mean_mse = tuple(float(m) for m in np.mean(table, axis=0))
    std_mse = tuple(float(s) for s in np.std(table, axis=0))
    best_position = int(np.argmin(mean_mse))
    best_lam = grid[best_position]
    refit_field = IVMField.lattice_ball(ball_radius)
    refit_field.learn(
        normalized,
        [values_by_site[q] for q in normalized],
        lam=best_lam,
        kernel_width=kernel_width,
    )
    refit_mse = refit_field.score(normalized, [values_by_site[q] for q in normalized])
    return CrossValidationResult(
        lam_grid=grid,
        fold_mse=tuple(fold_mse),
        mean_mse=mean_mse,
        std_mse=std_mse,
        best_lam=best_lam,
        best_mean_mse=mean_mse[best_position],
        refit_field=refit_field,
        refit_mse=refit_mse,
    )


def trajectory_train_test(
    observed: np.ndarray,
    split_frac: float,
    lattice: IVMLattice,
    *,
    kind: str = "heat",
    grid: Optional[Sequence[float]] = None,
    refine_rounds: int = 2,
) -> TrajectorySplitResult:
    """Identify dynamics parameters on a training prefix, score the held-out suffix.

    The snapshot matrix ``observed`` (shape ``(T+1, N)``, row 0 the initial
    condition) is split temporally: the first ``split_frac`` fraction of
    rows train :func:`ivm_dynamics.fit_trajectory`, the trailing rows are
    held out.  No shuffling is performed — for time series, random splits
    would interpolate between adjacent snapshots and hide forecast error.
    The split point is the rounded fraction, clamped so training keeps at
    least two rows (initial condition plus one snapshot) and testing keeps
    at least one row.

    Two held-out errors are reported.  The multi-step continuation MSE
    re-simulates from the observed initial condition with the identified
    coupling and scores only the held-out rows; for a wrong model class
    its error compounds with horizon.  The one-step-ahead MSE is
    teacher-forced: each held-out transition is predicted one step from
    the true previous state (the first transition starts from the last
    training row), isolating local prediction error from error
    accumulation.

    Parameters
    - observed: ``(T+1, N)`` snapshot matrix including the initial
      condition; the width must match ``lattice.size`` (checked by
      :func:`ivm_dynamics.fit_trajectory`).
    - split_frac: Training fraction in the open interval ``(0, 1)``.
    - lattice: Lattice the snapshots live on.
    - kind: Dynamics kind, ``"heat"`` or ``"majority"``.
    - grid: Candidate couplings for :func:`ivm_dynamics.fit_trajectory`;
      ``None`` uses ``numpy.linspace(0, 1, 11)``.
    - refine_rounds: Interval re-gridding rounds passed through to
      :func:`ivm_dynamics.fit_trajectory`.

    Returns
    - TrajectorySplitResult: split sizes, identified coupling, training
      prefix MSE, held-out continuation MSE, and one-step-ahead MSE.

    Raises
    - ValueError: If ``observed`` is not two-dimensional, has fewer than
      three rows (two train + one test), or ``split_frac`` is outside
      ``(0, 1)``; further validation is propagated from
      :func:`ivm_dynamics.fit_trajectory` and :func:`ivm_dynamics.step`.
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2:
        raise ValueError("observed must have shape (T+1, N)")
    rows = obs.shape[0]
    if rows < 3:
        raise ValueError("observed needs at least two training snapshots and one test snapshot")
    frac = float(split_frac)
    if not 0.0 < frac < 1.0:
        raise ValueError("split_frac must lie strictly between 0 and 1")
    n_train = min(max(int(round(frac * rows)), 2), rows - 1)
    if grid is None:
        grid_values: List[float] = [float(g) for g in np.linspace(0.0, 1.0, 11)]
    else:
        grid_values = [float(g) for g in grid]
    fit = fit_trajectory(obs[:n_train], grid_values, lattice, kind=kind, refine_rounds=refine_rounds)
    params = DynamicsParams(kind=kind, alpha=fit.best_alpha)
    traj = simulate(rows - 1, params, lattice=lattice, u0=obs[0])
    predicted = np.asarray(traj.fields[1:], dtype=float)
    test_mse = float(np.mean((predicted[n_train - 1:] - obs[n_train:]) ** 2))
    step_errors = [
        (step(obs[t], lattice, params) - obs[t + 1]) ** 2 for t in range(n_train - 1, rows - 1)
    ]
    one_step_mse = float(np.mean(step_errors))
    return TrajectorySplitResult(
        n_train_rows=n_train,
        n_test_rows=rows - n_train,
        best_alpha=fit.best_alpha,
        train_mse=fit.best_mse,
        test_mse=test_mse,
        one_step_mse=one_step_mse,
        kind=kind,
    )


def learning_curve(
    values: Sequence[float],
    sites: Sequence[Quadray],
    train_fracs: Sequence[float],
    seed: int,
    *,
    radius: Optional[int] = None,
    lam: float = 1e-2,
    kernel_width: float = 1.5,
) -> LearningCurveResult:
    """Trace held-out MSE as a function of the fraction of observed sites.

    One seeded permutation of the observed sites is drawn; for each
    requested fraction ``p`` the first ``m(p) = round(p * n)`` sites of
    that permutation (clamped to ``[1, n-1]``) form the training subset
    and the remainder the held-out complement.  The subsets are nested by
    construction, so the curve is a proper data-coverage curve: growing
    the training prefix never removes earlier observations.  For each
    fraction a fresh ``IVMField.lattice_ball`` is fit on the training
    subset (via :meth:`ivm_field.IVMField.learn`) and scored in-sample
    and on the complement.

    Parameters
    - values: Observed scalar values, one per site.
    - sites: Observed lattice sites (duplicates rejected; at least two).
    - train_fracs: Training fractions, each strictly between 0 and 1.
    - seed: RNG seed for the shared subset permutation.
    - radius: Lattice ball radius; ``None`` infers the smallest enclosing
      ball (:func:`enclosing_radius`).
    - lam: Regularization strength passed to
      :meth:`ivm_field.IVMField.learn`.
    - kernel_width: Positive kernel width ``tau`` (in graph hops).

    Returns
    - LearningCurveResult: fractions, realized sizes, and per-fraction
      in-sample and held-out MSE, in the caller's fraction order.

    Raises
    - ValueError: If lengths differ, sites are empty, fewer than two
      sites, not distinct after normalization, ``train_fracs`` is empty,
      or a fraction lies outside ``(0, 1)``; further validation is
      propagated from :meth:`ivm_field.IVMField.learn`.
    """
    if len(values) != len(sites):
        raise ValueError("sites and values must have the same length")
    if len(sites) == 0:
        raise ValueError("at least one observed site is required")
    normalized = _normalized_distinct(sites)
    n = len(normalized)
    if n < 2:
        raise ValueError("learning_curve needs at least two observed sites")
    fracs = tuple(float(p) for p in train_fracs)
    if not fracs:
        raise ValueError("train_fracs must be non-empty")
    for p in fracs:
        if not 0.0 < p < 1.0:
            raise ValueError("train fractions must lie strictly between 0 and 1")
    ball_radius = enclosing_radius(normalized) if radius is None else int(radius)
    values_by_site: Dict[Quadray, float] = dict(zip(normalized, (float(v) for v in values)))
    permutation = np.random.default_rng(seed).permutation(n)

    sizes: List[int] = []
    train_mse: List[float] = []
    test_mse: List[float] = []
    for p in fracs:
        m = min(max(int(round(p * n)), 1), n - 1)
        train_sites = [normalized[i] for i in sorted(int(i) for i in permutation[:m])]
        test_sites = [normalized[i] for i in sorted(int(i) for i in permutation[m:])]
        ball = IVMField.lattice_ball(ball_radius)
        ball.learn(
            train_sites,
            [values_by_site[q] for q in train_sites],
            lam=lam,
            kernel_width=kernel_width,
        )
        sizes.append(m)
        train_mse.append(ball.score(train_sites, [values_by_site[q] for q in train_sites]))
        test_mse.append(ball.score(test_sites, [values_by_site[q] for q in test_sites]))
    return LearningCurveResult(
        train_fracs=fracs,
        train_sizes=tuple(sizes),
        train_mse=tuple(train_mse),
        test_mse=tuple(test_mse),
    )
