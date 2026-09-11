"""Training/testing methodology for the IVM lattice learners.

Honest skill assessment for machine learning on a lattice requires
explicit held-out structure: spatial autocorrelation makes naive
resampling optimistic, and time-ordered dynamics data must never be
shuffled.  This module provides the four standard lattice evaluation
surfaces for the learners of :mod:`quadmath.lattice.ivm_field` and
:mod:`quadmath.lattice.ivm_dynamics`:

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

A plain tabular training surface complements the lattice workflows:

- :func:`three_way_split` — seeded, disjoint train/val/test partition of
  a plain index range.
- :func:`ridge_site_fit` — closed-form ridge regression with intercept,
  solved exactly via centered normal equations.
- :class:`GradientDescentTrainer` — full-batch gradient descent on
  standardized features with a per-iteration loss history.

Everything is ``numpy`` only (no ML frameworks) and fully deterministic
for fixed seeds; the split randomness is confined to :func:`kfold_site_splits`,
:func:`learning_curve`, and :func:`three_way_split` (fold/subset
assignment), never to the fits themselves.
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
    "GradientDescentTrainer",
    "LearningCurveResult",
    "RidgeSiteFit",
    "TrajectorySplitResult",
    "cross_validate_field",
    "enclosing_radius",
    "kfold_site_splits",
    "learning_curve",
    "ridge_site_fit",
    "three_way_split",
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


@dataclass(frozen=True)
class RidgeSiteFit:
    """Closed-form ridge fit of a single linear site model.

    - coefficients: fitted slope vector, one entry per feature column.
    - intercept: fitted bias term, recovered from the centered solve.
    - train_mse: in-sample mean squared error over the training rows.
    """

    coefficients: np.ndarray
    intercept: float
    train_mse: float


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


def three_way_split(
    n: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Split ``range(n)`` into seeded, disjoint train/val/test index lists.

    One seeded ``numpy.random.default_rng`` permutation of ``range(n)`` is
    drawn and cut into three contiguous blocks: the first
    ``round(train_frac * n)`` indices train, the next ``round(val_frac *
    n)`` validate, and the remainder test.  The returned lists are each
    sorted, pairwise disjoint, and cover ``range(n)`` exactly once, so
    the split is a deterministic function of ``(n, train_frac, val_frac,
    seed)``.  For small ``n`` rounding may empty the trailing blocks
    (e.g. ``n = 3`` with the default fractions leaves the test list
    empty).

    Parameters
    - n: Number of indices to partition (``n >= 0``).
    - train_frac: Training fraction, strictly between 0 and 1.
    - val_frac: Validation fraction, strictly between 0 and 1.
    - seed: RNG seed controlling the index permutation.

    Returns
    - (train_indices, val_indices, test_indices) tuple of sorted, pairwise
      disjoint lists whose union is exactly ``range(n)``.

    Raises
    - ValueError: If ``train_frac`` or ``val_frac`` is not strictly
      positive, or ``train_frac + val_frac`` is not below 1.
    """
    if not 0.0 < train_frac:
        raise ValueError(f"train_frac must be strictly positive, got {train_frac}")
    if not 0.0 < val_frac:
        raise ValueError(f"val_frac must be strictly positive, got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac ({train_frac + val_frac}) must stay below 1"
        )
    permutation = np.random.default_rng(seed).permutation(n)
    m_train = int(round(train_frac * n))
    m_val = int(round(val_frac * n))
    train_idx = sorted(int(i) for i in permutation[:m_train])
    val_idx = sorted(int(i) for i in permutation[m_train:m_train + m_val])
    test_idx = sorted(int(i) for i in permutation[m_train + m_val:])
    return train_idx, val_idx, test_idx


def ridge_site_fit(
    features: np.ndarray,
    values: np.ndarray,
    lam: float = 1e-3,
) -> RidgeSiteFit:
    """Fit a closed-form ridge regression with intercept on tabular rows.

    Features and values are centered to zero mean; the ridge normal
    equations ``(Xc^T Xc + lam * I) w = Xc^T yc`` are solved exactly for
    the slope vector ``w``, and the intercept is recovered as
    ``y_mean - x_mean @ w``.  The fit is a deterministic function of
    ``(features, values, lam)``.

    Parameters
    - features: Design matrix of shape ``(n_samples, n_features)``.
    - values: Target vector of shape ``(n_samples,)``.
    - lam: Non-negative ridge penalty on the (centered) coefficients.

    Returns
    - RidgeSiteFit: coefficients, intercept, and in-sample train MSE.

    Raises
    - ValueError: If ``lam`` is negative, ``features`` is not 2-D,
      ``values`` is not 1-D, their sample counts disagree, or no sample
      rows are given.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(values, dtype=float)
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if X.ndim != 2:
        raise ValueError(
            f"features must be 2-D (n_samples, n_features), got ndim={X.ndim}"
        )
    if y.ndim != 1:
        raise ValueError(f"values must be 1-D (n_samples,), got ndim={y.ndim}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"features has {X.shape[0]} rows but values has {y.shape[0]} entries"
        )
    if X.shape[0] == 0:
        raise ValueError("ridge_site_fit needs at least one sample row")
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    centered = X - x_mean
    centered_values = y - y_mean
    width = X.shape[1]
    coefficients = np.linalg.solve(
        centered.T @ centered + lam * np.eye(width),
        centered.T @ centered_values,
    )
    intercept = y_mean - float(x_mean @ coefficients)
    residuals = X @ coefficients + intercept - y
    train_mse = float(np.mean(residuals ** 2))
    return RidgeSiteFit(
        coefficients=coefficients, intercept=intercept, train_mse=train_mse
    )


class GradientDescentTrainer:
    """Full-batch gradient-descent fit of a linear model on standardized features.

    ``fit`` standardizes the feature columns internally (per-column zero
    mean, unit standard deviation; constant columns pass through with
    standard deviation 1), then runs at most ``max_iters`` full-batch
    gradient updates on the mean-squared-error loss.  The MSE at the top
    of each executed iteration is appended to ``loss_history``; training
    stops early with ``converged_ = True`` as soon as two consecutive
    losses differ by less than ``tol``.  ``coef_`` is expressed against
    the standardized features, and ``predict`` re-applies the stored
    standardization before the linear map.

    Attributes
    - lr: Learning rate (strictly positive).
    - max_iters: Update-iteration budget (at least 1).
    - tol: Convergence threshold on consecutive per-iteration MSE.
    - coef_: Standardized-space coefficient vector (set by ``fit``).
    - intercept_: Bias term in original target units (set by ``fit``).
    - loss_history: Per-iteration MSE, one float per executed iteration.
    - converged_: Whether the early-stopping tolerance was reached.
    - mean_: Per-column feature means used for standardization.
    - std_: Per-column feature standard deviations (constant columns
      pass through with standard deviation 1).
    """

    def __init__(
        self, lr: float = 0.05, max_iters: int = 300, tol: float = 1e-9
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be strictly positive, got {lr}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {max_iters}")
        self.lr = float(lr)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.loss_history: List[float] = []
        self.converged_: bool = False
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(
        self, features: np.ndarray, target: np.ndarray
    ) -> "GradientDescentTrainer":
        """Fit the linear model by full-batch gradient descent.

        Parameters
        - features: Design matrix of shape ``(n_samples, n_features)``.
        - target: Target vector of shape ``(n_samples,)``.

        Returns
        - self, with ``coef_``, ``intercept_``, ``loss_history``,
          ``converged_``, ``mean_``, and ``std_`` populated.

        Raises
        - ValueError: If ``features`` is not 2-D, ``target`` is not 1-D,
          their sample counts disagree, or no sample rows are given.
        """
        X = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"features must be 2-D (n_samples, n_features), got ndim={X.ndim}"
            )
        if y.ndim != 1:
            raise ValueError(f"target must be 1-D (n_samples,), got ndim={y.ndim}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"features has {X.shape[0]} rows but target has {y.shape[0]} entries"
            )
        if X.shape[0] == 0:
            raise ValueError("fit needs at least one sample row")
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        standardized = (X - mean) / std
        n_samples, n_features = X.shape
        coef = np.zeros(n_features, dtype=float)
        intercept = 0.0
        loss_history: List[float] = []
        converged = False
        previous: Optional[float] = None
        for _ in range(self.max_iters):
            errors = standardized @ coef + intercept - y
            loss = float(np.mean(errors ** 2))
            loss_history.append(loss)
            if previous is not None and abs(loss - previous) < self.tol:
                converged = True
                break
            coef -= self.lr * (2.0 / n_samples) * (standardized.T @ errors)
            intercept -= self.lr * (2.0 / n_samples) * float(np.sum(errors))
            previous = loss
        self.mean_ = mean
        self.std_ = std
        self.coef_ = coef
        self.intercept_ = intercept
        self.loss_history = loss_history
        self.converged_ = converged
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict target values with the fitted linear model.

        Parameters
        - features: Design matrix of shape ``(n_samples, n_features)``;
          the second dimension must match the fitted feature count.

        Returns
        - Predicted values of shape ``(n_samples,)``.

        Raises
        - ValueError: If ``fit`` has not been called yet, or ``features``
          is not 2-D with the fitted number of feature columns.
        """
        if self.coef_ is None or self.mean_ is None or self.std_ is None:
            raise ValueError("predict requires a prior call to fit")
        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"features must be 2-D (n_samples, n_features), got ndim={X.ndim}"
            )
        n_features = self.coef_.shape[0]
        if X.shape[1] != n_features:
            raise ValueError(
                f"features must have {n_features} feature columns, got {X.shape[1]}"
            )
        standardized = (X - self.mean_) / self.std_
        return standardized @ self.coef_ + self.intercept_
