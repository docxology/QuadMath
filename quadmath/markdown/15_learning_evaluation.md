# Learning and Evaluation on the IVM Lattice

## Overview

Learning on a lattice is easy to fake. A learner that merely interpolates its
observations through the graph can look excellent on the very data it was fit
to, and a dynamics model identified on early snapshots can silently diverge on
late ones. This section develops the training/testing methodology used to
evaluate the two IVM learners honestly — the static field learner of
[Static IVM Field Learning](11_ivm_field_learning.md) and the dynamics
identifier of [IVM Lattice Dynamics](12_ivm_dynamics.md). All utilities live
in the `learning_eval.py` module (`kfold_site_splits()`,
`cross_validate_field()`, `trajectory_train_test()`, `learning_curve()`,
`enclosing_radius()`), use `numpy` only, and are fully deterministic for
fixed seeds. Every behavioral claim below is pinned in
`tests/test_learning_eval.py` with fixed RNG seeds and real numerics.

## Why held-out evaluation is harder on a lattice

On independent data, random resampling is benign. On a lattice it is not:
field values at neighboring sites are strongly correlated, and the learners of
`ivm_field.py` exploit exactly that correlation — the kernel-weighted target
of the normal equations \eqref{eq:ivm-normal-equations} predicts an unobserved
site as a weighted average of its graph neighbors. A random split that leaves
graph neighbors on both sides therefore lets the learner "cheat": the held-out
site is predicted from observations one hop away, and the apparent skill
measures the smoothness of the field under \eqref{eq:ivm-objective}, not
generalization to unseen territory. The autocorrelation that makes lattice
learning work is the same autocorrelation that inflates its evaluation:

\begin{equation}
\label{eq:learn-correlation}
\operatorname{Cov}\bigl(f_i, f_j\bigr) \;\approx\; \sigma_f^{2}\,
K\bigl(d(i,j)\bigr),
\qquad d(i,j) \;=\; \text{hop distance on the IVM graph},
\end{equation}

with $K$ the Gaussian hop kernel of \eqref{eq:ivm-kernel} and $\sigma_f^2$ the
field variance. Because $K$ decays with hop distance, the leakage is local —
but the IVM ball is small, and a random split leaks everywhere.

The methodology below therefore always holds out *structure*, not just rows:
whole folds of sites for the field learner (never shared between training and
scoring), and a temporal suffix of snapshots for dynamics identification
(never shuffled). The pinned numerical result worth internalizing: on the
radius-2 ball (55 sites, 80% observed), the noise-free held-out error of the
kernel-weighted interpolator is dominated not by observation noise but by
*coverage geometry* — which sites happen to be observed — so per-fold scores
vary substantially even at fixed sample size, and honest evaluation must
average over folds.

## K-fold site splits

`kfold_site_splits(observed_sites, k, seed)` partitions observed sites into
$k$ folds by a seeded `numpy.random.default_rng` permutation cut with
`numpy.array_split` (the first $n \bmod k$ folds receive one extra site).
Each split is a (train, test) pair; folds are pairwise disjoint and jointly
exhaustive, and site order within each list follows the original normalized
index order, so the splits are a pure function of $(sites, k, seed)$. With
$s$ the seeded permutation and $F_j$ fold $j$'s index set, the held-out score
of a fit $\hat{f}^{(-F_j)}_\lambda$ that never saw $F_j$ is

\begin{equation}
\label{eq:learn-fold}
\widehat{M}(\lambda, F_j) \;=\; \frac{1}{|F_j|}
\sum_{i \in F_j}
\bigl( \hat{f}^{(-F_j)}_\lambda(i) - y_i \bigr)^{2},
\end{equation}

the quantity `IVMField.score()` returns. Duplicate sites are rejected after
normalization (two projective representatives of one lattice point would sit
on both sides of a split), as is $k < 2$ or $k > n$.

## Cross-validation and honest model selection

`cross_validate_field(field_values, sites, k, lam_grid, seed)` runs
\eqref{eq:learn-fold} for every fold and every regularization candidate of
`lam_grid`, fitting each fold from scratch on the training sites only — a
fresh `IVMField.lattice_ball` per fit, via
`IVMField.learn()` with the objective \eqref{eq:ivm-objective}. The result is
the full table $\widehat{M}(\lambda, F_j)$ with per-candidate mean and
standard deviation over folds, and the selected strength

\begin{equation}
\label{eq:learn-lambda-star}
\lambda^{*} \;=\; \arg\min_{\lambda \in \mathcal{G}}
\frac{1}{k} \sum_{j=1}^{k} \widehat{M}(\lambda, F_j),
\end{equation}

resolved to the earliest candidate on ties. Selection uses validation folds
only; the returned `refit_field` is then refit with $\lambda^{*}$ on *all*
observed sites, matching the deployment condition where every observation is
available. Separating the two is what makes the selection honest: choosing
$\lambda$ on the same data the final model fits is how interpolation gets
mistaken for skill.

Two regimes are pinned numerically on synthetic fields. With exact
observations of a harmonic (linear in embedded XYZ) field at 80% coverage,
cross-validation selects $\lambda^{*} = 0$: the kernel-weighted
interpolator generalizes best, and Laplacian smoothing only biases the fit
away from exact data — the mean held-out MSE is strictly increasing in
$\lambda$ across a grid spanning two decades. With noisy observations of a
smooth non-harmonic field, the selection flips to $\lambda^{*} > 0$:
interpolation carries observation noise into the fit, and moderate
regularization wins on the held-out folds. The margin is honest about the
learner's limits — because observed rows are pinned to their data, the noise
propagates through the graph regardless of $\lambda$, so the achievable gain
is set by how much neighbor averaging the kernel already performs, not by the
regularizer alone.

## Temporal splits for dynamics identification

`trajectory_train_test(observed, split_frac, lattice)` evaluates
`fit_trajectory()`-style identification of the coupling $\alpha$
([IVM Lattice Dynamics](12_ivm_dynamics.md)). The snapshot matrix
($T+1$ rows, row 0 the initial condition) is split **temporally**: the first
nearest-integer-rounded fraction of rows trains, the trailing rows are held out, and
no shuffling is ever applied — a shuffled split interpolates between adjacent
snapshots of a deterministic trajectory and hides exactly the forecast error
being measured. The split point is clamped so training keeps at least two
rows (initial condition plus one snapshot) and testing keeps at least one.
Identification uses the training prefix only, with candidate grid and
refinement passed through to `fit_trajectory()`. Two held-out errors are
reported. The multi-step continuation error re-simulates from the observed
initial condition with the identified $\hat{\alpha}$ and scores only the
held-out rows:

\begin{equation}
\label{eq:learn-horizon}
M_{\mathrm{test}}(\hat{\alpha}) \;=\; \frac{1}{(T - t_s)\,N}
\sum_{t=t_s+1}^{T} \big\lVert u_{\hat{\alpha}}(t) - u_{\mathrm{obs}}(t)
\big\rVert^{2} ,
\end{equation}

where $t_s$ is the last training row. This is the stringent metric: for any
model that is not exactly right, error compounds with horizon, so the
held-out tail punishes misspecification that early rows forgive. The
one-step-ahead error is teacher-forced — each held-out transition is
predicted a single step from the *true* previous state, the first transition
starting from the last training row:

\begin{equation}
\label{eq:learn-onestep}
M_{1}(\hat{\alpha}) \;=\; \frac{1}{(T - t_s)\,N}
\sum_{t=t_s}^{T-1} \big\lVert
\mathrm{step}\bigl(u_{\mathrm{obs}}(t);\, \hat{\alpha}\bigr)
- u_{\mathrm{obs}}(t+1) \big\rVert^{2},
\end{equation}

isolating local prediction error from error accumulation. The contrast
between \eqref{eq:learn-horizon} and \eqref{eq:learn-onestep} is diagnostic:
a correct model drives both to zero (pinned exactly: a heat trajectory with
$\alpha = 0.3$ in the candidate grid yields train, continuation, and
one-step errors of exactly $0$), while a misspecified model class — a heat
model identified on majority-dynamics data — shows a modest training error,
a held-out continuation error more than twice as large (horizon compounding),
and a one-step error five times smaller than the continuation error (local
prediction is easier than long-horizon forecasting). Note the direction of
the misspecification: `majority_step()` requires integer states, so the
float-continuum heat model can be fit to integer data, but not the reverse.

## Learning curves

`learning_curve(values, sites, train_fracs, seed)` traces the classic
data-coverage curve on a lattice. One seeded permutation of the observed
sites is drawn; for each requested fraction $p$ the first
$m(p) = \operatorname{round}(p \cdot n)$ sites of that permutation (clamped
to $[1, n-1]$) form the training subset and the remainder the held-out
complement:

\begin{equation}
\label{eq:learn-curve}
M(p) \;=\; \frac{1}{n - m(p)} \sum_{i \notin P_{m(p)}}
\bigl( \hat{f}_{P_{m(p)}}(i) - y_i \bigr)^{2},
\qquad P_{m} \;=\; \text{first } m \text{ entries of the permutation},
\end{equation}

reported together with the in-sample score of the same fit. The subsets are
*nested* by construction — growing $p$ never removes an earlier observation —
so the curve is a proper coverage curve rather than a family of independent
splits. The pinned example (radius-2 ball, quadratic bowl observed with noise
$\sigma = 0.4$ everywhere, $\lambda = 0.05$) shows held-out MSE falling by
more than a factor of two from 10% to 85% coverage, with non-monotone wiggles
in between: with few observations the held-out complement is large and its
geometry varies between fractions, so the curve is monotone-ish, not
monotone — the final point sits far below the first, and that is the
asserted invariant. The curve also separates interpolation from
generalization visibly: the in-sample MSE stays below $0.15$ at every
coverage level while the held-out MSE never drops below $0.7$ — the pinned
observations are fit closely at all sizes, and the difficulty is entirely on
unseen sites.

## Verification

`tests/test_learning_eval.py` pins all of the above with fixed seeds and no
mocks: disjoint-exhaustive folds with exact fold sizes, seed-determinism and
seed-sensitivity of the splits, the $\lambda^{*}=0$ and $\lambda^{*}>0$
regimes with their table identities (mean and standard deviation recomputed
from the fold table; the refit reproduced by hand), exact-zero errors for the
correctly specified dynamics against a $> 2\times$ continuation blow-up for
the misspecified one, manual recomputation of the one-step-ahead error,
fraction clamping at both extremes, and the learning curve's final-below-first
invariant. The module carries 100% statement and branch coverage, and the
validation rules of every entry point (distinct normalized sites, $k \in
[2, n]$, fractions in the open unit interval, non-empty grids, minimum row
counts) are each exercised.

## Cross-References

- Coordinate foundations and the twelve-around-one shell: [Quadray Methods](03_quadray_methods.md)
- The learner being evaluated: [Static IVM Field Learning](11_ivm_field_learning.md)
- The dynamics being identified: [IVM Lattice Dynamics](12_ivm_dynamics.md)
- Enumeration and search infrastructure: [Lattice Tooling](13_lattice_tooling.md)
