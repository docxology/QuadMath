# Learning and Evaluation on the IVM Lattice

## Prediction tasks, splitting, and model selection

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
`tests/unit/learn/test_learning_eval.py` with fixed RNG seeds and real numerics.

## Why held-out evaluation is harder on a lattice

Spatially correlated data require an explicit prediction task. Random
site holdout estimates interpolation from the remaining observations; nearby
training values are legitimate inputs for that task, not necessarily leakage.
It does not estimate transfer to a distant unobserved region. Spatial-block
or buffer-separated holdouts would be needed for that stronger question;
`kfold_site_splits` does not construct them. The following covariance expression
is an illustrative smooth-field assumption, not a covariance model fitted by
these utilities:

\begin{equation}
\label{eq:learn-correlation}
\operatorname{Cov}\bigl(f_i, f_j\bigr) \;\approx\; \sigma_f^{2}\,
K\bigl(d(i,j)\bigr),
\qquad d(i,j) \;=\; \text{hop distance on the IVM graph},
\end{equation}

Here $K$ is the Gaussian hop kernel and $\sigma_f^2$ a hypothetical field
variance. The learner does not infer this covariance or independence from it.

The implemented field split randomly assigns **distinct normalized sites**,
not spatial blocks. Dynamics instead holds out a temporal suffix. The
synthetic examples demonstrate sensitivity to observed-site coverage;
averaging folds estimates performance for that sampling design, not all
deployment geometries.

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
regularization wins on the held-out folds. Observed targets have unit
confidence but are **soft** constraints: positive $\lambda$ can move fitted
values at observed sites too. The gain depends on kernel targets, geometry,
noise, and the chosen regularizer; no universal preferred $\lambda$ is implied.

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

## Three-way splitting and trainable site fits {#sec:learn_three_way_site_fits}

The lattice surfaces above all share one shape: a seeded split of sites or
snapshots, a fit on the training side, and scores on the held-out side. The
same shape is available for plain tabular work in the same module
(`src/quadmath/learn/learning_eval.py`), under the same discipline — all
split randomness confined to the seeded permutation of the split, and fits
that are pure deterministic functions of their inputs.

`three_way_split(n, train_frac=0.6, val_frac=0.2, seed=0)` partitions
`range(n)` into disjoint train/validation/test index lists. One seeded
`numpy.random.default_rng` permutation of `range(n)` is drawn and cut into
three contiguous blocks: the first $m_{train} =
\operatorname{round}(\mathrm{train\_frac} \cdot n)$ indices train, the next
$m_{val} = \operatorname{round}(\mathrm{val\_frac} \cdot n)$ validate, and
the remainder test. Each returned list is sorted, the three are pairwise
disjoint with union exactly `range(n)`, and the split is a deterministic
function of $(n, \mathrm{train\_frac}, \mathrm{val\_frac}, seed)$ — the same
deterministic-permutation contract as the k-fold site splits above, applied
to a plain index range rather than observed lattice sites.
Rounding is unguarded at the tails: for small $n$ a trailing block may come
back empty ($n = 3$ under the default fractions leaves the test list empty),
and a `ValueError` is raised when either fraction is not strictly positive
or their sum reaches $1$ — a split with no held-out rows at all cannot be
honest.

`ridge_site_fit(features, values, lam=1e-3)` fits a single linear site model
in closed form. With $X_c$ the mean-centered design matrix, $y_c$ the
mean-centered target, and $\bar{x}$, $\bar{y}$ the feature-column means and
target mean, the ridge normal equations are solved numerically for the slope
vector $w$:

\begin{equation}
\label{eq:learn-ridge}
\bigl( X_c^{\!\top} X_c + \lambda I \bigr)\, w \;=\; X_c^{\!\top} y_c,
\qquad b \;=\; \bar{y} - \bar{x}^{\!\top} w,
\end{equation}
with $b$ the intercept recovered from the centered solve; $\lambda \geq 0$
penalizes the centered coefficient norm, shrinking $w$ toward zero, and
$\lambda=0$ gives unregularized normal equations when the centered system
is nonsingular; otherwise `numpy.linalg.solve` can raise.
The returned `RidgeSiteFit` carries `coefficients`, `intercept`, and the in-sample
`train_mse`, and the fit is a deterministic function of `(features, values,
lam)` — a `numpy.linalg.solve`, not an iterative optimizer, so repeated
calls on the same inputs reproduce it bit for bit. Validation is explicit: a
negative $\lambda$, a non-2-D design matrix, a non-1-D target, disagreeing
sample counts, or zero rows each raise before any linear algebra runs.

`GradientDescentTrainer(lr=0.05, max_iters=300, tol=1e-9)` is the iterative
counterpart, useful where the closed form is deliberately being compared
against: a full-batch gradient-descent fit of the same linear model. `fit`
standardizes the feature columns internally (per-column zero mean and unit
standard deviation, with constant columns passing through at standard
deviation $1$), then runs at most `max_iters` full-batch updates on the
mean-squared-error loss,

\begin{equation}
\label{eq:learn-gd}
\theta \;\leftarrow\; \theta \;-\; \frac{2\,\eta}{m}\,
\begin{bmatrix} X_s^{\!\top} e \\ \mathbf{1}^{\!\top} e \end{bmatrix},
\qquad e \;=\; X_s w + b - y,
\end{equation}

where $X_s$ is the standardized design, $m$ the row count, $\eta$ the
learning rate, $w$ and $b$ the coefficient vector and bias, and $y$ the
target. The MSE at the top of each executed iteration is appended to
`loss_history` — one float per iteration, recorded before the update it
describes — and training stops early with `converged_ = True` as soon as two
consecutive losses differ by less than `tol`. The fit exposes `coef_`
expressed against the standardized features and `intercept_` in original
target units, together with the stored `mean_` and `std_` it used, so
`predict` re-applies the recorded standardization before the linear map and
thus scores consistently across the train, validation, and test blocks of a
`three_way_split`. Like `ridge_site_fit` the trainer draws no randomness of
its own — no shuffling, no stochastic gradient — so its entire loss history
is deterministic for fixed inputs: the same rows and hyperparameters replay
the same loss curve to the last float, and the per-iteration record doubles
as the divergence/plateau diagnostic that a scalar final loss cannot
provide.

![**Convergence of the full-batch gradient-descent trainer.** Per-iteration
training loss (mean squared error, linear $y$-axis) of
`GradientDescentTrainer` on a deterministic synthetic linear problem: a
$40 \times 3$ standard-normal design matrix with true coefficients
$(1.5, -2.0, 0.75)$, intercept $0.5$, and Gaussian target noise
$\sigma = 0.05$ (seed $0$). Fitted with `lr = 0.05`, `max_iters = 300`,
`tol = 1e-9`; the loss falls from $6.23$ at iteration $0$ to the
noise-floor scale $\sigma^2 = 2.5\times 10^{-3}$, flagged as converged
after $126$ iterations. Regenerate with
`quadmath/scripts/learning_gallery.py`.](figures/learn_loss_history.png)

## Split, fit, and forecast regression scope

`tests/unit/learn/test_learning_eval.py` pins all of the above with fixed seeds and no
mocks: disjoint-exhaustive folds with exact fold sizes, seed-determinism and
seed-sensitivity of the splits, the $\lambda^{*}=0$ and $\lambda^{*}>0$
regimes with their table identities (mean and standard deviation recomputed
from the fold table; the refit reproduced by hand), exact-zero errors for the
correctly specified dynamics against a $> 2\times$ continuation blow-up for
the misspecified one, manual recomputation of the one-step-ahead error,
fraction clamping at both extremes, and the learning curve's final-below-first
invariant. These are test scopes, not a fresh coverage receipt. Validation
rules and numerical limits are defined by the current implementation;
the [operational workflow](../../WORKFLOW.md) describes how to obtain
dated test and coverage results.

## Cross-References

- Coordinate foundations and the twelve-around-one shell: [Quadray Methods](03_quadray_methods.md)
- The learner being evaluated: [Static IVM Field Learning](11_ivm_field_learning.md)
- The dynamics being identified: [IVM Lattice Dynamics](12_ivm_dynamics.md)
- Enumeration and search infrastructure: [Lattice Tooling](13_lattice_tooling.md)
