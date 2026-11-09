# IVM Learning Operations: Fields, Transport and Held-Out Evaluation

This page maps runtime methods to their canonical chapters. It does not define
an independent geometry or promise out-of-sample skill from a successful fit.
Start with the [methods](../../quadmath/markdown/03_quadray_methods.md), then
[field learning](../../quadmath/markdown/11_ivm_field_learning.md),
[dynamics](../../quadmath/markdown/12_ivm_dynamics.md),
[evaluation](../../quadmath/markdown/15_learning_evaluation.md) and
[lattice gallery](../../quadmath/markdown/16_lattice_gallery.md).

## Field fitting and hard-boundary extension

[`ivm_field.py`](../../src/quadmath/lattice/ivm_field.py) supplies
`IVMField.lattice_ball`, `learn`, `harmonic_extension`, `predict` and `score`.
`learn` solves soft Laplacian-regularized kernel-weighted least squares:
at positive regularization observed targets can move; at zero regularization
the kernel targets are evaluated directly and observed floats retained exactly.
Relative geodesic Gaussian mixtures keep narrow-kernel targets distinct from
the absolute confidence floor. The dense observation-response solve requires
an observation in every component and rejects inaccurate results atomically;
it is not a universal forward-accuracy guarantee.
`harmonic_extension` instead fixes supplied values exactly and solves the
interior graph Laplacian system; each component must have a boundary anchor.
Boundary sites can be any anchored subset, not just the outer shell.
Both reject duplicate gauge-equivalent sites, nonfinite values and unanchored
components. See the canonical chapter for their distinct numerical contracts.

`quadray_shell_norm` is the centered L1 norm, with `shell_sites(k)` targeting
norm `2k`; `shell_ball_sites(radius)` uses **shell units**. The positive-shell
formula excludes the singleton origin. `fit_geometry` estimates an unconstrained
3-by-3 linear map of labeled tetrahedral vertex centroids. It does not fit an
independent translation or constrain the map to a pure rotation plus uniform
scale; its reported scale is the signed cube root of the determinant.

## Dynamics: three distinct update laws

[`ivm_dynamics.py`](../../src/quadmath/lattice/ivm_dynamics.py) constructs
`IVMLattice` with `make_lattice(radius)`, whose ball uses **scale-one Euclidean
radius**, not the field learner's shell radius. The twelve canonical neighbor
moves have squared length 8; a Euclidean radius-3 ball has 13 sites.

| Method / `DynamicsParams.kind` | Meaning and limits |
|---|---|
| `heat_step` / `"heat"` | Symmetric-normalized adjacency update; squared-norm contraction for coupling in `[0,1]`, not general total-mass or constant preservation on irregular boundary degrees |
| `conservative_heat_step` / `"conservative"` | Unnormalized-Laplacian concentration transport scaled by maximum degree; mass/constant and range preservation with norm/Dirichlet-energy contraction in the finite-graph model, subject to floating arithmetic |
| `majority_step` / `"majority"` | Integer rounded neighbor averaging, exact round-half-to-even; range preserving, **not** generally squared-norm monotone |

`graph_laplacian`, `dirichlet_energy` and `sum_of_squares` provide observables.
`simulate` preserves the selected update law; `fit_trajectory` performs a
candidate-grid/refinement search for coupling, not a general optimizer or
identifiability theorem. Integer majority trajectories can admit coupling ties.
Neither normalized heat nor conservative transport is a proved continuum solver.

## Evaluation: selection is not an independent test

[`learning_eval.py`](../../src/quadmath/learn/learning_eval.py) provides:

- `kfold_site_splits`: seeded random partitions of distinct normalized observed
  sites. These are **not spatially blocked** folds; autocorrelation remains.
- `cross_validate_field`: select `lam` using mean validation-fold MSE, then refit
  on all observations. `best_mean_mse` is a selection score; `refit_mse` is
  in-sample. Neither is an untouched final-test estimate.
- `learning_curve`: seeded nested training subsets, scored on each complement;
  the held-out composition changes with training fraction.
- `trajectory_train_test`: fit only the chronological prefix; report held-out
  multi-step continuation and teacher-forced one-step errors separately.
  `kind` must match the intended update law, including `"conservative"` when
  modeling concentration transport. Do not shuffle snapshots.
- `three_way_split`, `ridge_site_fit`, `GradientDescentTrainer`: tabular index
  splits, a floating closed-form ridge baseline and standardized full-batch
  gradient descent, not exact rational fitting or an ML framework.

Choose an independent spatial/temporal evaluation design suited to the scientific
question. A seeded split is reproducible, not automatically statistically independent.

## Small numerical consumer recipe

After the [Python 3.13 setup](../../README.md#quick-start), run with
`uv run --python 3.13 python`:

```python
import numpy as np
from quadmath.lattice.ivm_field import IVMField
from quadmath.lattice.ivm_dynamics import conservative_heat_step, make_lattice
from quadmath.learn.learning_eval import cross_validate_field

field = IVMField.lattice_ball(1)
field.harmonic_extension([field.sites[0]], [2.0])
print("constant extension range:", field.values.min(), field.values.max())

lattice = make_lattice(3)
u = np.arange(lattice.size, dtype=float)
v = conservative_heat_step(u, lattice, 0.3)
print("transport mass:", u.sum(), v.sum())

sites = field.sites[:6]
values = [float(i) for i in range(len(sites))]
selection = cross_validate_field(values, sites, k=3,
                                 lam_grid=[0.01, 0.1], seed=7, radius=1)
print("selected lambda:", selection.best_lam)
print("validation / in-sample refit:", selection.best_mean_mse, selection.refit_mse)
```

This is a tiny synthetic consumer example, not evidence of real-data skill.
For a figure without tracked-output writes:

```bash
QUADMATH_OUTPUT_DIR="$(mktemp -d)/quadmath-results" MPLBACKEND=Agg \
  uv run --python 3.13 python quadmath/scripts/ivm_field_demo.py
```

Python renderers honor the output helper policy; the shell PDF pipeline does not
share that override. See [workflow](../../WORKFLOW.md) and
[development checks](../development/README.md). Return to the
[documentation index](../README.md).
