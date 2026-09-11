# learning/ — IVM learning surface

Documentation for the machine-learning modules over the IVM lattice — all
landed on disk (checked 2026-09-10):

- `src/ivm_field.py` — static field learning on the lattice
- `src/ivm_dynamics.py` — field dynamics + coupling identification
- `src/learning_eval.py` — training/testing methodology (honest evaluation)
- `src/vis_lattice.py` — the matplotlib rendering primitives behind the
  visualization gallery (see section `16_lattice_gallery.md`)

Design lineage: quadray coordinates (`src/quadray.py`) → IVM lattice geometry
→ field evolution → learning/identification → evaluation, all numpy-only and
deterministic (no ML frameworks; seeded `numpy.random.default_rng`).

Manuscript treatments (ported copies live under
[manuscript/](../manuscript/)): [11_ivm_field_learning.md](../manuscript/11_ivm_field_learning.md),
[12_ivm_dynamics.md](../manuscript/12_ivm_dynamics.md),
[15_learning_evaluation.md](../manuscript/15_learning_evaluation.md), and
[16_lattice_gallery.md](../manuscript/16_lattice_gallery.md).

## `src/ivm_field.py` — static field learning on the lattice

Public API (`__all__`): `IVM_NEIGHBOR_STEPS`, `IVMField`, `TetrahedronFit`,
`ball_sites`, `fit_geometry`, `is_ivm_site`, `quadray_shell_norm`,
`shell_sites`, `shell_cardinalities`.

- **Geometry**: a lattice site is a normalized quadray (non-negative, one
  zero) whose component sum is divisible by 4 — exactly the close-packed
  sphere centers. `quadray_shell_norm` is the centered L1 shell norm;
  shell `k` has cardinality `10k² + 2` (cuboctahedral numbers 1, 12, 42, …).
  `shell_sites` / `ball_sites` enumerate exactly (integers only).
- **`IVMField`**: scalar field over a lattice ball; sites ordered by
  (shell, lexicographic quadray) via `ball_sites`. Methods: `learn`
  (Laplacian-regularized kernel-weighted least squares on the lattice
  graph), `predict(site)`, `score(sites, values)`.
- **`fit_geometry`**: least-squares recovery of best-fit tetrahedron
  orientation+scale from noisy 3D points through the quadray basis
  (`to_xyz`); returns a `TetrahedronFit`.

## `src/ivm_dynamics.py` — dynamics and coupling identification

Public API: `neighbor_shifts`, `site_radius_sq`, `ball_sites`, `IVMLattice`,
`make_lattice`, `sum_of_squares`, `is_nonincreasing`, `heat_step`,
`majority_step`, `DynamicsParams`, `step`, `Trajectory`, `simulate`,
`FitResult`, `fit_trajectory`, `render_dynamics_demo`.

- **Move graph**: adjacency = embedding difference is one of the 12
  canonical IVM neighbor shifts (permutations of `(2,1,1,0)`). The R=3 ball
  (27 sites) decomposes into a 12-around-one cluster, two tetrahedra, and an
  octahedron — the update laws act independently per component.
- **Heat update** (`heat_step`): `u ← (1−α)u + α·S·u` with the symmetric
  normalization `S = D^{-1/2} A D^{-1/2}`; provably non-increasing in ‖u‖²
  (lemma in the manuscript section, asserted per step in
  `tests/test_ivm_dynamics.py::test_heat_update_l2_nonincreasing_per_step`).
  No claim is made for plain row-average diffusion.
- **Majority update** (`majority_step`): rounded neighbor averaging on
  integer fields; range-preserving (max non-increasing, min non-decreasing).
  Deliberately *no* L2 claim — rounding can increase sum of squares, and a
  test pins a demonstrating case.
- **Identification** (`fit_trajectory`): gradient-free grid + refinement
  search over the coupling α minimizing trajectory MSE against observed
  snapshots; deterministic, ties to smallest α. Demo: α_true = 0.3 recovered
  at machine precision (`render_dynamics_demo` produces
  `quadmath/output/figures/ivm_dynamics_demo.png`).

## `src/learning_eval.py` — honest evaluation methodology

Public API (`__all__`): `CrossValidationResult`, `LearningCurveResult`,
`TrajectorySplitResult`, `cross_validate_field`, `enclosing_radius`,
`kfold_site_splits`, `learning_curve`, `trajectory_train_test`.

Principle: spatial autocorrelation makes naive resampling optimistic, and
time-ordered dynamics data must never be shuffled — every surface here holds
out explicit structure. Four surfaces:

- **`kfold_site_splits`** — deterministic seeded k-fold partition of the
  observed lattice sites into disjoint, exhaustive train/test pairs.
- **`cross_validate_field`** — k-fold cross-validation of the
  Laplacian-regularized field learner (`IVMField.learn`): the regularization
  strength is chosen on held-out folds only and the final model is refit on
  all observed data (honest model selection). Returns
  `CrossValidationResult` (per-fold, pooled, and refit MSE).
- **`trajectory_train_test`** — temporal (first-fraction) train/test split
  for `ivm_dynamics.fit_trajectory`: identify on the training prefix, score
  by multi-step continuation MSE on the held-out suffix plus a
  teacher-forced one-step-ahead error (`TrajectorySplitResult`).
- **`learning_curve`** — held-out MSE as a function of the fraction of
  observed sites, the classic data-coverage curve on a lattice
  (`LearningCurveResult`).

All numpy-only and deterministic for fixed seeds; the split randomness is
confined to fold/subset assignment, never to the fits themselves.

## Reading order

1. [manuscript/11_ivm_field_learning.md](../manuscript/11_ivm_field_learning.md)
   — field learning on the lattice
2. [manuscript/12_ivm_dynamics.md](../manuscript/12_ivm_dynamics.md) — the
   analytical treatment (lemma, proposition, honest non-claims)
3. [manuscript/15_learning_evaluation.md](../manuscript/15_learning_evaluation.md)
   — the evaluation methodology this folder implements
4. This README — API map
5. `tests/test_ivm_field.py`, `tests/test_ivm_dynamics.py`,
   `tests/test_learning_eval.py` — the contracts as executable assertions
   (100% coverage, fixed seeds, no mocks — see
   [development](../development/README.md))
