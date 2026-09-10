# learning/ — IVM field and dynamics modules

Documentation for the machine-learning modules over the IVM lattice:
`src/ivm_field.py` (static field learning) and `src/ivm_dynamics.py` (field
dynamics + coupling identification). Both modules exist on disk;
`tests/test_ivm_dynamics.py` is landed (40 tests, 100% branch coverage), while
`tests/test_ivm_field.py` had not landed at last check — the corresponding
manuscript section is `quadmath/markdown/12_ivm_dynamics.md`
(ported to [docs/manuscript/12_ivm_dynamics.md](../manuscript/12_ivm_dynamics.md)).

Design lineage: quadray coordinates (`src/quadray.py`) → IVM lattice geometry
→ field evolution → learning/identification, all numpy-only and
deterministic (no ML frameworks; seeded `numpy.random.default_rng`).

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

## Reading order

1. [docs/manuscript/12_ivm_dynamics.md](../manuscript/12_ivm_dynamics.md) —
   the analytical treatment (lemma, proposition, honest non-claims)
2. This README — API map
3. `tests/test_ivm_dynamics.py` — the contracts as executable assertions
   (100% coverage, fixed seeds, no mocks — see
   [development](../development/README.md)); the `ivm_field` test file is
   owned by its implementing agent and pending at last check.
