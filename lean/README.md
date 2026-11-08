# lean/ — exact integer quadray foundations

Core-only Lean 4 sources, with no Mathlib or external package dependency.
Package: `Quadlean`; toolchain: `leanprover/lean4:v4.35.0`.
These are handwritten mathematical models, not extraction from, or verification
of, the Python, JavaScript or C++ implementations or floating simulations.

## Build

From this directory:

```bash
lake build
```

This elaborates the four modules imported by `Quadlean.lean`. Existing `#eval`
commands print examples during compilation; printed output is not a theorem.
The foundation additions were written without a worker-side build; integration
must elaborate them before describing the revised sources as machine-verified.

## Universal theorem source

### `QuadMath/Quadray.lean`

* `SameClass` is common-offset equivalence on integer four-tuples. Its
  reflexivity, symmetry and transitivity supply `quadraySetoid` and the actual
  quotient `PhysicalQuadray`; `physicalClass_eq_iff` characterizes equality there.
* `normalize_spec`, `quadMin_normalize` and `normalize_idem` establish the
  nonnegative, minimum-zero canonical representative.
  `normalize_eq_iff_sameClass` and `canonical_unique` establish uniqueness,
  not just existence/idempotence.
* `sameClass_iff_proj_eq` identifies classes exactly by the three integer
  coordinates `(a-d,b-d,c-d)`. `proj_surjective` supplies every integer triple.
* `tetraDet_sameClass` and `tetraVolume_sameClass` allow **four independent
  vertex gauges**, including the base vertex. The older `tetraDet_congr`
  is the narrower simultaneous common-shift result.
* Exact native volume is `mkRat |tetraDet| 4`, not universally an integer.
  Closed kernel-decided examples establish the unit volume 1 and primitive
  volume 1/4. No general Bareiss implementation-correctness theorem is claimed.

### `QuadMath/Distance.lean`

* `deltaDist2_eq`: for every integer delta, the scale-one embedding has
  `d² = 4 Σδᵢ² − (Σδᵢ)²`. The correction vanishes for equal-sum endpoints,
  not merely because two labels are in the same physical class.
* `dist2_nonneg`, `dist2_comm`, `dist2_congr` and `dist2_eq_zero_iff` give a
  nonnegative, symmetric squared distance on physical classes. Zero distance
  means `SameClass`, equivalently equal canonical representatives
  (`dist2_eq_zero_iff_normalize_eq`), not equal raw game-cell labels.
  `xyzInt_eq_iff_sameClass` also characterizes equality of physical images.
  This is **not** a Lean `MetricSpace`, nor an ordinary triangle inequality
  for squared distance, nor a square-root metric formalization.
* `xyzTetraDet_eq` proves the signed scale-one XYZ determinant is
  `4 * tetraDet`. `tetraVolume_xyz_bridge` proves native volume equals
  `(3/8) * xyzTetraVolume`; the unit tetrahedron has XYZ volume `8/3`.
  The row order is exactly `DEFAULT_EMBEDDING`. QuadCraft's separate
  `(X,Z,Y)/sqrt(2)` convention reverses orientation and rescales volume;
  that irrational rescaling is **not** formalized here. Do not apply its S3
  conversion factor to the scale-one XYZ volumes in this module.
* `shellNorm4_le_dist2` proves the universal radial lower bound for every
  integer class with component sum divisible by four.
  `shellSites_dist2_lower` gives `8*k ≤ d²(0,q)` for `q ∈ shellSites k`;
  `shellSites_outside_cutoff` gives strict exclusion beyond a squared-radius
  cutoff. The proof uses integer `|z| ≤ z²` in the sum-zero gauge, not
  finite checking. It applies to the centered-norm enumeration. No theorem
  here identifies that enumeration with a BFS implementation.
* `dist2_neighborMove` establishes squared radius 8 for each of the twelve
  declared moves. Other shells need not have constant Euclidean radius.

### `QuadMath/IVM.lean`

* `IsIVMSite q` requires **canonical** coordinates. The class-level predicate
  is `IsIVMSite q.normalize`, characterized by `isIVMSite_normalize` and
  invariant under `SameClass` by `isIVMSite_normalize_congr`.
* `ivmCentered` is a different gauge: it has sum zero when the component sum
  is divisible by four (`ivmCentered_sum_zero`), preserves physical class
  and shell norm, and need not have nonnegative components.
* `shellNorm4_component_bound` proves the scan box `[0,2k]^4` is complete;
  `mem_shellSites_iff` characterizes membership exactly by canonical site
  status and `shellNorm4 q = 8*k`. `shellSites_unique` gives a unique index.
* Neighbor distinctness/site membership and the explicitly indexed finite
  cardinality proofs have their stated finite scope.

### `QuadMath/Lattice.lean`

`omniNumber k = 10*k*k+2` and `cumulativeCount` are formula definitions.
Theorems connect enumeration to that formula on shells 1–3 only. Defining
a formula is not a universal proof that it counts sites.

## Finite computation and trust

* `decide` proofs, including `shellSites_card_target_1` through `_4`, produce
  proof terms checked by kernel reduction. They are computational proofs,
  not computations with “no proof term”.
* The existing `_5` through `_8` declarations use `native_decide`. They also
  have theorem terms, but the pinned Lean 4.35 implementation trusts compiled
  evaluation through per-instance axioms. The mechanism is not
  `Lean.ofReduceBool`. Those dependencies propagate to theorem users of these
  declarations, not to every unrelated theorem importing `IVM`.
* No new axioms, `sorry`, or native-decision foundation proofs are introduced.
  Zero `sorry` alone is not an axiom audit. Use `#print axioms theoremName`
  when auditing a particular declaration; no fresh axiom audit was performed
  during this source-only implementation.
* `#eval` shell counts and membership examples are output-producing commands,
  not asserted proofs. Neither the finite instances nor printed agreement
  establish the universal positive-shell count.

`shellSites_card_target k` is only a proposition-valued definition of one
indexed equality. There is no theorem proving it for arbitrary `k ≥ 1`, and
no universal cumulative ball-cardinality theorem. There is also no Lean walk,
reachability, shortest-path/BFS-shell equivalence, graph-field dynamics,
harmonic learning, relativistic geometry or floating-arithmetic theorem.
The constructive reachability explanation in the manuscript remains outside
this formalization.
