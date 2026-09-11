# lean/ — Lean 4 formalization of the QuadMath core

A Lean 4 (core only, **no Mathlib** dependency) formalization of the
quadray-coordinate foundations used by QuadMath. Package name: `Quadlean`.

## Build and run

Requires [elan](https://elan.lean-lang.org) (the toolchain is pinned in
`lean-toolchain`, currently `leanprover/lean4:v4.33.1`; `elan show` resolves
the same default). No other dependencies:

```bash
cd lean
lake build        # builds QuadMath.Quadray, QuadMath.IVM, QuadMath.Distance, QuadMath.Lattice
```

`lake build` exits 0 on success. The `#eval` smoke checks print their
results during the build (shell cardinalities, the `10k^2 + 2` sequence,
membership examples). To interact with the code, use `lake env lean` with a
scratch file that imports `QuadMath.IVM`, or open `lean/` in an LSP-enabled
editor.

## Modules

| Module | Formalizes | Python correspondence |
|---|---|---|
| `QuadMath.Quadray` | Quadray coordinates as integer 4-tuples: projective normalization by subtracting the minimum component, and the exact integer tetra-volume via the `(a-d, b-d, c-d)` projected 3x3 determinant identity (`|det| / 4`; unit IVM tetra `det = 4`, volume `1`; primitive tetra `det = 1`, volume `1/4`). Proves normalization is idempotent, lands in the class (`q ~ q + t(1,1,1,1)`), and that the determinant is class-invariant. | `src/quadray.py` — `Quadray` dataclass, `normalize` (`src/quadray.py:29-36`), `integer_tetra_volume` (`src/quadray.py:60-82`); determinant arithmetic mirrors `bareiss_determinant_int` in `src/linalg_utils.py:6-56`. |
| `QuadMath.IVM` | The IVM (FCC/CCP) lattice in quadray coordinates: the 12 canonical neighbor moves (permutations of `(2,1,1,0)`), a computable membership predicate `IsIVMSite` (normalized ∧ component sum divisible by 4 — the `(1,0,0,0)`-type tetrahedral-void directions are excluded), the L1-centered shell norm `shellNorm4` (4x-scaled `quadray_shell_norm`; shell `k` ⟺ `shellNorm4 = 8k`), and a computable shell enumeration `shellSites` over the `[0, 2k]^4` scan box. Proves the scan-box completeness bound (`shellNorm4_component_bound`), the exact enumeration characterization (`mem_shellSites_iff`), that membership and shell norm are projective-class properties (`isIVMSite_normalize`, `shellNorm4_sameClass`), that shell membership determines the shell index (`shellSites_unique`), and machine-checks the shell cardinalities `10k² + 2` through frequency 8 (`shellSites_card_target_*`). | `src/ivm_field.py` — `IVM_NEIGHBOR_STEPS` (`:61-74`), `is_ivm_site` (`:81-89`), `quadray_shell_norm` (`:92-118`), `shell_sites` (`:121-157`); `neighbor_moves_ivm` in `src/discrete_variational.py:10-14`. |
| `QuadMath.Distance` | The exact squared Euclidean distance under `DEFAULT_EMBEDDING`: proves, universally over integer deltas, `d² = 4·(δa² + δb² + δc² + δd²) − (δa + δb + δc + δd)²` (`deltaDist2_eq`), plus symmetry, zero self-distance, and the twelve-around-one instance (every neighbor move at squared distance `8` from the origin). | `src/quadray.py` — `DEFAULT_EMBEDDING` (`:108-113`), `distance` (`:156-176`); the exact `d² = 4Σδ² − (Σδ)²` identity used by `src/lattice_search.py`. |
| `QuadMath.Lattice` | The omnidirectional close-packing numbering: `omniNumber k = 10k^2 + 2` (shell `k ≥ 1`) and cumulative counts, cross-checked against the concrete `shellSites` enumeration — kernel-checked realization on shells 1-3 (`shellSites_omni_*`; shell 3 rides on the kernel-decided `shellSites_card_target_3`), `#eval` agreement through shell 8. | `src/omni_numbering.py` (frequency shells, `10k^2 + 2`). |

`src/geometry.py` (Minkowski interval, Lorentz factor, proper time — the
Einstein.4D namespace) has **no counterpart in this formalization**: it is
relativistic geometry, not quadray/lattice math, and is cited here only to
state that absence.

## Conventions

* A quadray is an integer 4-tuple `(a, b, c, d)`; `Quadray.normalize`
  subtracts `k = min(a, b, c, d)`, selecting the canonical representative of
  the projective class `q ~ q + t(1,1,1,1)`.
* An IVM lattice site is a normalized quadray whose component sum is
  divisible by 4 (`src/ivm_field.py:81-89`). Membership is decidable;
  normalization shifts the sum by a multiple of 4, so it is a class property.
* Shell `k` collects sites with L1-centered shell norm `N(q) = 2k`
  (`src/ivm_field.py:92-118`), encoded here integrally as
  `shellNorm4 q = 8k`, enumerated over the box `[0, 2k]^4` exactly as
  `shell_sites` does.

## What is proved vs. machine-checked

Proved in Lean (kernel-checked, core only — no Mathlib):

* Normalization: non-negative components, minimum zero, idempotence, and
  class preservation (`normalize_spec`, `normalize_idem`, `normalize_sameClass`).
* The determinant identity: unit IVM tetrahedron `det = 4`, volume `= 1`;
  primitive tetrahedron `det = 1`, volume `= 1/4` (`tetraDet_unit`,
  `tetraVolume_unit`, `tetraDet_primitive`, `tetraVolume_primitive`); the
  determinant is invariant under shifting all four vertices by the same
  `t(1,1,1,1)` (`tetraDet_congr`).
* The 12 neighbor moves are pairwise distinct and are all IVM sites.
* **Scan-box completeness** (`shellNorm4_component_bound`): every IVM site
  with shell norm `8k` has all four components in `[0, 2k]`, so the
  `[0, 2k]^4` scan box of `shell_sites` loses no shell sites — the
  bounding-box claim of `src/ivm_field.py:124-125`, proved here rather than
  taken from the Python reference.
* **Enumeration characterization** (`mem_shellSites_iff`): `shellSites k`
  contains exactly the IVM sites whose shell norm is `8k` — the
  frequency-`k` shell and nothing else.
* Membership and shell norm are projective-class properties
  (`isIVMSite_normalize`, `shellNorm4_sameClass`), and shell membership
  determines the shell index (`shellSites_unique`).
* **Exact squared-distance identity** (`deltaDist2_eq`, universal over
  integer deltas): under `DEFAULT_EMBEDDING`,
  `d² = 4·(δa² + δb² + δc² + δd²) − (δa + δb + δc + δd)²`; symmetry
  (`dist2_comm`), zero self-distance (`dist2_self`), and the
  twelve-around-one instance — every neighbor move at squared distance `8`
  from the origin (`neighborMoves_dist2`, `dist2_neighborMove`).
* Shell cardinalities for `k = 0, 1, 2`: `1`, `12`, `42`
  (`shellSites_*_length`), and `shellSites k` realizes `omniNumber k` on
  shells 1, 2, 3 (`shellSites_omni_*`; shell 3 via the machine-checked
  instance).

Machine-checked (computed, no proof term):

* `shellSites_card_target k` — `(shellSites k).length = 10k² + 2` — for
  `k = 1..4` by kernel-evaluated `decide` (an axiom audit with
  `#print axioms` shows these instances depend on **no** axioms) and for
  `k = 5..8` by `native_decide`: the virtual machine evaluates the same
  closed enumeration and the kernel trusts the result through one
  per-instance `native_decide` axiom (kernel evaluation of the `[0, 2k]^4`
  filter is the build-time bottleneck, so larger shells use VM evaluation).

`#eval` smoke checks (printed at build time): membership examples (neighbor
class true, void direction false), the shell lengths
`[12, 42, 92, 162, 252, 362, 492, 642]` agreeing with `10k² + 2` through
shell 8, the `omniNumber` sequence, cumulative counts `[1, 13, 55, 147, 309,
561]`, and `shellSites (k+1)` agreeing with `omniNumber (k+1)` for `k ≤ 8`.

Stated but **not** proved: the universal shell-count theorem itself —
`shellSites_card_target k` for arbitrary `k ≥ 1`. It is kept as a
proposition-valued definition (`def shellSites_card_target`), so the tree
ends with zero unproved declarations while the general claim remains open in
this formalization. The Python reference machine-checks the corresponding claim
through frequency 6 (`src/ivm_field.py:97-98`, `src/omni_numbering.py:24-26`).
Proof sketch for the general case, for future work: via the sum-zero
representative `w = q − (s/4)·(1,1,1,1)` of each site, shell `k` is in
bijection with the integer 4-vectors `w` satisfying `Σw = 0` and
`Σ|w| = 2k`; counting those by sign pattern (positive positions `p` and
negative positions `n` with `p, n ≤ 3`, compositions of `k` into `p`
positive and `n` negative parts) gives
`12 + 24(k−1) + 8·(k−1)(k−2)/2 + 6(k−1)² = 10k² + 2`; formalizing the
stars-and-bars composition count in core Lean is the missing step.

## Constraints honored

* No Mathlib (or any other) dependency — `Int`, `Nat`, `List`, decidable
  propositions, `mkRat`, `#eval`.
* Nothing outside `lean/` was created or modified.