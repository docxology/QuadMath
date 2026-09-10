# lean/ — Lean 4 formalization of the QuadMath core

A Lean 4 (core only, **no Mathlib** dependency) formalization of the
quadray-coordinate foundations used by QuadMath. Package name: `Quadlean`.

## Build and run

Requires [elan](https://elan.lean-lang.org) (the toolchain is pinned in
`lean-toolchain`, currently `leanprover/lean4:v4.33.1`; `elan show` resolves
the same default). No other dependencies:

```bash
cd lean
lake build        # builds QuadMath.Quadray, QuadMath.IVM, QuadMath.Lattice
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
| `QuadMath.IVM` | The IVM (FCC/CCP) lattice in quadray coordinates: the 12 canonical neighbor moves (permutations of `(2,1,1,0)`), a computable membership predicate `IsIVMSite` (normalized ∧ component sum divisible by 4 — the `(1,0,0,0)`-type tetrahedral-void directions are excluded), the L1-centered shell norm `shellNorm4` (4x-scaled `quadray_shell_norm`; shell `k` ⟺ `shellNorm4 = 8k`), and a computable shell enumeration `shellSites` over the `[0, 2k]^4` scan box. | `src/ivm_field.py` — `IVM_NEIGHBOR_STEPS` (`:61-74`), `is_ivm_site` (`:81-89`), `quadray_shell_norm` (`:92-118`), `shell_sites` (`:121-157`); `neighbor_moves_ivm` in `src/discrete_variational.py:10-14`. |
| `QuadMath.Lattice` | The omnidirectional close-packing numbering: `omniNumber k = 10k^2 + 2` (shell `k ≥ 1`) and cumulative counts, cross-checked against the concrete `shellSites` enumeration. | `src/omni_numbering.py` (frequency shells, `10k^2 + 2`). |

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

## What is proved vs. stated

Proved in Lean (kernel-checked, core only):

* Normalization: non-negative components, minimum zero, idempotence, and
  class preservation (`normalize_spec`, `normalize_idem`, `normalize_sameClass`).
* The determinant identity: unit IVM tetrahedron `det = 4`, volume `= 1`;
  primitive tetrahedron `det = 1`, volume `= 1/4` (`tetraDet_unit`,
  `tetraVolume_unit`, `tetraDet_primitive`, `tetraVolume_primitive`); the
  determinant is invariant under shifting all four vertices by the same
  `t(1,1,1,1)` (`tetraDet_congr`).
* The 12 neighbor moves are pairwise distinct and are all IVM sites.
* Shell cardinalities for `k = 0, 1, 2`: `1`, `12`, `42` (`shellSites_*_length`),
  and `shellSites k` realizes `omniNumber k` on shells 1 and 2.

`#eval` smoke checks (printed at build time): membership examples (neighbor
class true, void direction false), `shellSites 3 = 92`, `shellSites 4 = 162`,
the `omniNumber` sequence `[2, 12, 42, 92, 162]`, cumulative counts
`[1, 13, 55, 147, 309, 561]`, and `shellSites (k+1)` agreeing with
`omniNumber (k+1)` for `k ≤ 4`.

Stated but **not** proved: `shellSites_card` — every shell `k ≥ 1` contains
exactly `10k^2 + 2` sites. It is the shell-count theorem statement requested
for this formalization and is deliberately left `sorry`-marked. The Python
reference machine-checks the corresponding claim through frequency 6
(`src/ivm_field.py:97-98`, `src/omni_numbering.py:24-26`), and an independent
enumeration cross-check through `k = 6` was run while authoring this tree;
the two shell characterizations used here agree exactly on all those shells.
The `[0, 2k]^4` bounding-box claim is taken from
`src/ivm_field.py:124-125` and is not re-proved in Lean.

## Constraints honored

* No Mathlib (or any other) dependency — `Int`, `Nat`, `List`, decidable
  propositions, `mkRat`, `#eval`.
* Nothing outside `lean/` was created or modified.