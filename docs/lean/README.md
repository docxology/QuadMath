# lean/ — Lean 4 formalization

The QuadMath repository carries a **Lean 4 formalization** of the quadray
foundations at `lean/` (repository root): integer quadray coordinates,
projective normalization, the exact integer tetra-volume determinant, IVM
lattice membership and shell enumeration, and the omnidirectional
close-packing numbering `10k² + 2`.

## Build

```bash
cd lean && lake build
```

- Toolchain pinned in `lean/lean-toolchain`: `leanprover/lean4:v4.33.1`.
- **Core Lean 4 only** — zero external packages (`lake-manifest.json` has an
  empty `packages` list); no Mathlib. Proofs rely on `Int`/`Nat`/`List`,
  `decide`, and `rfl`.
- The root module is `lean/Quadlean.lean` (library target `Quadlean`,
  `defaultTargets = ["Quadlean"]`).

## Module map

| Module | Mirrors | Contents |
|---|---|---|
| `lean/QuadMath/Quadray.lean` | `src/quadray.py`, `src/linalg_utils.py` | `Quadray` structure (integer 4-tuple), projective normalization (classes differ by multiples of `(1,1,1,1)`), exact tetra-volume determinant `|det| / 4`; unit IVM tetrahedron has determinant `4` and volume exactly `1` |
| `lean/QuadMath/IVM.lean` | `src/ivm_field.py` | The 12 neighbor moves (permutations of `(2,1,1,0)`), computable site membership `IsIVMSite` (normalized quadray with component sum ≡ 0 mod 4), L1-centered shell norm, computable `shellSites` enumeration via a bounded `quadBox` filter |
| `lean/QuadMath/Lattice.lean` | `src/omni_numbering.py` | `omniNumber k = 10k² + 2` (shells `k ≥ 1`), cumulative site counts, and cross-checks of the numbering function against the concrete `IVM.shellSites` enumeration |
| `lean/QuadMath/Distance.lean` | `src/lattice_search.py` | Universal squared-distance identity `d² = 4Σδ² − (Σδ)²` over integer deltas (`deltaDist2_eq`), neighbor-move distances (all 12 moves at d² = 8) |
| `lean/Quadlean.lean` | — | Root module importing the three modules above |

## Proof status (checked 2026-09-10)

- `lake build` exits 0.
- **Zero sorries.** The former sorry'd `shellSites_card` was deleted and
  restated honestly as `def shellSites_card_target` (universal
  `(shellSites k).length = 10k² + 2` for `k ≥ 1`), machine-checked through
  shell 8: `k ∈ {1..4}` by kernel `decide` (no axioms), `k ∈ {5..8}` by
  `native_decide` (one per-instance VM-trust axiom, disclosed in
  `lean/README.md`). `#eval` smoke checks reach shell 8
  ([12, 42, 92, 162, 252, 362, 492, 642]).
- **Newly proved (kernel-checked; propext/Quot.sound only)**:
  `shellNorm4_component_bound` (scan-box completeness — every shell-k site has
  components in [0, 2k]; previously taken from the Python reference),
  `mem_shellSites_iff` (exact enumeration characterization),
  `shellSites_unique`, `isIVMSite_normalize`, `shellNorm4_sameClass`, and
  `Distance.deltaDist2_eq` — the universal squared-distance identity
  `d² = 4Σδ² − (Σδ)²` matching `src/lattice_search.py`.
- The universal shell count remains an explicit open obligation (proof sketch
  in `lean/README.md`: sum-zero representative bijection; missing step =
  composition counting in core Lean).
- Everything else is fully proved: the neighbor-move inventory (length 12,
  no duplicates, all are IVM sites), the normalization congruence
  (`ivmSum_normalize`), the shell-0/1/2 cardinalities, `omniNumber`/`cumulativeCount`
  instances (12, 42, 92; cumulative 147 through shell 3), and the
  enumeration↔numbering agreement on shells 1–2.

## Relationship to the manuscript

Lean statements are a third witness alongside the Python implementations and
the analytical text: volume identities in
[the equations appendix](../manuscript/08_equations_appendix.md), IVM shell
structure and dynamics in
[12_ivm_dynamics.md](../manuscript/12_ivm_dynamics.md) (with the field-learning
treatment in source section `11_ivm_field_learning.md`), and the `10k² + 2`
sequence mirrored in `src/omni_numbering.py`.
