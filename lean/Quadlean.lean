import QuadMath.Quadray
import QuadMath.IVM
import QuadMath.Distance
import QuadMath.Lattice

/-!
# Quadlean — Lean 4 formalization of the QuadMath core

Root module of the `lean/` formalization tree.  Core Lean 4 only (no
Mathlib): `Int`, `Nat`, `List`, decidable propositions, and `#eval` smoke
checks.

Module map:

* `QuadMath.Quadray` — quadray coordinates as integer 4-tuples: projective
  normalization and the exact integer tetra-volume determinant
  (`src/quadray.py`, `src/linalg_utils.py`).
* `QuadMath.IVM` — the IVM / FCC-CCP lattice in quadray coordinates: the 12
  neighbor moves, a computable membership predicate, the L1-centered shell
  norm, computable shell enumeration, its proved scan-box completeness and
  enumeration characterization, and machine-checked shell cardinalities
  through frequency 8 (`src/ivm_field.py`).
* `QuadMath.Distance` — the exact squared Euclidean distance under
  `DEFAULT_EMBEDDING`: the universal identity
  `d² = 4·Σδᵢ² − (Σδᵢ)²` over integer deltas (`src/quadray.py`).
* `QuadMath.Lattice` — the omnidirectional close-packing numbering sequence
  `10k^2 + 2` as a computable function with `#eval` smoke checks
  (`src/omni_numbering.py`).
-/