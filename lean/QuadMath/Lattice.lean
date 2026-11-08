import QuadMath.IVM
/-!
# Omnidirectional close-packing numbering: `10k^2 + 2`

Lean 4 formalization of the close-packing numbering sequence.  Mirrors
`src/quadmath/lattice/omni_numbering.py` (shell counts, `10k² + 2`) and cross-checks the
abstract numbering function against the concrete lattice enumeration of
`QuadMath.IVM.shellSites`.

The cuboctahedral numbering formula for positive shells is defined below.
Its equality to the concrete `shellSites` enumeration is proved here only
for shells 1–3; `IVM.lean` supplies further finite instances through shell 8
with distinct kernel/native trust boundaries. Neither this formula definition
nor the printed computations establish a universal enumeration theorem.
-/

/-- Cuboctahedral numbering formula, intended for positive shell indices.
Equality to enumerated site counts requires a separate theorem. -/
def omniNumber (k : Nat) : Nat := 10 * k * k + 2

/-- Cumulative sum of the numbering formula plus the central site.
This definition alone does not prove a cardinality of an enumerated ball. -/
def cumulativeCount (k : Nat) : Nat :=
  1 + ((List.range k).map (fun j => omniNumber (j + 1))).sum

theorem omniNumber_1 : omniNumber 1 = 12 := rfl
theorem omniNumber_2 : omniNumber 2 = 42 := rfl
theorem omniNumber_3 : omniNumber 3 = 92 := rfl

/-- Cumulative counts through shell 3: `1 + 12 + 42 + 92 = 147`. -/
theorem cumulativeCount_3 : cumulativeCount 3 = 147 := rfl

/-- The computable shell enumeration realizes the numbering function on
shell 1. -/
theorem shellSites_omni_1 : (shellSites 1).length = omniNumber 1 := by decide

/-- The computable shell enumeration realizes the numbering function on
shell 2. -/
theorem shellSites_omni_2 : (shellSites 2).length = omniNumber 2 := by decide

/-- The computable enumeration realizes the numbering function on shell 3,
via the kernel-decided `shellSites_card_target_3` proof. -/
theorem shellSites_omni_3 : (shellSites 3).length = omniNumber 3 :=
  shellSites_card_target_3

-- Smoke checks (printed at build time):
#eval (List.range 9).map omniNumber       -- [2, 12, 42, 92, 162, 252, 362, 492, 642]
#eval (List.range 6).map cumulativeCount  -- [1, 13, 55, 147, 309, 561]
#eval (List.range 8).all fun k => (shellSites (k + 1)).length == omniNumber (k + 1)  -- true
#eval (shellSites 0).length == 1          -- true