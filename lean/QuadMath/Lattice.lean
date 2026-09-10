import QuadMath.IVM
/-!
# Omnidirectional close-packing numbering: `10k^2 + 2`

Lean 4 formalization of the close-packing numbering sequence.  Mirrors
`src/omni_numbering.py` (shell counts, `10k² + 2`) and cross-checks the
abstract numbering function against the concrete lattice enumeration of
`QuadMath.IVM.shellSites`.

Starting from a central sphere, the omnidirectional close packing builds up
in consecutive frequency shells: shell `k ≥ 1` carries exactly `10k² + 2`
sphere centers, and the center is the lone site of shell 0.  The first shell
is the 12 neighbors — the permutations of `(2, 1, 1, 0)` in quadray
coordinates — forming the cuboctahedron (vector equilibrium).
-/

/-- Number of lattice sites on frequency shell `k ≥ 1` of the omnidirectional
close packing (`src/omni_numbering.py:4-9`); shell 0 is the lone central
site and is not described by this function. -/
def omniNumber (k : Nat) : Nat := 10 * k * k + 2

/-- Cumulative site count through shell `k`: shell 0 contributes 1, each
shell `j ≥ 1` contributes `10j² + 2`. -/
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

-- Smoke checks (printed at build time):
#eval (List.range 5).map omniNumber       -- [2, 12, 42, 92, 162]
#eval (List.range 6).map cumulativeCount  -- [1, 13, 55, 147, 309, 561]
#eval (List.range 5).all fun k => (shellSites (k + 1)).length == omniNumber (k + 1)  -- true
#eval (shellSites 0).length == 1          -- true