import QuadMath.Quadray
import QuadMath.IVM

/-!
# Exact squared distance under `DEFAULT_EMBEDDING`

Lean 4 formalization of the exact integer squared-distance identity used by
`src/quadray.py` (`DEFAULT_EMBEDDING` at `src/quadray.py:108-113`, `distance`
at `src/quadray.py:156-176`).  The embedding sends an integer delta
`δ = p - q = (da, db, dc, dd)` to the three projected rows

* row 1 `(1, -1, -1, 1)`:  `da - db - dc + dd`,
* row 2 `(1,  1, -1, -1)`: `da + db - dc - dd`,
* row 3 `(1, -1,  1, -1)`: `da - db + dc - dd`,

and `distance` returns the square root of the sum of their squares.  The
**exact identity** proved here — valid for *every* integer delta, with no
sum-zero precondition — is

`d²(p, q) = 4·(da² + db² + dc² + dd²) − (da + db + dc + dd)²`,

i.e. the three projected rows have Gram matrix `4·I` plus the rank-one
correction `−(Σδ)²`; on sum-zero deltas (deltas between lattice points of the
same class) the correction vanishes and `d² = 4·Σδ²` exactly.
-/

/-- Projected row 1 of `DEFAULT_EMBEDDING` applied to integer delta
components `(da, db, dc, dd)`: `(1, -1, -1, 1) · δ`
(`src/quadray.py:108-113`). -/
def deltaRow1 (da db dc dd : Int) : Int := da - db - dc + dd

/-- Projected row 2 of `DEFAULT_EMBEDDING`: `(1, 1, -1, -1) · δ`. -/
def deltaRow2 (da db dc dd : Int) : Int := da + db - dc - dd

/-- Projected row 3 of `DEFAULT_EMBEDDING`: `(1, -1, 1, -1) · δ`. -/
def deltaRow3 (da db dc dd : Int) : Int := da - db + dc - dd

/-- Exact squared Euclidean norm of an integer delta under
`DEFAULT_EMBEDDING`: the sum of squares of the three projected rows (the
Python `distance` returns the square root of this exact quantity). -/
def deltaDist2 (da db dc dd : Int) : Int :=
  (deltaRow1 da db dc dd) * (deltaRow1 da db dc dd)
    + (deltaRow2 da db dc dd) * (deltaRow2 da db dc dd)
    + (deltaRow3 da db dc dd) * (deltaRow3 da db dc dd)

/-- Exact squared Euclidean distance between lattice points `p` and `q` under
`DEFAULT_EMBEDDING` (`distance` in `src/quadray.py:156-176`). -/
def dist2 (p q : Quadray) : Int :=
  deltaDist2 (p.a - q.a) (p.b - q.b) (p.c - q.c) (p.d - q.d)

/-- **Exact squared-distance identity (universal, proved).**  For every
integer delta `(da, db, dc, dd)`,

`deltaDist2 da db dc dd = 4·(da² + db² + dc² + dd²) − (da + db + dc + dd)²`.

This is the exact form of the floating-point identity machine-checked against
the Python `distance` helper; the proof is a kernel-checked normalization of
the quadratic form (no Mathlib). -/
theorem deltaDist2_eq (da db dc dd : Int) :
    deltaDist2 da db dc dd
      = 4 * (da * da + db * db + dc * dc + dd * dd)
        - (da + db + dc + dd) * (da + db + dc + dd) := by
  unfold deltaDist2 deltaRow1 deltaRow2 deltaRow3
  simp only [Int.mul_sub, Int.sub_mul, Int.mul_add, Int.add_mul]
  omega

/-- The squared distance is symmetric in its two arguments. -/
theorem dist2_comm (p q : Quadray) : dist2 p q = dist2 q p := by
  have h1 := deltaDist2_eq (p.a - q.a) (p.b - q.b) (p.c - q.c) (p.d - q.d)
  have h2 := deltaDist2_eq (q.a - p.a) (q.b - p.b) (q.c - p.c) (q.d - p.d)
  show deltaDist2 (p.a - q.a) (p.b - q.b) (p.c - q.c) (p.d - q.d)
      = deltaDist2 (q.a - p.a) (q.b - p.b) (q.c - p.c) (q.d - p.d)
  rw [h1, h2]
  simp only [Int.mul_sub, Int.sub_mul, Int.mul_add, Int.add_mul]
  omega

/-- A point has squared distance `0` from itself. -/
theorem dist2_self (p : Quadray) : dist2 p p = 0 := by
  unfold dist2 deltaDist2 deltaRow1 deltaRow2 deltaRow3
  omega

-- Machine-checked instances: the twelve-around-one.
/-- Every neighbor move sits at squared Euclidean distance `8` from the
origin (radius `2·√2`, i.e. the `(2,1,1,0)`-type moves have Gram `4·I` norm
`8` under `DEFAULT_EMBEDDING`). -/
theorem neighborMoves_dist2 : (neighborMoves.all fun m => decide (dist2 ⟨0, 0, 0, 0⟩ m = 8)) = true :=
  rfl

/-- The distance of an arbitrary neighbor move from the origin is `8`. -/
theorem dist2_neighborMove (m : Quadray) (hm : m ∈ neighborMoves) :
    dist2 ⟨0, 0, 0, 0⟩ m = 8 := by
  exact of_decide_eq_true ((List.all_eq_true.mp neighborMoves_dist2) m hm)

-- Smoke checks (printed at build time):
#eval dist2 ⟨0, 0, 0, 0⟩ ⟨2, 1, 1, 0⟩   -- 8
#eval dist2 ⟨0, 0, 0, 0⟩ ⟨4, 2, 2, 0⟩   -- 32
#eval dist2 ⟨0, 0, 0, 0⟩ ⟨3, 1, 0, 0⟩   -- 24 (shell-2 site, non-uniform radius)
#eval dist2 ⟨1, 0, 0, 0⟩ ⟨0, 1, 0, 0⟩   -- 4
