import QuadMath.Quadray
import QuadMath.IVM
import Init.Data.Rat.Lemmas

/-!
# Exact squared distance under `DEFAULT_EMBEDDING`

Lean 4 formalization of the exact integer squared-distance identity used by
`src/quadmath/core/quadray.py` (`DEFAULT_EMBEDDING` at `src/quadmath/core/quadray.py:116-121`, `distance`
at `src/quadmath/core/quadray.py:165-179`).  The embedding sends an integer delta
`δ = p - q = (da, db, dc, dd)` to the three projected rows

* row 1 `(1, -1, -1, 1)`:  `da - db - dc + dd`,
* row 2 `(1,  1, -1, -1)`: `da + db - dc - dd`,
* row 3 `(1, -1,  1, -1)`: `da - db + dc - dd`,

and `distance` returns the square root of the sum of their squares.  The
**exact identity** proved here — valid for *every* integer delta, with no
sum-zero precondition — is

`d²(p, q) = 4·(da² + db² + dc² + dd²) − (da + db + dc + dd)²`,

i.e. the Gram matrix is `4·I − 11ᵀ`. The correction vanishes on sum-zero
deltas (for example, between representatives with equal component sums).
This is not the same as two labels belonging to one physical class.
-/

/-- Projected row 1 of `DEFAULT_EMBEDDING` applied to integer delta
components `(da, db, dc, dd)`: `(1, -1, -1, 1) · δ`
(`src/quadmath/core/quadray.py:116-121`). -/
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
`DEFAULT_EMBEDDING` (`distance` in `src/quadmath/core/quadray.py:165-179`). -/
def dist2 (p q : Quadray) : Int :=
  deltaDist2 (p.a - q.a) (p.b - q.b) (p.c - q.c) (p.d - q.d)

/-- **Exact squared-distance identity (universal, proved).**  For every
integer delta `(da, db, dc, dd)`,

`deltaDist2 da db dc dd = 4·(da² + db² + dc² + dd²) − (da + db + dc + dd)²`.

This is a handwritten exact-arithmetic model, not a verification of the
Python implementation or its floating-point evaluation. The proof normalizes
the quadratic form using core integer arithmetic, without Mathlib. -/
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

/-- The scale-one integer XYZ image, with the row order of `DEFAULT_EMBEDDING`. -/
def xyzInt (q : Quadray) : Int × Int × Int :=
  (deltaRow1 q.a q.b q.c q.d, deltaRow2 q.a q.b q.c q.d,
    deltaRow3 q.a q.b q.c q.d)

theorem xyzInt_sameClass {q r : Quadray} (h : SameClass q r) :
    xyzInt q = xyzInt r := by
  obtain ⟨t, ha, hb, hc, hd⟩ := h
  apply Prod.ext
  · simp only [xyzInt, deltaRow1]
    omega
  · apply Prod.ext
    · simp only [xyzInt, deltaRow2]
      omega
    · simp only [xyzInt, deltaRow3]
      omega

/-- Equality of the physical images identifies exactly the common-offset classes. -/
theorem xyzInt_eq_iff_sameClass (q r : Quadray) :
    xyzInt q = xyzInt r ↔ SameClass q r := by
  constructor
  · intro h
    have ha := congrArg (fun v : Int × Int × Int => v.1) h
    have hb := congrArg (fun v : Int × Int × Int => v.2.1) h
    have hc := congrArg (fun v : Int × Int × Int => v.2.2) h
    simp only [xyzInt, deltaRow1, deltaRow2, deltaRow3] at ha hb hc
    exact ⟨r.d - q.d, by omega, by omega, by omega, by omega⟩
  · exact xyzInt_sameClass

private theorem square_nonneg (z : Int) : 0 ≤ z * z := by
  by_cases hz : 0 ≤ z
  · exact Int.mul_nonneg hz hz
  · exact Int.mul_nonneg_of_nonpos_of_nonpos (by omega) (by omega)

/-- Squared distance is nonnegative for all raw integer labels. -/
theorem dist2_nonneg (p q : Quadray) : 0 ≤ dist2 p q := by
  have h1 := square_nonneg (deltaRow1 (p.a-q.a) (p.b-q.b) (p.c-q.c) (p.d-q.d))
  have h2 := square_nonneg (deltaRow2 (p.a-q.a) (p.b-q.b) (p.c-q.c) (p.d-q.d))
  have h3 := square_nonneg (deltaRow3 (p.a-q.a) (p.b-q.b) (p.c-q.c) (p.d-q.d))
  unfold dist2 deltaDist2
  omega

/-- Squared distance descends to independently gauged physical endpoints. -/
theorem dist2_congr {p p' q q' : Quadray}
    (hp : SameClass p p') (hq : SameClass q q') :
    dist2 p q = dist2 p' q' := by
  have h := xyzInt_sameClass (sameClass_sub hp hq)
  exact congrArg (fun v : Int × Int × Int =>
    v.1 * v.1 + v.2.1 * v.2.1 + v.2.2 * v.2.2) h

/-- Separation is physical, not equality of the four raw integer labels. -/
theorem dist2_eq_zero_iff (p q : Quadray) : dist2 p q = 0 ↔ SameClass p q := by
  constructor
  · intro h
    let x := deltaRow1 (p.a-q.a) (p.b-q.b) (p.c-q.c) (p.d-q.d)
    let y := deltaRow2 (p.a-q.a) (p.b-q.b) (p.c-q.c) (p.d-q.d)
    let z := deltaRow3 (p.a-q.a) (p.b-q.b) (p.c-q.c) (p.d-q.d)
    have hx := square_nonneg x
    have hy := square_nonneg y
    have hz := square_nonneg z
    have hs : x*x + y*y + z*z = 0 := h
    have hx0 : x*x = 0 := by omega
    have hy0 : y*y = 0 := by omega
    have hz0 : z*z = 0 := by omega
    have ex : x = 0 := (Int.mul_eq_zero.mp hx0).elim id id
    have ey : y = 0 := (Int.mul_eq_zero.mp hy0).elim id id
    have ez : z = 0 := (Int.mul_eq_zero.mp hz0).elim id id
    dsimp [x, y, z, deltaRow1, deltaRow2, deltaRow3] at ex ey ez
    exact ⟨q.d - p.d, by omega, by omega, by omega, by omega⟩
  · intro h
    rw [dist2_congr (sameClass_refl p) (sameClass_symm h)]
    exact dist2_self p

theorem dist2_eq_zero_iff_normalize_eq (p q : Quadray) :
    dist2 p q = 0 ↔ p.normalize = q.normalize :=
  (dist2_eq_zero_iff p q).trans (normalize_eq_iff_sameClass p q).symm

private theorem xyzInt_proj (q : Quadray) :
    xyzInt q = ((proj q).1 - (proj q).2.1 - (proj q).2.2,
      (proj q).1 + (proj q).2.1 - (proj q).2.2,
      (proj q).1 - (proj q).2.1 + (proj q).2.2) := by
  apply Prod.ext
  · simp only [xyzInt, proj, deltaRow1]
    omega
  · apply Prod.ext
    · simp only [xyzInt, proj, deltaRow2]
      omega
    · simp only [xyzInt, proj, deltaRow3]
      omega

/-- The fixed transform from projected coordinates to scale-one XYZ has determinant +4. -/
private theorem det3_xyz_transform (u v w : Int × Int × Int) :
    det3 (u.1-u.2.1-u.2.2, u.1+u.2.1-u.2.2, u.1-u.2.1+u.2.2)
      (v.1-v.2.1-v.2.2, v.1+v.2.1-v.2.2, v.1-v.2.1+v.2.2)
      (w.1-w.2.1-w.2.2, w.1+w.2.1-w.2.2, w.1-w.2.1+w.2.2)
      = 4 * det3 u v w := by
  unfold det3
  simp only [Int.mul_sub, Int.sub_mul, Int.mul_add, Int.add_mul]
  omega

/-- Signed XYZ edge determinant, not yet divided by the Euclidean factor six. -/
def xyzTetraDet (p0 p1 p2 p3 : Quadray) : Int :=
  det3 (xyzInt (p1.sub p0)) (xyzInt (p2.sub p0)) (xyzInt (p3.sub p0))

/-- Universal orientation-sensitive embedding bridge in the scale-one row order. -/
theorem xyzTetraDet_eq (p0 p1 p2 p3 : Quadray) :
    xyzTetraDet p0 p1 p2 p3 = 4 * tetraDet p0 p1 p2 p3 := by
  unfold xyzTetraDet tetraDet
  rw [xyzInt_proj, xyzInt_proj, xyzInt_proj]
  exact det3_xyz_transform _ _ _

/-- Exact Euclidean volume for the scale-one integer embedding. -/
def xyzTetraVolume (p0 p1 p2 p3 : Quadray) : Rat :=
  mkRat (xyzTetraDet p0 p1 p2 p3).natAbs 6

/-- Native IVM volume is 3/8 of scale-one XYZ volume, not the rescaled S3 factor. -/
theorem tetraVolume_xyz_bridge (p0 p1 p2 p3 : Quadray) :
    mkRat 3 8 * xyzTetraVolume p0 p1 p2 p3 = tetraVolume p0 p1 p2 p3 := by
  unfold xyzTetraVolume tetraVolume
  rw [Rat.mkRat_mul_mkRat]
  apply (Rat.mkRat_eq_iff (by decide) (by decide)).mpr
  rw [xyzTetraDet_eq]
  have hfour : ((4 : Int).natAbs : Int) = 4 := by decide
  simp only [Int.natAbs_mul, Int.natCast_mul, hfour]
  omega

theorem xyzTetraVolume_unit :
    xyzTetraVolume ⟨0,0,0,0⟩ ⟨2,1,1,0⟩ ⟨1,2,1,0⟩ ⟨1,1,2,0⟩ = mkRat 8 3 := by
  decide

/-- Integrality, not a real-valued inequality: `|z| ≤ z²` for integer `z`. -/
private theorem natAbs_le_square (z : Int) : (z.natAbs : Int) ≤ z*z := by
  by_cases hz : 0 ≤ z
  · rw [Int.natAbs_of_nonneg hz]
    by_cases h0 : z = 0
    · subst z
      decide
    · have hm := Int.mul_nonneg hz (show 0 ≤ z - 1 by omega)
      simp only [Int.mul_sub, Int.mul_one] at hm
      omega
  · rw [Int.ofNat_natAbs_of_nonpos (show z ≤ 0 by omega)]
    have hm := Int.mul_nonneg_of_nonpos_of_nonpos
      (show z ≤ 0 by omega) (show z + 1 ≤ 0 by omega)
    simp only [Int.mul_add, Int.mul_one] at hm
    omega

/-- A sum-zero integer gauge bounds its centered L1 shell measure by squared radius. -/
theorem shellNorm4_le_dist2_of_sum_zero (q : Quadray) (hq : ivmSum q = 0) :
    (shellNorm4 q : Int) ≤ dist2 ⟨0,0,0,0⟩ q := by
  rw [dist2_comm]
  have hd := deltaDist2_eq q.a q.b q.c q.d
  have hs : q.a + q.b + q.c + q.d = 0 := hq
  have ha := natAbs_le_square q.a
  have hb := natAbs_le_square q.b
  have hc := natAbs_le_square q.c
  have he := natAbs_le_square q.d
  have hfour : ((4 : Int).natAbs : Int) = 4 := by decide
  simp only [shellNorm4, hq, Int.sub_zero, Int.natAbs_mul,
    Int.natCast_add, Int.natCast_mul, hfour]
  simp only [hs, Int.zero_mul, Int.sub_zero] at hd
  simpa only [dist2, Int.sub_zero] using
    (show 4 * (q.a.natAbs : Int) + 4 * (q.b.natAbs : Int) +
      4 * (q.c.natAbs : Int) + 4 * (q.d.natAbs : Int) ≤
        deltaDist2 q.a q.b q.c q.d by omega)

/-- Universal radial bound for integer IVM classes, without a walk-count assumption. -/
theorem shellNorm4_le_dist2 (q : Quadray) (hq : ivmSum q % 4 = 0) :
    (shellNorm4 q : Int) ≤ dist2 ⟨0,0,0,0⟩ q := by
  have h := shellNorm4_le_dist2_of_sum_zero (ivmCentered q) (ivmCentered_sum_zero q hq)
  rw [shellNorm4_centered] at h
  rw [dist2_congr (sameClass_refl ⟨0,0,0,0⟩) (ivmCentered_sameClass q)]
  exact h

/-- Every site on enumerated shell k has squared scale-one radius at least 8k.
This concerns the centered-norm enumeration; no BFS equivalence is assumed or proved. -/
theorem shellSites_dist2_lower (q : Quadray) (k : Nat) (hq : q ∈ shellSites k) :
    8 * (k : Int) ≤ dist2 ⟨0,0,0,0⟩ q := by
  obtain ⟨hsite, hn⟩ := (mem_shellSites_iff q k).mp hq
  have h := shellNorm4_le_dist2 q hsite.2
  rw [hn] at h
  have height : ((8 : Nat) : Int) = 8 := by decide
  simpa only [Int.natCast_mul, height] using h

/-- A strict squared-radius cutoff excludes every subsequent centered-norm shell. -/
theorem shellSites_outside_cutoff (q : Quadray) (k : Nat) (cutoff : Int)
    (hq : q ∈ shellSites k) (hk : cutoff < 8 * (k : Int)) :
    cutoff < dist2 ⟨0,0,0,0⟩ q := by
  have h := shellSites_dist2_lower q k hq
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
