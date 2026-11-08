/-!
# Quadray coordinates (Fuller.4D)

Lean 4 formalization of the integer quadray coordinates used throughout
QuadMath.  This module mirrors the Python reference implementation in
`src/quadmath/core/quadray.py` (the `Quadray` dataclass, projective normalization, and the
exact integer tetra-volume determinant) and `src/quadmath/core/linalg_utils.py`
(`bareiss_determinant_int`).

Conventions (see `src/quadmath/core/quadray.py`):

* A quadray is an integer 4-tuple `(a, b, c, d)`.
* Two quadrays are *projectively equivalent* when they differ by a multiple
  of `(1, 1, 1, 1)`; `Quadray.normalize` selects the canonical representative
  of the class by subtracting the minimum component (`src/quadmath/core/quadray.py:30-37`).
* The exact IVM tetra-volume of the lattice tetrahedron spanned by `p0, p1,
  p2, p3` is `|det| / 4`, where `det` is the 3x3 integer determinant of the
  `(a - d, b - d, c - d)` projections of the edge vectors `p1 - p0`,
  `p2 - p0`, `p3 - p0` (`src/quadmath/core/quadray.py:61-83`).  The unit IVM tetrahedron
  (origin plus three permutations of `(2, 1, 1, 0)`) has determinant `4` and
  volume exactly `1`; the primitive tetrahedron spanned by the origin and the
  basis vectors has determinant `1` and volume exactly `1/4`.
-/

/-- Quadray vector: integer 4-tuple `(a, b, c, d)` (Fuller.4D). -/
structure Quadray where
  /-- First coordinate. -/
  a : Int
  /-- Second coordinate. -/
  b : Int
  /-- Third coordinate. -/
  c : Int
  /-- Fourth coordinate. -/
  d : Int
deriving Repr, DecidableEq

/-- Component-wise sum (`src/quadmath/core/quadray.py:38-40`). -/
def Quadray.add (q r : Quadray) : Quadray :=
  ⟨q.a + r.a, q.b + r.b, q.c + r.c, q.d + r.d⟩

/-- Component-wise difference (`src/quadmath/core/quadray.py:42-44`). -/
def Quadray.sub (q r : Quadray) : Quadray :=
  ⟨q.a - r.a, q.b - r.b, q.c - r.c, q.d - r.d⟩

/-- Minimum of the four components (`src/quadmath/core/quadray.py:35`). -/
def quadMin (q : Quadray) : Int :=
  min q.a (min q.b (min q.c q.d))

/-- `min x y` equals one of its arguments (`Int.min_def` case split). -/
private theorem min2_eq (x y : Int) : min x y = x ∨ min x y = y := by
  rw [Int.min_def]
  split
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- `min x y` is bounded by its left argument (`Int.min_def` case split). -/
private theorem min2_le_left (x y : Int) : min x y ≤ x := by
  rw [Int.min_def]
  split <;> omega

/-- `min x y` is bounded by its right argument (`Int.min_def` case split). -/
private theorem min2_le_right (x y : Int) : min x y ≤ y := by
  rw [Int.min_def]
  split <;> omega

/-- The minimum of the four components bounds each component. -/
theorem quadMin_le (q : Quadray) :
    quadMin q ≤ q.a ∧ quadMin q ≤ q.b ∧ quadMin q ≤ q.c ∧ quadMin q ≤ q.d := by
  have h1 : quadMin q ≤ min q.b (min q.c q.d) := min2_le_right _ _
  have h2 : min q.b (min q.c q.d) ≤ q.b := min2_le_left _ _
  have h3 : min q.b (min q.c q.d) ≤ min q.c q.d := min2_le_right _ _
  have h4 : min q.c q.d ≤ q.c := min2_le_left _ _
  have h5 : min q.c q.d ≤ q.d := min2_le_right _ _
  refine ⟨min2_le_left _ _, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega

/-- The minimum of the four components is attained by one of them. -/
theorem quadMin_eq (q : Quadray) :
    quadMin q = q.a ∨ quadMin q = q.b ∨ quadMin q = q.c ∨ quadMin q = q.d := by
  show min q.a (min q.b (min q.c q.d)) = q.a ∨
    min q.a (min q.b (min q.c q.d)) = q.b ∨
    min q.a (min q.b (min q.c q.d)) = q.c ∨
    min q.a (min q.b (min q.c q.d)) = q.d
  rcases min2_eq q.a (min q.b (min q.c q.d)) with h | h
  · exact Or.inl h
  · rw [h]
    rcases min2_eq q.b (min q.c q.d) with h2 | h2
    · exact Or.inr (Or.inl h2)
    · rw [h2]
      exact Or.inr (Or.inr (min2_eq q.c q.d))

/-- Projective normalization: translate by `-(k,k,k,k)` so at least one
component is zero (`src/quadmath/core/quadray.py:30-37`). -/
def Quadray.normalize (q : Quadray) : Quadray :=
  ⟨q.a - quadMin q, q.b - quadMin q, q.c - quadMin q, q.d - quadMin q⟩

/-- Projective quadray equivalence `q ~ q + t(1,1,1,1)`
(`src/quadmath/core/quadray.py:32-34`). -/
def SameClass (q r : Quadray) : Prop :=
  ∃ t : Int, r.a = q.a + t ∧ r.b = q.b + t ∧ r.c = q.c + t ∧ r.d = q.d + t

/-- Common-offset equivalence is reflexive. -/
theorem sameClass_refl (q : Quadray) : SameClass q q := by
  exact ⟨0, by omega, by omega, by omega, by omega⟩

/-- Common-offset equivalence is symmetric. -/
theorem sameClass_symm {q r : Quadray} (h : SameClass q r) : SameClass r q := by
  obtain ⟨t, ha, hb, hc, hd⟩ := h
  exact ⟨-t, by omega, by omega, by omega, by omega⟩

/-- Common-offset equivalence is transitive. -/
theorem sameClass_trans {q r s : Quadray}
    (hqr : SameClass q r) (hrs : SameClass r s) : SameClass q s := by
  obtain ⟨t, ha, hb, hc, hd⟩ := hqr
  obtain ⟨u, ha', hb', hc', hd'⟩ := hrs
  exact ⟨t + u, by omega, by omega, by omega, by omega⟩

/-- The additive common-offset quotient, not an identification of raw game cells. -/
def quadraySetoid : Setoid Quadray where
  r := SameClass
  iseqv := ⟨sameClass_refl, sameClass_symm, sameClass_trans⟩

/-- Integer physical points modulo independent common offsets. -/
def PhysicalQuadray := Quotient quadraySetoid

/-- Place a raw integer label in its physical equivalence class. -/
def physicalClass (q : Quadray) : PhysicalQuadray := Quotient.mk quadraySetoid q

theorem physicalClass_eq_iff (q r : Quadray) :
    physicalClass q = physicalClass r ↔ SameClass q r := by
  constructor
  · exact Quotient.exact
  · intro h
    change Quotient.mk quadraySetoid q = Quotient.mk quadraySetoid r
    exact Quotient.sound h

private theorem quadray_ext {q r : Quadray}
    (ha : q.a = r.a) (hb : q.b = r.b) (hc : q.c = r.c) (hd : q.d = r.d) :
    q = r := by
  cases q
  cases r
  simp_all

/-- Independently changing representatives also preserves displacement classes. -/
theorem sameClass_sub {p p' q q' : Quadray}
    (hp : SameClass p p') (hq : SameClass q q') :
    SameClass (p.sub q) (p'.sub q') := by
  obtain ⟨t, ha, hb, hc, hd⟩ := hp
  obtain ⟨u, ha', hb', hc', hd'⟩ := hq
  refine ⟨t - u, ?_, ?_, ?_, ?_⟩
  all_goals simp only [Quadray.sub] <;> omega

/-- Normalization stays inside the projective class: `normalize q ~ q`. -/
theorem normalize_sameClass (q : Quadray) : SameClass q q.normalize := by
  refine ⟨-(quadMin q), ?_, ?_, ?_, ?_⟩
  all_goals simp only [Quadray.normalize] <;> omega

/-- Normalized quadrays have all components non-negative with at least one
component equal to zero. -/
theorem normalize_spec (q : Quadray) :
    0 ≤ q.normalize.a ∧ 0 ≤ q.normalize.b ∧ 0 ≤ q.normalize.c ∧ 0 ≤ q.normalize.d ∧
      (q.normalize.a = 0 ∨ q.normalize.b = 0 ∨ q.normalize.c = 0 ∨ q.normalize.d = 0) := by
  have h1 := quadMin_le q
  have h2 := quadMin_eq q
  show 0 ≤ q.a - quadMin q ∧ 0 ≤ q.b - quadMin q ∧ 0 ≤ q.c - quadMin q ∧
    0 ≤ q.d - quadMin q ∧
    (q.a - quadMin q = 0 ∨ q.b - quadMin q = 0 ∨ q.c - quadMin q = 0 ∨
      q.d - quadMin q = 0)
  omega

private theorem min2_zero_left (y : Int) (hy : 0 ≤ y) : min 0 y = 0 := by
  rcases min2_eq 0 y with h | h <;> omega

private theorem min2_zero_right (x : Int) (hx : 0 ≤ x) : min x 0 = 0 := by
  rcases min2_eq x 0 with h | h <;> omega

private theorem zero_le_min2 (x y : Int) (hx : 0 ≤ x) (hy : 0 ≤ y) : 0 ≤ min x y := by
  rcases min2_eq x y with h | h <;> omega

/-- The minimum of four non-negative integers, at least one of which is zero,
is zero. -/
private theorem min4_zero {a b c d : Int}
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hz : a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0) :
    min a (min b (min c d)) = 0 := by
  rcases hz with rfl | rfl | rfl | rfl
  · exact min2_zero_left _ (zero_le_min2 b (min c d) hb (zero_le_min2 c d hc hd))
  · rw [min2_zero_left (min c d) (zero_le_min2 c d hc hd)]
    exact min2_zero_right a ha
  · rw [min2_zero_left d hd, min2_zero_right b hb, min2_zero_right a ha]
  · rw [min2_zero_right c hc, min2_zero_right b hb, min2_zero_right a ha]

/-- The minimum of the components of a normalized quadray is zero. -/
theorem quadMin_normalize (q : Quadray) : quadMin q.normalize = 0 :=
  min4_zero (normalize_spec q).1 (normalize_spec q).2.1 (normalize_spec q).2.2.1
    (normalize_spec q).2.2.2.1 (normalize_spec q).2.2.2.2

/-- Normalization is idempotent: the canonical representative of the
canonical representative is itself. -/
theorem normalize_idem (q : Quadray) : q.normalize.normalize = q.normalize := by
  have h0 : quadMin q.normalize = 0 := quadMin_normalize q
  have hrfl : q.normalize.normalize
      = ⟨q.normalize.a - quadMin q.normalize, q.normalize.b - quadMin q.normalize,
         q.normalize.c - quadMin q.normalize, q.normalize.d - quadMin q.normalize⟩ := rfl
  rw [hrfl, h0]
  have e1 : q.normalize.a - 0 = q.normalize.a := by omega
  have e2 : q.normalize.b - 0 = q.normalize.b := by omega
  have e3 : q.normalize.c - 0 = q.normalize.c := by omega
  have e4 : q.normalize.d - 0 = q.normalize.d := by omega
  rw [e1, e2, e3, e4]

/-- A representative shift translates the minimum by exactly that shift. -/
theorem quadMin_shift {q r : Quadray} (t : Int)
    (ha : r.a = q.a + t) (hb : r.b = q.b + t)
    (hc : r.c = q.c + t) (hd : r.d = q.d + t) :
    quadMin r = quadMin q + t := by
  have hq := quadMin_le q
  have hr := quadMin_le r
  rcases quadMin_eq q with h | h | h | h <;>
    rcases quadMin_eq r with h' | h' | h' | h' <;> omega

/-- Every physical class has exactly one minimum-zero canonical representative. -/
theorem normalize_eq_iff_sameClass (q r : Quadray) :
    q.normalize = r.normalize ↔ SameClass q r := by
  constructor
  · intro h
    apply sameClass_trans (normalize_sameClass q)
    rw [h]
    exact sameClass_symm (normalize_sameClass r)
  · rintro ⟨t, ha, hb, hc, hd⟩
    have hm := quadMin_shift t ha hb hc hd
    apply quadray_ext
    all_goals simp only [Quadray.normalize] <;> omega

/-- A minimum-zero representative is unique even before normalization is applied. -/
theorem canonical_unique {q r : Quadray} (hq : quadMin q = 0)
    (hr : quadMin r = 0) (h : SameClass q r) : q = r := by
  have hn := (normalize_eq_iff_sameClass q r).mpr h
  simpa only [Quadray.normalize, hq, hr, Int.sub_zero] using hn

/-- Row of the projected 3x3 volume matrix: the `(a - d, b - d, c - d)`
projection (`src/quadmath/core/quadray.py:77-78`). -/
def proj (q : Quadray) : Int × Int × Int := (q.a - q.d, q.b - q.d, q.c - q.d)

/-- Projection identifies precisely the common-offset classes, universally. -/
theorem sameClass_iff_proj_eq (q r : Quadray) : SameClass q r ↔ proj q = proj r := by
  constructor
  · rintro ⟨t, ha, hb, hc, hd⟩
    apply Prod.ext
    · simp only [proj]
      omega
    · apply Prod.ext <;> simp only [proj] <;> omega
  · intro h
    have ha := congrArg (fun v : Int × Int × Int => v.1) h
    have hb := congrArg (fun v : Int × Int × Int => v.2.1) h
    have hc := congrArg (fun v : Int × Int × Int => v.2.2) h
    simp only [proj] at ha hb hc
    exact ⟨r.d - q.d, by omega, by omega, by omega, by omega⟩

/-- Every integer triple occurs: the quotient has exactly three integer coordinates. -/
theorem proj_surjective (v : Int × Int × Int) : ∃ q : Quadray, proj q = v := by
  refine ⟨⟨v.1, v.2.1, v.2.2, 0⟩, ?_⟩
  simp only [proj, Int.sub_zero]

/-- Canonical equality and projection equality are the same physical identity test. -/
theorem normalize_eq_iff_proj_eq (q r : Quadray) :
    q.normalize = r.normalize ↔ proj q = proj r :=
  (normalize_eq_iff_sameClass q r).trans (sameClass_iff_proj_eq q r)

/-- Exact 3x3 integer determinant (cofactor expansion along the first row),
the arithmetic core of `bareiss_determinant_int` for 3x3 inputs
(`src/quadmath/core/linalg_utils.py:6-56`). -/
def det3 (r1 r2 r3 : Int × Int × Int) : Int :=
  r1.1 * (r2.2.1 * r3.2.2 - r2.2.2 * r3.2.1)
    - r1.2.1 * (r2.1 * r3.2.2 - r2.2.2 * r3.1)
    + r1.2.2 * (r2.1 * r3.2.1 - r2.2.1 * r3.1)

/-- Exact IVM tetra-determinant of the lattice tetrahedron `(p0, p1, p2, p3)`:
the 3x3 integer determinant of the `(a-d, b-d, c-d)` projections of the edge
vectors `p1 - p0`, `p2 - p0`, `p3 - p0` (`src/quadmath/core/quadray.py:61-83`). -/
def tetraDet (p0 p1 p2 p3 : Quadray) : Int :=
  det3 (proj (Quadray.sub p1 p0)) (proj (Quadray.sub p2 p0))
    (proj (Quadray.sub p3 p0))

/-- Exact IVM tetra-volume `|det| / 4` (`src/quadmath/core/quadray.py:61-83`); general
lattice tetrahedra may have non-integral volume, so no integrality is
assumed. -/
def tetraVolume (p0 p1 p2 p3 : Quadray) : Rat :=
  mkRat (Int.natAbs (tetraDet p0 p1 p2 p3)) 4

private theorem sub_shift (x y t : Int) : x + t - (y + t) = x - y := by omega

/-- Simultaneously shifting all four vertices by one common gauge leaves the
tetra-determinant unchanged; `tetraDet_sameClass` below allows independent gauges. -/
theorem tetraDet_congr (t : Int) (p0 p1 p2 p3 : Quadray) :
    tetraDet ⟨p0.a + t, p0.b + t, p0.c + t, p0.d + t⟩
        ⟨p1.a + t, p1.b + t, p1.c + t, p1.d + t⟩
        ⟨p2.a + t, p2.b + t, p2.c + t, p2.d + t⟩
        ⟨p3.a + t, p3.b + t, p3.c + t, p3.d + t⟩
      = tetraDet p0 p1 p2 p3 := by
  simp only [tetraDet, Quadray.sub, proj, sub_shift]

/-- Each vertex may change its gauge independently, including the base vertex. -/
theorem tetraDet_sameClass {p0 p1 p2 p3 r0 r1 r2 r3 : Quadray}
    (h0 : SameClass p0 r0) (h1 : SameClass p1 r1)
    (h2 : SameClass p2 r2) (h3 : SameClass p3 r3) :
    tetraDet p0 p1 p2 p3 = tetraDet r0 r1 r2 r3 := by
  unfold tetraDet
  rw [(sameClass_iff_proj_eq _ _).mp (sameClass_sub h1 h0),
    (sameClass_iff_proj_eq _ _).mp (sameClass_sub h2 h0),
    (sameClass_iff_proj_eq _ _).mp (sameClass_sub h3 h0)]

/-- Exact quarter-unit volume descends to four independently chosen physical classes. -/
theorem tetraVolume_sameClass {p0 p1 p2 p3 r0 r1 r2 r3 : Quadray}
    (h0 : SameClass p0 r0) (h1 : SameClass p1 r1)
    (h2 : SameClass p2 r2) (h3 : SameClass p3 r3) :
    tetraVolume p0 p1 p2 p3 = tetraVolume r0 r1 r2 r3 := by
  unfold tetraVolume
  rw [tetraDet_sameClass h0 h1 h2 h3]

/-- The unit IVM tetrahedron (origin plus three `(2,1,1,0)`-type neighbor
moves) has determinant `4` and IVM volume exactly `1`
(`src/quadmath/core/quadray.py:67-69`, `tests/unit/core/test_quadray.py:33-40`). -/
theorem tetraDet_unit :
    tetraDet ⟨0, 0, 0, 0⟩ ⟨2, 1, 1, 0⟩ ⟨1, 2, 1, 0⟩ ⟨1, 1, 2, 0⟩ = 4 := by decide

theorem tetraVolume_unit :
    tetraVolume ⟨0, 0, 0, 0⟩ ⟨2, 1, 1, 0⟩ ⟨1, 2, 1, 0⟩ ⟨1, 1, 2, 0⟩ = 1 := by decide

/-- The primitive tetrahedron spanned by the origin and the basis vectors has
determinant `1` and IVM volume exactly `1/4` (`src/quadmath/core/quadray.py:69-71`,
`tests/unit/core/test_quadray.py:43-50`). -/
theorem tetraDet_primitive :
    tetraDet ⟨0, 0, 0, 0⟩ ⟨1, 0, 0, 0⟩ ⟨0, 1, 0, 0⟩ ⟨0, 0, 1, 0⟩ = 1 := by decide

theorem tetraVolume_primitive :
    tetraVolume ⟨0, 0, 0, 0⟩ ⟨1, 0, 0, 0⟩ ⟨0, 1, 0, 0⟩ ⟨0, 0, 1, 0⟩ = mkRat 1 4 := by
  decide