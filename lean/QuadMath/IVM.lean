import QuadMath.Quadray
/-!
# The IVM lattice (isotropic vector matrix, FCC/CCP) in quadray coordinates

Lean 4 formalization of the IVM lattice as QuadMath defines it.  Mirrors
`src/quadmath/lattice/ivm_field.py` (membership test, L1-centered shell norm, shell
enumeration) and `src/quadmath/optimize/discrete_variational.py` (`neighbor_moves_ivm`).

Conventions (see `src/quadmath/lattice/ivm_field.py` and the manuscript sections):

* A *normalized* quadray has non-negative components and minimum zero — the
  canonical representative of its projective class (`src/quadmath/core/quadray.py:30-37`).
* A canonical *IVM lattice site* is a normalized quadray whose component sum
  is divisible by 4. The class-level predicate is `IsIVMSite q.normalize`;
  `IsIVMSite q` itself is not invariant under raw representative changes.
* The *shell norm* uses the centered, sum-zero representative of an IVM
  class, distinct from its minimum-zero canonical representative.
  Shell `k` means `shellNorm4 q = 8k`. Scan-box completeness and exact
  enumeration membership are universal theorems below. No walk/reachability
  equivalence or universal shell cardinality is proved here.
* Cardinality instances for shells 1–4 use kernel reduction; shells 5–8 use
  trusted native computation. `shellSites_card_target` is only a definition
  of the desired equality at a specified index, not a universal theorem.
-/

/-- The 12 canonical IVM neighbor moves: all distinct permutations of
`(2, 1, 1, 0)` (already normalized), in lexicographic order
(`src/quadmath/lattice/ivm_field.py:61-74`, `src/quadmath/optimize/discrete_variational.py:10-14`). -/
def neighborMoves : List Quadray :=
  [⟨0, 1, 1, 2⟩, ⟨0, 1, 2, 1⟩, ⟨0, 2, 1, 1⟩, ⟨1, 0, 1, 2⟩, ⟨1, 0, 2, 1⟩,
    ⟨1, 1, 0, 2⟩, ⟨1, 1, 2, 0⟩, ⟨1, 2, 0, 1⟩, ⟨1, 2, 1, 0⟩, ⟨2, 0, 1, 1⟩,
    ⟨2, 1, 0, 1⟩, ⟨2, 1, 1, 0⟩]

/-- The neighbor-move set has the twelve-around-one cardinality. -/
theorem neighborMoves_length : neighborMoves.length = 12 := rfl

/-- The 12 neighbor moves are pairwise distinct. -/
theorem neighborMoves_nodup : neighborMoves.Nodup := by decide

/-- Component sum `s = a + b + c + d`. -/
def ivmSum (q : Quadray) : Int := q.a + q.b + q.c + q.d

/-- Normalized quadray: non-negative components, minimum zero. -/
def IsNormalized (q : Quadray) : Prop :=
  0 ≤ q.a ∧ 0 ≤ q.b ∧ 0 ≤ q.c ∧ 0 ≤ q.d ∧
    (q.a = 0 ∨ q.b = 0 ∨ q.c = 0 ∨ q.d = 0)

/-- Canonical IVM site: normalized and component sum divisible by 4.
Use `IsIVMSite q.normalize` for raw-label class membership; see
`isIVMSite_normalize`. -/
def IsIVMSite (q : Quadray) : Prop :=
  IsNormalized q ∧ ivmSum q % 4 = 0

/-- Decidability of the normalized-quadray condition (computable). -/
instance instDecidableIsNormalized (q : Quadray) : Decidable (IsNormalized q) := by
  unfold IsNormalized
  infer_instance

/-- Decidability of IVM site membership (computable membership predicate). -/
instance instDecidableIsIVMSite (q : Quadray) : Decidable (IsIVMSite q) := by
  unfold IsIVMSite
  infer_instance

/-- Normalization shifts the component sum by `-4k`, a multiple of 4 — the
formal anchor for class-level membership. -/
theorem ivmSum_normalize (q : Quadray) :
    ivmSum q.normalize = ivmSum q - 4 * quadMin q := by
  simp only [ivmSum, Quadray.normalize]
  omega

/-- Computable check that every canonical neighbor move is itself an IVM
lattice site (each move is normalized with component sum `4`). -/
theorem neighborMoves_all_sites :
    (neighborMoves.all fun m => decide (IsIVMSite m)) = true := rfl

/-- `4 * N(q)` for the L1-centered shell norm `N(q) = Σ|q_i - s/4|` of
`src/quadmath/lattice/ivm_field.py:80-106`: on IVM sites, shell `k` is exactly
`shellNorm4 q = 8k`.  Defined for all quadrays (the Python reference raises
on non-sites instead). -/
def shellNorm4 (q : Quadray) : Nat :=
  (4 * q.a - ivmSum q).natAbs + (4 * q.b - ivmSum q).natAbs +
    (4 * q.c - ivmSum q).natAbs + (4 * q.d - ivmSum q).natAbs

/-- Component range `0, ..., n` as `Int`s (the scan-box bounds). -/
private def intRange (n : Nat) : List Int :=
  (List.range (n + 1)).map (fun (x : Nat) => (x : Int))

/-- All quadrays with components in `[0, n]`, lexicographic order — the scan
box used by `shell_sites` (`src/quadmath/lattice/ivm_field.py:143-146`). -/
def quadBox (n : Nat) : List Quadray :=
  (intRange n).flatMap fun a =>
    (intRange n).flatMap fun b =>
      (intRange n).flatMap fun c =>
        (intRange n).map fun d => ⟨a, b, c, d⟩

/-- All IVM lattice sites on shell `k`, in lexicographic order — computable
mirror of `shell_sites` (`src/quadmath/lattice/ivm_field.py:121-157`): scan the box
`[0, 2k]^4` (any shell-`k` site has components at most `2k` after
normalization), keep normalized quadrays whose component sum is divisible by
4 and whose L1-centered shell norm is `2k`, i.e. `shellNorm4 q = 8k`. -/
def shellSites (k : Nat) : List Quadray :=
  (quadBox (2 * k)).filter fun q =>
    decide (IsIVMSite q ∧ shellNorm4 q = 8 * k)

/-- Shell 0 is the lone central site (`src/quadmath/lattice/ivm_field.py:115-116`). -/
theorem shellSites_0_length : (shellSites 0).length = 1 := by decide

/-- Shell 1 is the twelve-around-one cuboctahedron
(`src/quadmath/lattice/omni_numbering.py:5-9`). -/
theorem shellSites_1_length : (shellSites 1).length = 12 := by decide

/-- Shell-2 cardinality is the cuboctahedral number `42`
(`src/quadmath/lattice/ivm_field.py:115-116`). -/
theorem shellSites_2_length : (shellSites 2).length = 42 := by decide

/-- **Scan-box completeness (proved).**  Every IVM site with shell norm `8k`
has all four components in `[0, 2k]`, so the `[0, 2k]^4` scan box used by
`shell_sites` (`src/quadmath/lattice/ivm_field.py:112-113`) and by `shellSites` loses no shell
sites.  Proof: normalization gives a zero component, whose centered
contribution is `|−s| = s` (with `s = ivmSum q ≥ 0`); every other component
`i` satisfies `4·q_i ≤ |4·q_i − s| + s ≤ 8k`. -/
theorem shellNorm4_component_bound (q : Quadray) (k : Nat)
    (hq : IsIVMSite q) (hn : shellNorm4 q = 8 * k) :
    0 ≤ q.a ∧ q.a ≤ 2 * k ∧ 0 ≤ q.b ∧ q.b ≤ 2 * k ∧
      0 ≤ q.c ∧ q.c ≤ 2 * k ∧ 0 ≤ q.d ∧ q.d ≤ 2 * k := by
  obtain ⟨ha, hb, hc, hd, hz⟩ := hq.1
  have hs : 0 ≤ ivmSum q := by
    simp only [ivmSum]
    omega
  simp only [shellNorm4] at hn
  rcases hz with h | h | h | h
  · exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩
  · exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩
  · exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩
  · exact ⟨by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩

/-- `intRange` membership: exactly the integers in `[0, n]`. -/
theorem mem_intRange_iff (n : Nat) (x : Int) :
    x ∈ intRange n ↔ 0 ≤ x ∧ x ≤ n := by
  simp only [intRange, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨y, hy, hxy⟩
    subst hxy
    constructor <;> omega
  · rintro ⟨h0, h1⟩
    exact ⟨x.toNat, by omega, by omega⟩

/-- Scan-box membership: a quadray with all components in `[0, n]` lies in
the scan box `quadBox n`. -/
private theorem mem_quadBox_of_bounds (n : Nat) (q : Quadray)
    (ha : 0 ≤ q.a) (ha' : q.a ≤ n) (hb : 0 ≤ q.b) (hb' : q.b ≤ n)
    (hc : 0 ≤ q.c) (hc' : q.c ≤ n) (hd : 0 ≤ q.d) (hd' : q.d ≤ n) :
    q ∈ quadBox n := by
  unfold quadBox
  rw [List.mem_flatMap]
  refine ⟨q.a, (mem_intRange_iff n q.a).mpr ⟨ha, ha'⟩, ?_⟩
  rw [List.mem_flatMap]
  refine ⟨q.b, (mem_intRange_iff n q.b).mpr ⟨hb, hb'⟩, ?_⟩
  rw [List.mem_flatMap]
  refine ⟨q.c, (mem_intRange_iff n q.c).mpr ⟨hc, hc'⟩, ?_⟩
  rw [List.mem_map]
  exact ⟨q.d, (mem_intRange_iff n q.d).mpr ⟨hd, hd'⟩, rfl⟩

/-- **Enumeration characterization (proved).**  `shellSites k` contains
exactly the IVM sites whose shell norm is `8k` — the frequency-`k` shell and
nothing else. -/
theorem mem_shellSites_iff (q : Quadray) (k : Nat) :
    q ∈ shellSites k ↔ IsIVMSite q ∧ shellNorm4 q = 8 * k := by
  unfold shellSites
  rw [List.mem_filter]
  constructor
  · rintro ⟨_, hd⟩
    exact of_decide_eq_true hd
  · rintro ⟨h1, h2⟩
    obtain ⟨b0, b1, b2, b3, b4, b5, b6, b7⟩ :=
      shellNorm4_component_bound q k h1 h2
    exact ⟨mem_quadBox_of_bounds (2 * k) q b0 b1 b2 b3 b4 b5 b6 b7,
      decide_eq_true ⟨h1, h2⟩⟩

/-- Site membership is a property of the projective class: the normalized
representative of `q` is an IVM site iff the component sum of `q` is
divisible by 4 (normalization shifts the sum by a multiple of 4,
`ivmSum_normalize`). -/
theorem isIVMSite_normalize (q : Quadray) :
    IsIVMSite q.normalize ↔ ivmSum q % 4 = 0 := by
  constructor
  · rintro ⟨_, hmod⟩
    have hsum := ivmSum_normalize q
    omega
  · intro h
    refine ⟨normalize_spec q, ?_⟩
    have hsum := ivmSum_normalize q
    omega

/-- The shell norm is a property of the projective class: shifting all four
components by the same `t` leaves it unchanged. -/
theorem shellNorm4_sameClass (t : Int) (q : Quadray) :
    shellNorm4 ⟨q.a + t, q.b + t, q.c + t, q.d + t⟩ = shellNorm4 q := by
  simp only [shellNorm4, ivmSum]
  omega

/-- The integer sum-zero gauge, available on classes with sum divisible by four.
Unlike `normalize`, this does not promise nonnegative components. -/
def ivmCentered (q : Quadray) : Quadray :=
  ⟨q.a - ivmSum q / 4, q.b - ivmSum q / 4,
    q.c - ivmSum q / 4, q.d - ivmSum q / 4⟩

theorem ivmCentered_sameClass (q : Quadray) : SameClass q (ivmCentered q) := by
  refine ⟨-(ivmSum q / 4), ?_, ?_, ?_, ?_⟩
  all_goals simp only [ivmCentered] <;> omega

/-- Divisibility is exactly what permits an integer sum-zero representative. -/
theorem ivmCentered_sum_zero (q : Quadray) (hq : ivmSum q % 4 = 0) :
    ivmSum (ivmCentered q) = 0 := by
  have hs : ivmSum q = 4 * (ivmSum q / 4) := by omega
  simp only [ivmSum, ivmCentered] at *
  omega

theorem shellNorm4_centered (q : Quadray) :
    shellNorm4 (ivmCentered q) = shellNorm4 q := by
  unfold ivmCentered
  simp only [Int.sub_eq_add_neg]
  exact shellNorm4_sameClass (-(ivmSum q / 4)) q

/-- Class-level IVM membership is invariant under arbitrary gauge changes. -/
theorem isIVMSite_normalize_congr {q r : Quadray} (h : SameClass q r) :
    IsIVMSite q.normalize ↔ IsIVMSite r.normalize := by
  rw [(normalize_eq_iff_sameClass q r).mpr h]

/-- Shell membership determines the shell index: the same site cannot lie on
two different shells. -/
theorem shellSites_unique (q : Quadray) {k k' : Nat}
    (h1 : q ∈ shellSites k) (h2 : q ∈ shellSites k') : k = k' := by
  have e1 := (mem_shellSites_iff q k).mp h1
  have e2 := (mem_shellSites_iff q k').mp h2
  obtain ⟨_, hn1⟩ := e1
  obtain ⟨_, hn2⟩ := e2
  omega

/- The universal positive-shell count remains unproved. The predicate below
states one indexed equality, not `∀ k ≥ 1, ...`. Shells 1–4 have proof terms
checked by kernel reduction (`decide`). Shells 5–8 also have theorem terms,
but the pinned Lean 4.35 `native_decide` implementation introduces a
per-instance axiom justified by compiled evaluation. This trust dependency
affects those declarations and their transitive users, not unrelated
theorems merely importing this module. `#eval` outputs are not proofs. -/

/-- The indexed shell-count target; no theorem establishes it for arbitrary `k ≥ 1`. -/
def shellSites_card_target (k : Nat) : Prop :=
  (shellSites k).length = 10 * k * k + 2

set_option maxRecDepth 1000000
theorem shellSites_card_target_1 : shellSites_card_target 1 := by
  unfold shellSites_card_target; decide
theorem shellSites_card_target_2 : shellSites_card_target 2 := by
  unfold shellSites_card_target; decide
theorem shellSites_card_target_3 : shellSites_card_target 3 := by
  unfold shellSites_card_target; decide
theorem shellSites_card_target_4 : shellSites_card_target 4 := by
  unfold shellSites_card_target; decide
/-- Trusted native-computation instances for shells 5–8; unlike shells 1–4,
these rely on the toolchain's per-instance native-evaluation axioms. -/
theorem shellSites_card_target_5 : shellSites_card_target 5 := by
  unfold shellSites_card_target; native_decide
theorem shellSites_card_target_6 : shellSites_card_target 6 := by
  unfold shellSites_card_target; native_decide
theorem shellSites_card_target_7 : shellSites_card_target 7 := by
  unfold shellSites_card_target; native_decide
theorem shellSites_card_target_8 : shellSites_card_target 8 := by
  unfold shellSites_card_target; native_decide

-- Smoke checks (printed at build time):
#eval decide (IsIVMSite ⟨2, 1, 1, 0⟩)  -- true: neighbor-move class
#eval decide (IsIVMSite ⟨1, 0, 0, 0⟩)  -- false: tetrahedral void direction
#eval (List.range 8).map (fun k => (shellSites (k + 1)).length)  -- [12, 42, 92, 162, 252, 362, 492, 642]
#eval (List.range 8).map (fun k => 10 * (k + 1) * (k + 1) + 2)    -- same