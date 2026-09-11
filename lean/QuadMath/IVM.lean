import QuadMath.Quadray
/-!
# The IVM lattice (isotropic vector matrix, FCC/CCP) in quadray coordinates

Lean 4 formalization of the IVM lattice as QuadMath defines it.  Mirrors
`src/ivm_field.py` (membership test, L1-centered shell norm, shell
enumeration) and `src/discrete_variational.py` (`neighbor_moves_ivm`).

Conventions (see `src/ivm_field.py` and the manuscript sections):

* A *normalized* quadray has non-negative components and minimum zero — the
  canonical representative of its projective class (`src/quadray.py:29-36`).
* An *IVM lattice site* is a normalized quadray whose component sum is
  divisible by 4 (`src/ivm_field.py:81-89`).  Normalization shifts the sum by
  a multiple of 4, so this is a property of the projective class; the
  `(1,0,0,0)`-type void directions (sum `1` mod 4) are *not* sites.
* The *shell norm* of an IVM site is the L1 magnitude of its sum-zero
  representative, `N(q) = Σ|q_i - s/4|` with `s = a+b+c+d`
  (`src/ivm_field.py:92-118`).  Shell `k` collects the sites with `N(q) = 2k`
  and has `10k² + 2` sites for `k ≥ 1` (cuboctahedral numbers) — the universal
  count is stated as `shellSites_card_target` and machine-checked through
  shell 8, while the scan-box completeness bound (`[0, 2k]^4`) is proved
  (`shellNorm4_component_bound`), as is the exact enumeration
  characterization (`mem_shellSites_iff`).
-/

/-- The 12 canonical IVM neighbor moves: all distinct permutations of
`(2, 1, 1, 0)` (already normalized), in lexicographic order
(`src/ivm_field.py:61-74`, `src/discrete_variational.py:10-14`). -/
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

/-- IVM lattice site: a normalized quadray whose component sum is divisible
by 4 (`src/ivm_field.py:81-89`).  Decidable, hence a computable membership
predicate; well-defined on projective classes because normalization shifts
the sum by a multiple of 4 (`ivmSum_normalize` below). -/
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
`src/ivm_field.py:92-118`: on IVM sites, shell `k` is exactly
`shellNorm4 q = 8k`.  Defined for all quadrays (the Python reference raises
on non-sites instead). -/
def shellNorm4 (q : Quadray) : Nat :=
  (4 * q.a - ivmSum q).natAbs + (4 * q.b - ivmSum q).natAbs +
    (4 * q.c - ivmSum q).natAbs + (4 * q.d - ivmSum q).natAbs

/-- Component range `0, ..., n` as `Int`s (the scan-box bounds). -/
private def intRange (n : Nat) : List Int :=
  (List.range (n + 1)).map (fun (x : Nat) => (x : Int))

/-- All quadrays with components in `[0, n]`, lexicographic order — the scan
box used by `shell_sites` (`src/ivm_field.py:143-146`). -/
def quadBox (n : Nat) : List Quadray :=
  (intRange n).flatMap fun a =>
    (intRange n).flatMap fun b =>
      (intRange n).flatMap fun c =>
        (intRange n).map fun d => ⟨a, b, c, d⟩

/-- All IVM lattice sites on shell `k`, in lexicographic order — computable
mirror of `shell_sites` (`src/ivm_field.py:121-157`): scan the box
`[0, 2k]^4` (any shell-`k` site has components at most `2k` after
normalization), keep normalized quadrays whose component sum is divisible by
4 and whose L1-centered shell norm is `2k`, i.e. `shellNorm4 q = 8k`. -/
def shellSites (k : Nat) : List Quadray :=
  (quadBox (2 * k)).filter fun q =>
    decide (IsIVMSite q ∧ shellNorm4 q = 8 * k)

/-- Shell 0 is the lone central site (`src/ivm_field.py:127-128`). -/
theorem shellSites_0_length : (shellSites 0).length = 1 := by decide

/-- Shell 1 is the twelve-around-one cuboctahedron
(`src/omni_numbering.py:5-9`). -/
theorem shellSites_1_length : (shellSites 1).length = 12 := by decide

/-- Shell-2 cardinality is the cuboctahedral number `42`
(`src/ivm_field.py:127-128`). -/
theorem shellSites_2_length : (shellSites 2).length = 42 := by decide

/-- **Scan-box completeness (proved).**  Every IVM site with shell norm `8k`
has all four components in `[0, 2k]`, so the `[0, 2k]^4` scan box used by
`shell_sites` (`src/ivm_field.py:124-125`) and by `shellSites` loses no shell
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

/-- Shell membership determines the shell index: the same site cannot lie on
two different shells. -/
theorem shellSites_unique (q : Quadray) {k k' : Nat}
    (h1 : q ∈ shellSites k) (h2 : q ∈ shellSites k') : k = k' := by
  have e1 := (mem_shellSites_iff q k).mp h1
  have e2 := (mem_shellSites_iff q k').mp h2
  obtain ⟨_, hn1⟩ := e1
  obtain ⟨_, hn2⟩ := e2
  omega

/- **The shell-count theorem, stated but not proved in this tree.**  For
every `k ≥ 1` the IVM shell of frequency `k` contains exactly the
cuboctahedral number `10k² + 2` sites (`src/ivm_field.py:97-98`;
`src/omni_numbering.py:4-9`).  This formalization keeps the universal claim
as a proposition and **machine-checks** the instances through shell 8
(see `lean/README.md`, "What is proved vs. machine-checked"):
shells 1-4 by kernel-evaluated `decide`, shells 5-8 by `native_decide`
(the virtual machine evaluates the same closed expression; the kernel
trusts its result through `Lean.ofReduceBool`, so these instances are
machine-checked, not kernel-checked). -/

/-- The universal shell-count proposition itself. -/
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
/-- Machine-checked (VM-evaluated) instances for shells 5-8; the kernel-only
`decide` variants above cover shells 1-4 (larger boxes exceed practical
kernel-evaluation cost). -/
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