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
  and has `10k² + 2` sites for `k ≥ 1` (cuboctahedral numbers).
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

/-- **Shell-count theorem (statement, not proved here).**  For every `k ≥ 1`
the IVM shell of frequency `k` contains exactly the cuboctahedral number
`10k² + 2` sites (`src/ivm_field.py:97-98`; `src/omni_numbering.py:4-9`,
machine-checked there through frequency 6, and independently cross-checked
through `k = 6` while authoring this file). -/
theorem shellSites_card (k : Nat) (hk : 1 ≤ k) :
    (shellSites k).length = 10 * k * k + 2 := by
  sorry

-- Smoke checks (printed at build time):
#eval decide (IsIVMSite ⟨2, 1, 1, 0⟩)  -- true: neighbor-move class
#eval decide (IsIVMSite ⟨1, 0, 0, 0⟩)  -- false: tetrahedral void direction
#eval (shellSites 3).length            -- 92
#eval (shellSites 4).length            -- 162