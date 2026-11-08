# Lean Formalization: Integer Quotients, Metric/Volume Bridges and Proof Limits

The root [`lean/`](../../lean/README.md) tree contains handwritten core-only
Lean 4 mathematical models. It is not extraction from, or verification of, the
Python library, sibling implementations or floating simulations. This guide
maps declarations and their scope; it does not report a fresh Lean build.

## Build

From the repository root, when Lean verification is assigned:

```bash
(cd lean && lake build)
```

`lean/lean-toolchain` pins `leanprover/lean4:v4.35.0`. `lakefile.toml` defines
`Quadlean` as the default target; `Quadlean.lean` imports four model modules.
There is no Mathlib dependency. Printed `#eval` examples during compilation are
not theorem assertions. Integration must elaborate current sources before a
new revision is described as machine-verified.

## Module map

| Source | Mathematical declarations and scope |
|---|---|
| [`Quadray.lean`](../../lean/QuadMath/Quadray.lean) | Integer four-tuples, `SameClass`, actual `PhysicalQuadray` quotient, unique minimum-zero normalization, three-coordinate projection and independently gauged tetra-volume invariance |
| [`Distance.lean`](../../lean/QuadMath/Distance.lean) | Universal scale-one integer squared-distance identity, separation modulo common offset, signed XYZ determinant/native-volume bridge and shell radial lower bounds |
| [`IVM.lean`](../../lean/QuadMath/IVM.lean) | Canonical site predicate, centered gauge/shell norm, bounded enumeration completeness/uniqueness, neighbor inventory and finite shell-count declarations |
| [`Lattice.lean`](../../lean/QuadMath/Lattice.lean) | Numbering/cumulative formula definitions and explicitly indexed links to enumerated shells |

## Exact geometry scope

`PhysicalQuadray` is the integer common-offset quotient, not an identification
of separate raw game slots. `dist2_eq_zero_iff` separates physical classes;
it does not turn raw four-tuples into four independent Euclidean axes.
The squared-distance results are not a Lean `MetricSpace` structure or a
square-root metric/ordinary triangle-inequality formalization.

Native volume is rational `abs(tetraDet)/4`; a primitive tetrahedron can have
volume `1/4`. `tetraVolume_xyz_bridge` uses **scale-one XYZ**: native volume is
`3/8` of XYZ volume. QuadCraft's swapped/rescaled embedding is not the same
convention and its irrational scaling is not formalized here. See the canonical
[methods](../../quadmath/markdown/03_quadray_methods.md),
[conversions](../../quadmath/markdown/14_conversions_spec.md) and
[equations](../../quadmath/markdown/08_equations_appendix.md).

The shell lower bounds concern centered-norm enumeration. No Lean theorem here
identifies that enumeration with Python's BFS implementation, proves a general
Bareiss implementation correct, or verifies graph transport/harmonic fitting.

## Finite computation and open obligations

- `shellSites_card_target` is a proposition-valued definition, **not** a proof
  of the universal positive-shell formula.
- Explicit shells 1–4 use `decide` proof terms checked by kernel reduction.
  Shells 5–8 use `native_decide`; the pinned implementation introduces
  per-instance compiled-evaluation trust dependencies. These do not infect
  unrelated declarations merely because a module is imported.
- Formula definitions and printed matching values do not prove universal
  enumeration counts. There is no arbitrary-positive-shell count or universal
  cumulative-ball-cardinality theorem in this model.
- Zero `sorry` is not an axiom audit. Review `#print axioms` for the actual
  declaration in the pinned toolchain; no new axiom audit is claimed here.

The fuller statement/trust inventory is maintained in
[`lean/README.md`](../../lean/README.md), not duplicated as a dated green-build
receipt in this operational guide. Return to [documentation](../README.md) or
[development operations](../development/README.md).
