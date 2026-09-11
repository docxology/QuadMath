# lean/ — Lean formalization (in progress)

The QuadMath repository carries a **Lean 4 formalization** of the quadray
foundations — integer quadray coordinates, normalization, tetravolume
identities, and the lattice structures used throughout the analytical review.

**Status: being built in parallel by another agent.** The `lean/` directory at
the repository root is the target location and does not exist on disk yet, so
this page deliberately contains no links to `lean/` files. When the
formalization lands, this page becomes the documentation entry point for it.

## What to expect (per the fleet brief)

- A `lean/` tree at the repository root with its own `lakefile`/toolchain pins,
  independent of the Python `pyproject.toml` environment.
- Formal statements corresponding to the analytical content in the manuscript:
  quadray normalization, the integer-volume quantization theorem (unit IVM
  tetrahedron ⇒ tetravolume 1; lattice tetrahedra on the 1/4-grain grid), and
  the Ace 5×5 / Cayley–Menger agreement results (see
  [the equations appendix](../manuscript/08_equations_appendix.md)).
- Relationship to the existing formalism sources: `src/symbolic.py`
  (SymPy implementations) and the equations in `quadmath/markdown/` — Lean
  proofs are a third witness alongside the Python implementations and the
  cross-language validation ecosystem documented in
  [the resources section](../manuscript/99_resources.md).

## Verification checklist once `lean/` lands

1. `lean/` builds (`lake build`) in CI or locally with pinned toolchain
2. Repository README/AGENTS updated to describe the Lean surface
3. This page re-written from "intended" to "documented", with working links
