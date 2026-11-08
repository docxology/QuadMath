# Lean Documentation Agent Guide: Declaration Scope and Trust

This folder documents the root `lean/` model; it does not own Lean source edits.
Read [scope](README.md), [the authoritative Lean inventory](../../lean/README.md)
and [docs guidance](../AGENTS.md).

## Required distinctions

- Handwritten integer models are not verification of Python or floating solvers.
- Physical common-offset quotient and raw label equality are different concepts.
- Native rational volume, scale-one XYZ volume and swapped/rescaled suite
  embeddings must not be conflated.
- A theorem source is not a fresh successful elaboration receipt. Do not state
  a current `lake build` result without integration evidence.
- `decide` proofs have kernel-checked proof terms; `native_decide` has the
  pinned toolchain's disclosed compiled-evaluation trust dependencies.
- `shellSites_card_target` is a proposition-valued definition, not a universal
  shell-count theorem. Finite counts and printed `#eval` values do not close it.
- Zero `sorry` is not a declaration-level axiom audit.

## Maintenance

Read actual declarations and the toolchain pin before changing the inventory.
Link canonical mathematics and the fuller root Lean README rather than copying
large proof-status snapshots. Preserve existing anchors and report cross-owner
heading changes. Do not move sources/artifacts into docs or hand-edit manuscript
projections. During delegated work skip builds/tests/lint/formatters; the
integration owner elaborates and audits current declarations separately.
