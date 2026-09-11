# lean/ — Agent Notes

Documents the Lean 4 formalization at `lean/` (repository root), which has
**landed**: `cd lean && lake build` (toolchain `leanprover/lean4:v4.33.1`,
core Lean only, no Mathlib). Build exits 0 with **zero sorries**.

## Rules

- Keep this page's proof-status inventory in lock-step with
  `lean/QuadMath/*.lean`; the Lean sources are owned by their implementing
  agent — propose changes there, don't patch `lean/` from here.
- The former single `sorry` (`shellSites_card`) is closed: the universal
  claim is restated as `shellSites_card_target` and machine-checked through
  shell 8 (kernel `decide` k≤4; `native_decide` k∈{5..8} with disclosed
  per-instance axioms). Current state: **zero sorries**; any future `sorry`
  must be reintroduced with a documented line number on both pages here.
- Reference Lean sources by inline code paths (module/file names); this
  folder links only to files that resolve on disk (the docs link checker
  validates every relative target).
- Do not move Lean sources or build artifacts into `docs/`; this folder is
  documentation only.
- Statement→source mapping lives in [README.md](README.md) (module map and
  the manuscript cross-references).
