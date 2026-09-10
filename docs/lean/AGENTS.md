# lean/ — Agent Notes

Pointer/documentation page for the Lean formalization under `lean/` at the
repository root (parallel fleet work, not on disk yet).

## Rules

- Until `lean/` exists, keep every reference to it as inline code — no
  markdown links to nonexistent files (docs link checker enforces this).
- When the formalization lands, document here: build command, toolchain pin,
  statement inventory mapped to manuscript equations, and the proof-status
  policy (what is proven vs `sorry`-ed).
- Do not move Lean sources or build artifacts into `docs/`; this folder is
  documentation only.
