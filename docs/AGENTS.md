# Agent guide — QuadMath docs/

## Layout

- `manuscript/` — template-layout manuscript projection of
  `quadmath/markdown/` (`config.yaml`, `preamble.md`, `references.bib`,
  sections `01…99`, `figures/`). `quadmath/markdown/` remains the editing
  source of truth; content changes happen there and are re-ported.
- `development/` — test/coverage workflow (`README.md`) and the docs-tree
  link checker (`check_links.py`).
- `learning/` — DRAFT docs for the in-progress IVM field/dynamics modules;
  intent-only until `src/ivm_field.py` / `src/ivm_dynamics.py` exist.
- `lean/` — pointer page for the Lean formalization at `lean/` (parallel
  work); no links to `lean/` files until they exist.
- `overview.md` — repository tour.

## Conventions observed

- Every subfolder ships `README.md` (human-facing) + `AGENTS.md` (agent
  notes), matching the repository's docs contract.
- Manuscript sections in `manuscript/` follow the template discovery buckets:
  numbered `NN_*.md`, glossary `98_*`, references `99_*` (always last),
  `preamble.md` with a ` ```latex ` block, top-level `*.bib`.
- Generated outputs (`quadmath/output/`, manuscript `figures/` copies) are
  regeneratable — never hand-edit.

## Docs maintenance

- After any `docs/` edit, run `uv run python docs/development/check_links.py`
  — every relative link/image in these files must resolve on disk.
- Manuscript source-tree invariants are checked separately with
  `uv run python quadmath/scripts/validate_markdown.py`.
- Keep claims traceable to root `README.md` / `pyproject.toml`; the DOI and
  title live in the root `README.md` and `docs/manuscript/config.yaml`.
- `manuscript/MANUSCRIPT_STATUS.md` tracks render parity between the two
  manuscript trees; update it when either tree changes structurally.
