# Agent guide — QuadMath docs/

## Layout

- `manuscript/` — template-layout manuscript projection of
  `quadmath/markdown/` (`config.yaml`, `preamble.md`, `references.bib`,
  sections `01…99`, `figures/`). `quadmath/markdown/` remains the editing
  source of truth; content changes happen there and are re-ported.
- `development/` — test/coverage workflow (`README.md`) and the docs-tree
  link checker (`check_links.py`).
- `learning/` — docs for the landed IVM learning surface (`src/ivm_field.py`,
  `src/ivm_dynamics.py`, `src/learning_eval.py`, `src/vis_lattice.py`).
- `analysis/` — docs for the landed benchmarks/statistics surface
  (`src/benchmarks.py`, `src/statistics.py`, `src/vis_stats.py`).
- `lean/` — docs for the landed Lean 4 formalization at `lean/` (build
  command, module map, proof status: zero sorries; universal shell count
  machine-checked through shell 8, universal distance identity proved).
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
