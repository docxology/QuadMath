# Agent guide — QuadMath

## Layout

- Paper source of truth: `quadmath/markdown/` (00–09 numbered sections).
- Build: `quadmath/scripts/render_pdf.sh`; clean: `quadmath/scripts/clean_output.sh`;
  everything via `run_all.sh`.
- `src/` and `tests/` hold the supporting code; `pyproject.toml`/`uv.lock` pin the environment.

## Conventions observed

- Sections are numbered Markdown files assembled in order by the render script.
- `WORKFLOW.md` and `ARCHITECTURE.md` at the root describe the build pipeline and layout.
- Generated outputs land under `quadmath/output/` and are regeneratable — do not hand-edit.

## Docs maintenance

- `docs/` here is short by design: entry (`README.md`), this guide, and
  `docs/manuscript/MANUSCRIPT_STATUS.md`. Keep claims traceable to
  `README.md` / `pyproject.toml`; the DOI and title live in `README.md`.
