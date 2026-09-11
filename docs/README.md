# QuadMath — documentation

Analytical review of Quadray coordinates (4D): integer volume quantization,
optimization on tetrahedral lattices, and information geometry. Published with
DOI [10.5281/zenodo.16887791](https://zenodo.org/records/16887791).

## Repository map

| Path | Contents |
|---|---|
| `quadmath/markdown/` | Numbered manuscript sections (`00_preamble.md` + `01_introduction.md` … `18_stats_gallery.md`, 19 files) — editing source of truth |
| `quadmath/scripts/` | Render/clean scripts plus figure, data, glossary, GIF, and validation generators (19 Python + 2 shell scripts) |
| `src/quadmath/`, `tests/` | Factored source package (`core`, `lattice`, `optimize`, `inference`, `stats`, `learn`, `viz`, `validate`, `tools`) and mirrored test suite |
| `docs/manuscript/` | Template-layout manuscript projection for the shared docxology render pipeline (`config.yaml`, `preamble.md`, `references.bib`, sections `01…99`, `figures/`) — see its `README.md` and `docs/MANUSCRIPT_STATUS.md` |
| `docs/development/` | Test/coverage workflow and the docs link checker (`check_links.py`) |
| `docs/learning/` | Docs for the landed IVM learning surface (`src/quadmath/lattice/ivm_field.py`, `src/quadmath/lattice/ivm_dynamics.py`, `src/quadmath/learn/learning_eval.py`, `src/quadmath/viz/vis_lattice.py`) |
| `docs/lean/` | Lean 4 formalization docs (`lean/` landed — core Lean 4, one open shell-count theorem) |
| `docs/overview.md` | Repository tour |
| `QuadMath_v1_DAF_08-16-2025.pdf` | Published version-1 PDF (not git-tracked; regenerate via `render_pdf.sh`, canonical copy on Zenodo) |
| `run_all.sh` | Full build entry point |

## How to run

From the repository root (commands from `README.md`):

```bash
uv sync
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run -m pytest -q
bash quadmath/scripts/render_pdf.sh   # generate figures, LaTeX, PDFs
bash quadmath/scripts/clean_output.sh # remove generated outputs
```

## Docs checks

```bash
uv run python quadmath/scripts/validate_markdown.py  # manuscript source tree
uv run python docs/development/check_links.py        # this docs tree
```

## Status

Published (v1, Zenodo). The manuscript now has two coordinated trees:
`quadmath/markdown/` (source) and `docs/manuscript/` (template-layout
projection for the shared render pipeline). See
`docs/MANUSCRIPT_STATUS.md` for the layout, render parity, and
caveats.
