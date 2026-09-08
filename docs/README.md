# QuadMath — documentation

Analytical review of Quadray coordinates (4D): integer volume quantization,
optimization on tetrahedral lattices, and information geometry. Published with
DOI [10.5281/zenodo.16887791](https://zenodo.org/records/16887791).

## Repository map

| Path | Contents |
|---|---|
| `quadmath/markdown/` | Numbered manuscript sections (`00_preamble.md` … `10_symbols_glossary.md`, 11 files) |
| `quadmath/scripts/` | Render/clean scripts plus 14 figure, data, glossary, and validation generators (16 scripts) |
| `src/`, `tests/` | Supporting code and test suite |
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

## Status

Published (v1, Zenodo). See `docs/manuscript/MANUSCRIPT_STATUS.md` for why the
paper lives under `quadmath/markdown/` rather than a top-level `manuscript/` tree.
