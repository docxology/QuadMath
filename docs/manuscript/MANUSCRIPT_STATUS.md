# MANUSCRIPT_STATUS

## Repo type

Published analytical paper repository. The paper content lives under
`quadmath/markdown/` as numbered section files with render/clean scripts,
plus a versioned PDF (`QuadMath_v1_DAF_08-16-2025.pdf`, not kept in git — the
canonical copy is on Zenodo via the DOI above; regenerate locally with
`bash quadmath/scripts/render_pdf.sh`) and a Zenodo DOI.

## Evidence checked

- `README.md` (title, DOI 10.5281/zenodo.16887791, render/clean commands)
- `quadmath/markdown/` (00_preamble.md through 09_free_energy_active_inference.md)
- `run_all.sh`, `quadmath/scripts/render_pdf.sh`, `quadmath/scripts/clean_output.sh`

## Why no canonical `manuscript/` tree exists today

The paper already exists as a complete sectioned manuscript under
`quadmath/markdown/` with its own working build path. Creating parallel
template-format section stubs would duplicate that content.

## What would trigger creating one

Adopting the shared template render pipeline: port the `quadmath/markdown/`
sections into `manuscript/00..99` with `config.yaml`, `preamble.md`, and
`references.bib`, then use `stage_03_render.py` instead of
`quadmath/scripts/render_pdf.sh`.
