# manuscript/

Template-layout manuscript tree for the shared docxology render pipeline, ported
from `../../quadmath/markdown/`. Status: `MANUSCRIPT_STATUS.md`.
Re-port after source edits with: `uv run python docs/manuscript/port_from_source.py`.

## Layout

| File | Origin | Role |
|---|---|---|
| `config.yaml` | new (metadata from root `README.md`) | Paper metadata (title, authors, DOI, keywords, license, render toggles) — consumed by `stage_03_render` / `infrastructure.rendering` |
| `preamble.md` | `00_preamble.md` + closing fence | LaTeX preamble; the fenced `latex` block is extracted by `infrastructure.rendering._pdf_latex_helpers.extract_preamble` |
| `references.bib` | built from `07_resources.md` + in-text citations | BibTeX database; `*.bib` files at manuscript top level are auto-discovered (`_bibliography.resolve_bibliography`) |
| `01_introduction.md` … `06_discussion.md` | same-name sources, verbatim | Main sections |
| `08_equations_appendix.md`, `09_free_energy_active_inference.md` | same-name sources, verbatim except 3 repointed links (see below) | Appendices A–B |
| `11_ivm_field_learning.md`, `12_ivm_dynamics.md`, `13_lattice_tooling.md` | same-name sources, verbatim | IVM lattice sections (field learning, dynamics, tooling) |
| `14_conversions_spec.md`, `15_learning_evaluation.md`, `16_lattice_gallery.md` | same-name sources, verbatim | Conversion/specification, learning & evaluation, and the visualization gallery (embeds `figures/vis_gallery_*.png`) |
| `98_symbols_glossary.md` | `10_symbols_glossary.md`, verbatim | Template glossary bucket (`98_*.md`); snapshot of the auto-generated source |
| `99_resources.md` | `07_resources.md`, verbatim | Template references bucket (`99_*.md`, rendered last) |
| `figures/` | copies of `quadmath/output/figures/*.png` (21 files) | Figure images referenced by the sections |
| `port_from_source.py` | new | Reproducible re-port script (copies sections/figures, applies the rewrites, closes the preamble fence, verifies links) |
| `MANUSCRIPT_STATUS.md` | — | Fleet status tracker |

## Path rewrites applied during the port

- Images: `../output/figures/X` → `figures/X` (self-contained: `quadmath/output/`
  is disposable — `clean_output.sh` wipes it)
- LICENSE: `../LICENSE` → `../../LICENSE` (repo root)
- Section links: `07_resources.md` → `99_resources.md`, `10_symbols_glossary.md` → `98_symbols_glossary.md`
- Code-anchor repoint (upstream link defect): section 09 links
  `{#code:free_energy}`, `{#code:fisher_information_matrix}`,
  `{#code:natural_gradient_step}` at `08_equations_appendix.md`, which defines
  only `{#eq:*}` anchors; the targets live in `03_quadray_methods.md`, so the
  port repoints the three links there.
- Preamble: the upstream `00_preamble.md` opens a ` ```latex ` fence that is
  **never closed** (upstream defect: the local `render_pdf.sh` sed extractor
  tolerates this; the template `extract_preamble` regex does not and would
  silently drop the whole preamble). The port appends the closing fence.

Everything else (equations, labels, anchors, captions, tables, code blocks) is
ported verbatim. Sections `11`–`16` port under their own numbers.

## Relationship to the source

`quadmath/markdown/` remains the **authoritative** editing source; this tree is
the template-layout projection. Content changes happen in the source and are
re-ported with `port_from_source.py`. `98_symbols_glossary.md` is generated
upstream by `quadmath/scripts/generate_glossary.py` — regenerate there, then
re-port.
