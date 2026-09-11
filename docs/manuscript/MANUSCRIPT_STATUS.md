# MANUSCRIPT_STATUS

## Repo type

Published analytical paper repository. The paper is maintained under
`quadmath/markdown/` as numbered section files with render/clean scripts,
plus a versioned PDF (`QuadMath_v1_DAF_08-16-2025.pdf`, not kept in git — the
canonical copy is on Zenodo via the DOI above; regenerate locally with
`bash quadmath/scripts/render_pdf.sh`) and a Zenodo DOI.

## Current layout (updated 2026-09-10)

The manuscript now exists in **two coordinated trees**:

1. **Source tree** — `quadmath/markdown/` (**authoritative for content
   edits**): `00_preamble.md` (LaTeX preamble) and numbered sections
   `01_introduction.md` … `12_ivm_dynamics.md`, built by
   `quadmath/scripts/render_pdf.sh`. Future content edits happen here.

2. **Template-layout tree** — `docs/manuscript/` (this directory; **generated
   projection — do not hand-edit section content**, re-port with
   `port_from_source.py`): the same content ported into the shared
   docxology/template render layout:

   - `config.yaml` — title/authors/DOI/keywords/license/render toggles
     (metadata sourced from root `README.md` and the manuscript itself)
   - `preamble.md` — LaTeX preamble (port of `00_preamble.md` + closing fence)
   - `references.bib` — BibTeX built from `07_resources.md` plus in-text
     citations (Conway & Sloane, Coxeter, methods references, self-citation)
   - numbered sections `01…06`, `08`, `09`, `12` (ports), plus the two
     template buckets: `98_symbols_glossary.md` (glossary; from
     `10_symbols_glossary.md`) and `99_resources.md` (references; from
     `07_resources.md`)
   - `figures/` — copies of the 17 PNGs referenced by the sections

   Path rewrites applied by the port: images `../output/figures/X` →
   `figures/X`; `../LICENSE` → `../../LICENSE`; section links renumbered to
   the 98/99 buckets. Equations, labels, anchors, and captions are otherwise
   verbatim (one deliberate deviation, see Known caveats).

## Render parity

- **Target pipeline**: template `scripts/pipeline/stage_03_render.py` →
  `infrastructure.rendering`. Discovery contract met by this tree:
  top-level numbered `NN_*.md` sections, `preamble.md` (```latex block),
  `config.yaml`, top-level `*.bib`.
- **Local pipeline**: `quadmath/scripts/render_pdf.sh` remains the working
  local render path (unchanged, builds from `quadmath/markdown/`).
- The published v1 PDF stays on Zenodo (DOI 10.5281/zenodo.16887791);
  neither pipeline is wired into CI here.

## Evidence checked

- `README.md` (title, DOI 10.5281/zenodo.16887791, render/clean commands)
- `quadmath/markdown/` (00_preamble.md through 12_ivm_dynamics.md)
- `run_all.sh`, `quadmath/scripts/render_pdf.sh`, `quadmath/scripts/clean_output.sh`
- Template contracts: `template/docs/RUN_GUIDE.md` (stage map),
  `infrastructure/rendering/manuscript_discovery.py` (section discovery +
  `EXCLUDE_NAMES`), `_pdf_combined_preamble.py` (preamble injection),
  `_bibliography.py` (bib discovery + key-uniqueness gate),
  `projects/templates/template_active_inference/manuscript/config.yaml` (config shape)

## Known caveats

- **Deliberate content divergence (code-anchor repoint)**: section 09
  upstream links `{#code:*}` anchors at `08_equations_appendix.md`, but 08
  defines only `{#eq:*}` anchors — the `{#code:*}` targets live in
  `03_quadray_methods.md` (an upstream link defect). The port repoints the
  three affected links (`free_energy`, `fisher_information_matrix`,
  `natural_gradient_step`) to `03#code:*`. Everything else is verbatim.
- **Upstream defect**: `quadmath/markdown/00_preamble.md` opens a
  ` ```latex ` fence that is never closed. The local `render_pdf.sh` sed
  extractor tolerates this; the template `extract_preamble` does not
  (silently drops the whole preamble). `preamble.md` here ships with the
  closing fence appended; fix upstream when touching the source tree.
- `MANUSCRIPT_STATUS.md` (this file) is not in the template `EXCLUDE_NAMES`
  set, so a future `stage_03_render` run pointed at this directory would sweep
  it into the "other" section bucket. Relocate it or extend the exclusion when
  wiring the render stage.
- `figures/` holds copies; regenerate via `quadmath/scripts/` into
  `quadmath/output/figures/` and re-port — never hand-edit either tree's
  images (`quadmath/output/` is disposable: `clean_output.sh` wipes it).
- `98_symbols_glossary.md` is a snapshot of the auto-generated
  `10_symbols_glossary.md`; regenerate upstream and re-port.
- Coverage note: `11_equations_appendix`-style gaps do not exist, but section
  numbering skips `07` (renamed to `99_resources.md`) and `10` (renamed to
  `98_symbols_glossary.md`) by template bucket convention; `12_ivm_dynamics.md`
  ports under its own number.
- Link integrity for this tree: `uv run python docs/development/check_links.py`
  (validates relative targets **and** `{#id}` fragments);
  `uv run python quadmath/scripts/validate_markdown.py --strict` covers the
  source tree; re-port with `uv run python docs/manuscript/port_from_source.py`.
