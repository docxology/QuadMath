# Manuscript sources, generated projection and rendering evidence

## Edit canonical chapters, then regenerate the template projection

| Tree | Role | Update path |
|---|---|---|
| [`quadmath/markdown/`](../quadmath/markdown/01_introduction.md) | Authoritative preamble, chapters, equations, captions and API glossary | Edit authored chapters; regenerate the glossary |
| [`docs/manuscript/`](manuscript/config.yaml) | Generated template-layout projection and copied figures | Run [`port_from_source.py`](manuscript/port_from_source.py); do not hand-edit projected chapters |
| `quadmath/output/` or `QUADMATH_OUTPUT_DIR` | Generated Python figure/data artifacts | Run the registered figure generators |
| `quadmath/output/pdf/`, `quadmath/output/tex/` | Local shell renderer's publications | Run [`render_pdf.sh`](../quadmath/scripts/render_pdf.sh) with its real media/TeX prerequisites |

The source has preamble `00` and chapters `01`–`18`. The projection maps chapter
`07_resources.md` to `99_resources.md` and `10_symbols_glossary.md` to
`98_symbols_glossary.md`; other numbered chapters retain their filenames.
The porter rewrites figure/section/resource links and copies available figures.
Its required manuscript-figure inventory is a render-input contract, not a
claim that every generated PNG is cited in the paper.

## Regenerate Python artifacts without overwriting local output

Run from the repository root, with the scientific and media prerequisites in
the [build guide](../quadmath/README.md):

```bash
OUT="$(mktemp -d)"
export MPLBACKEND=Agg
export QUADMATH_OUTPUT_DIR="$OUT"
export UV_PROJECT_ENVIRONMENT="$OUT/environment"
uv run --python 3.13 --group dev python quadmath/scripts/make_all_figures.py
uv run --python 3.13 --group dev python quadmath/scripts/generate_glossary.py
uv run --python 3.13 --group dev python quadmath/scripts/validate_markdown.py --strict
uv run --python 3.13 --group dev python docs/manuscript/port_from_source.py --figures-dir "$OUT/figures"
uv run --python 3.13 --group dev python docs/development/check_links.py
```

The current figure registry includes nineteen generators. The manifest contains
deduplicated **emitted** paths relative to the configured output root, including
the graphical abstract's PNG and SVG. It is not an inventory of every file
ever written. Resolve manifest entries against that output root.

Porting updates tracked projection files: preserve unrelated local manuscript
edits before using it. Generated default output must not be casually deleted or
committed merely because it differs from a fresh isolated render.

## Keep mathematical captions aligned with actual simulations

- The Quadray abstract draws a rank-three quotient in physical scale-one XYZ,
  actual basis-ray lengths and a complete twelve-around-one contact cluster.
- Regression figures distinguish actual fitted/generating coefficients, loss
  gradients and their Gram matrices. They do not simulate a biological agent
  or establish geodesic motion.
- Categorical free energy is sampled on its declared feasible simplex face;
  the displayed minimum is grid-restricted. Ideal entropy curvature is labeled
  separately from the library's pseudocount-smoothed objective.
- Integer Nelder–Mead snapshots include the final recorded frame. The marked
  best vertex is measured from that frame, not hard-coded at the origin.

## Separate source checks, PDF builds and archived publications

The local shell pipeline regenerates figures/glossary, validates the canonical
chapters and builds per-chapter plus combined PDFs through Pandoc/XeLaTeX.
`--skip-figures` skips figure generation, not glossary or source validation.
Python output helpers honor `QUADMATH_OUTPUT_DIR`; the shell's PDF/TeX
publication directories remain fixed under the checkout's default output.
Use a separate checkout for a fully isolated PDF build.

The [Linux verification workflow](../.github/workflows/verify.yml) provisions
the real font/TeX/Pandoc/video toolchain and exercises the manuscript build.
The local macOS build previously failed to load the required DejaVu fonts;
source validation or an alternate-font diagnostic is not evidence that this
production-font build succeeded.

The historical Zenodo deposit, DOI `10.5281/zenodo.16887791`, and immutable
GitHub release assets identify their own archived versions. Neither their page
counts nor old render/test statistics describe a changed `main` checkout.
Current release and verification boundaries are in the [changelog](../CHANGELOG.md)
and [suite guide](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md).
