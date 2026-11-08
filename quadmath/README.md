# QuadMath Manuscript Tools: Figures, Glossary, PDF and TeX

This checkout directory is not the runtime Python package. Reusable numerical
and rendering methods live in [`src/quadmath/`](../src/README.md); scripts here
select examples, call package methods and orchestrate manuscript conversion.

## Directory Structure

| Directory | Responsibility |
|---|---|
| `markdown/` | Canonical mathematical chapters, preamble and API-glossary source |
| `scripts/` | Figure/data recipes, glossary generation, Markdown validation and shell PDF tools |
| `output/figures/`, `output/data/` | Default Python-generated artifacts and relative-path figure manifest |
| `output/pdf/`, `output/tex/` | Fixed shell-renderer outputs |
| `output/latex_temp/` | Renderer intermediate files |

Outputs are generated and partly tracked; they may contain user changes.
Do not hand-edit or purge them as a routine preparation step.

## Python figure recipes with isolated outputs

Run from the repository root after `uv sync --python 3.13`:

```bash
export QUADMATH_OUTPUT_DIR="$(mktemp -d)/quadmath-results"
MPLBACKEND=Agg uv run --python 3.13 python quadmath/scripts/stats_gallery.py
MPLBACKEND=Agg uv run --python 3.13 python quadmath/scripts/graphical_abstract_quadray.py
```

The abstract script delegates to
`quadmath.viz.abstracts.render_quadray_abstract`, returning the canonical
`graphical_abstract_quadray.png` and a vector SVG companion. Its scale-one basis
and FCC 12-contact neighborhood use the declared embedding and physical units.
Other focused recipes include field/dynamics, lattice, learning, quaternion
and statistics galleries; consult the actual script for its inputs and outputs.

For assigned integrated generation:

```bash
MPLBACKEND=Agg uv run --python 3.13 python quadmath/scripts/make_all_figures.py
```

`FIGURE_SCRIPTS` is the explicit run order. The orchestrator uses the same Python
executable for children, collects recognized emitted paths under the selected
root and writes `data/figure_manifest.txt`. It does not run tests or establish
mathematical/visual correctness. Changing the output root does not rewrite
canonical manuscript image links.

## The render_pdf.sh Pipeline

```bash
# Writes checkout-generated artifacts and the canonical API block.
# For fully isolated PDF work, use a separate checkout.
unset QUADMATH_OUTPUT_DIR
UV_PYTHON=3.13 bash quadmath/scripts/render_pdf.sh
# Reuse existing checkout figures, but still regenerate glossary and compile:
UV_PYTHON=3.13 bash quadmath/scripts/render_pdf.sh --skip-figures
```

### Pipeline Phases

1. Verify Pandoc, XeLaTeX, `fc-list` and DejaVu Serif/DejaVu Sans Mono.
2. Check the explicit `MODULES`/`EXCLUDED_MODULES` inventory against numbered
   Markdown files, prepare output directories and extract the preamble.
3. Run figure generation unless skipped, then regenerate the API glossary,
   then run strict manuscript validation.
4. Build each module and the combined `quadmath_review.pdf`, exporting TeX
   through shared Pandoc options and XeLaTeX passes.

The renderer does not run pytest, coverage, Lean or the operational-doc checker.
Its `OUTPUT_DIR` is fixed to `quadmath/output/`: **`QUADMATH_OUTPUT_DIR` only
redirects cooperating Python generators**, not PDF/TeX/temp paths. Mixing an
override with this renderer can leave the manuscript using older checkout figures.

## Manuscript Sections

Canonical mathematics is maintained in `markdown/`, not repeated in this guide:

- [Methods and exact geometry](markdown/03_quadray_methods.md),
  [equations](markdown/08_equations_appendix.md), [specification](../SPEC.md).
- [Field learning](markdown/11_ivm_field_learning.md),
  [dynamics](markdown/12_ivm_dynamics.md), [lattice tooling](markdown/13_lattice_tooling.md),
  [conversions](markdown/14_conversions_spec.md), [evaluation](markdown/15_learning_evaluation.md).
- [Lattice gallery](markdown/16_lattice_gallery.md),
  [benchmarks/statistics](markdown/17_benchmarks_statistics.md),
  [statistics gallery](markdown/18_stats_gallery.md),
  [API glossary](markdown/10_symbols_glossary.md).

`00_preamble.md` supplies the extracted TeX header, not a standalone numbered
section. The renderer's module table is the authoritative build inventory;
adding a chapter requires updating it or recording an explicit exclusion.
`docs/manuscript/` is a separate scripted projection; see its
[guide](../docs/manuscript/README.md) and [status](../docs/MANUSCRIPT_STATUS.md).
A new build is not the archived version-1 Zenodo PDF.

## Build Requirements

Python requirements are declared in [`pyproject.toml`](../pyproject.toml):
NumPy, Matplotlib, Pillow and SymPy, with development test/coverage tools.
Scientific recipes explicitly use Python 3.13 without changing `.python-version`.
Animation formats can additionally depend on external encoders such as ffmpeg.

```bash
# macOS: TeX Live/MacTeX may require PATH setup in a new shell.
brew install pandoc fontconfig
brew install --cask mactex-no-gui font-dejavu
# Debian/Ubuntu:
sudo apt-get install pandoc texlive-xetex texlive-fonts-recommended fontconfig fonts-dejavu
```

Fonts and toolchain versions affect layout. `SOURCE_DATE_EPOCH` controls the
renderer date; it does not alone ensure byte-identical PDFs across environments.

## Validation and cleanup boundaries

```bash
uv run --python 3.13 python quadmath/scripts/generate_glossary.py --check
uv run --python 3.13 python quadmath/scripts/validate_markdown.py --strict
```

Without `--check`, glossary generation updates only its marked canonical API
block. Validators do not replace semantic review or visual inspection.
`clean_output.sh` removes the whole fixed checkout `quadmath/output/` tree;
it does not honor the Python override. Never invoke it without explicit output
preservation and authorization.

## Cross-References

- [Agent guidance](AGENTS.md)
- [Script guide](scripts/README.md)
- [Canonical source formatting](markdown/README.md)
- [Development workflow](../WORKFLOW.md)
