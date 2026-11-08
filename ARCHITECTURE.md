# QuadMath Architecture: Library, Mathematical Sources and Render Pipelines

QuadMath has related but separate runtime, manuscript and formalization surfaces.
A successful render establishes that one recipe completed; it does not establish
universal mathematical correctness, coverage or agreement of every document.

## Component Interactions

```text
src/quadmath/ ──► numerical consumers and tests/
      │
      └──► quadmath/scripts/ ──► selected output root/figures and data
quadmath/markdown/ ──► render_pdf.sh ──► quadmath/output/pdf and tex
      │
      └──► docs/manuscript/port_from_source.py ──► docs/manuscript/
lean/ ──► independent Lean elaboration (not Python verification)
```

### Source Code (`src/`)

The wheel packages `src/quadmath/`. `core/` provides coordinates, exact/rational
metric and affine/oriented operations, determinant volumes, symbolic helpers
and metric diagnostics. `lattice/` provides distinct enumeration/search,
conversion, field and dynamics surfaces. `learn/` supplies evaluation and
linear baselines; `stats/` supplies inference and a timing harness; `viz/`
contains reusable renderers. `pipeline.py` composes typed operations, while
`paths.py` selects Python output locations. See the [module guide](src/README.md).

Runtime code must not require sibling repositories, manuscript scripts, Pandoc
or a Lean installation. Matplotlib, NumPy, Pillow and SymPy are declared package
dependencies; external media/PDF tools are separate concerns.

### Test Suite (`tests/`)

Tests mirror subpackages under `tests/unit/`; `tests/tools/` covers checkout
orchestrators. Some orchestration tests use external-tool substitutes; these
are not evidence of an actual TeX installation or rendered PDF quality.
The branch-enabled `.coveragerc` has `fail_under = 100`, an aggregate report
gate, not a statement that every behavior or theorem has been proved. See
[tests/README.md](tests/README.md) for running and extending the suite.

### Generation Scripts (`quadmath/scripts/`)

Scripts bootstrap `src/quadmath` in a checkout, choose deterministic example
inputs and print generated paths. Reusable computation and rendering belong in
the package, not duplicated orchestration code. `make_all_figures.py` uses its
explicit `FIGURE_SCRIPTS` sequence, drives children with `sys.executable`,
rejects emitted recognized paths outside the configured output root and writes
a relative-path manifest. It requires at least one existing emitted artifact
per generator, not independent inspection of every figure's content.

### Documentation (`quadmath/markdown/`)

Canonical chapters hold the mathematical explanations; [SPEC.md](SPEC.md)
defines conventions. [Operational docs](docs/README.md) explain how to use the
implementation, linking to chapters instead of maintaining another derivation.
Only the marked API block in `10_symbols_glossary.md` is automatically
regenerated; prose elsewhere remains manually maintained.

`docs/manuscript/` is a template-layout projection with rewritten figure paths
and section ordering. Port changes from canonical sources; do not independently
edit its numbered chapters or copied figures. The shared external rendering
workflow is distinct from the checkout's shell renderer; see
[projection documentation](docs/manuscript/README.md) and
[manuscript status](docs/MANUSCRIPT_STATUS.md).

## The render_pdf.sh Pipeline

`quadmath/scripts/render_pdf.sh`:

1. Checks Pandoc, XeLaTeX, fontconfig and the required DejaVu families.
2. Checks its explicit `MODULES`/`EXCLUDED_MODULES` inventory against numbered
   manuscript files and extracts `00_preamble.md` into temporary TeX.
3. Runs `make_all_figures.py` unless `--skip-figures`, then
   `generate_glossary.py` and `validate_markdown.py --strict`, in that order.
4. Converts module/combined Markdown with shared Pandoc options, then compiles
   individual PDFs and `quadmath_review.pdf`, retaining TeX exports.

This renderer does not run pytest, coverage, the `docs/` checker or Lean.
`run_all.sh` separately runs coverage/tests, figures, glossary freshness and
strict manuscript validation; `--with-pdf` adds the shell renderer with
`--skip-figures`. Its default is not a PDF build. See [WORKFLOW.md](WORKFLOW.md)
for interpreter selection and exact commands.

## Data Flow and Dependencies

### Mathematical contracts

Physical four-tuples are classes under a common offset. Exact native volume
can be fractional; Cartesian volume and native IVM volume have different
normalizations. The [methods chapter](quadmath/markdown/03_quadray_methods.md)
and [conversion chapter](quadmath/markdown/14_conversions_spec.md) are canonical.

Field shell radius, Euclidean radius and BFS shell numbering must not be
silently identified. Likewise soft fitting is not hard-boundary harmonic
extension; normalized heat is not mass-preserving concentration transport.
See [learning operations](docs/learning/README.md).

### Output Structure

```text
selected Python output root/
├── figures/       # image/animation artifacts
└── data/          # numeric artifacts and figure_manifest.txt
quadmath/output/   # fixed shell-renderer destination
├── pdf/           # per-section and combined PDFs
├── tex/           # exported TeX
└── latex_temp/    # intermediate render files
```

`QUADMATH_OUTPUT_DIR` is expanded and made absolute by `quadmath.paths`; path
helpers create requested directories. Without an override they search upward
from the package for checkout markers. Installed consumers should set the
override. The shell PDF and cleanup scripts instead use the checkout's
`quadmath/output/`, so the override does not isolate an entire PDF pipeline.

## Quality Assurance Mechanisms

- Numerical and resource/error regressions exercise concrete consumer behavior.
- Coverage identifies unexecuted code; it does not prove arithmetic contracts.
- Manuscript and docs checkers cover different trees and limited link syntax;
  they do not check every citation, semantic claim or rendered layout.
- Seeded inputs aid repeatability. Wall timings, fonts, plotting/TeX versions
  and encoders can change output; do not promise cross-platform byte identity.
- Lean supplies handwritten integer theorem sources with explicit trust and
  finite-computation boundaries, not verification of floating implementations.
  See [Lean scope](docs/lean/README.md).

## Development Workflow

Use [WORKFLOW.md](WORKFLOW.md) for implementation and integration responsibilities,
[root agent guidance](AGENTS.md) for ownership and safe output handling, and
[the documentation index](docs/README.md) for focused operational topics.
