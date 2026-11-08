# Manuscript Tool Agent Guide: Canonical Sources and Safe Outputs

Read [root guidance](../AGENTS.md), [build operations](README.md), and scoped
instructions before editing `markdown/` or `scripts/`.

## Directory Roles

- `markdown/`: canonical chapters, preserved equation/code anchors and preamble.
  Only the marked API-glossary block is generated from Python.
- `scripts/`: thin numerical/render orchestration and checkout build tools.
- `output/`: generated and partly tracked artifacts that may contain user work.
  Regeneratable does not mean safe to delete or overwrite.

`docs/manuscript/` is a scripted projection of canonical sources. Do not hand-edit
its chapter/figure copies; coordinate porting with the integration owner.

## Script Development Standards

Import reusable operations from `quadmath.*`, not obsolete flat modules such as
`quadray` or `paths`. Use existing package renderers, `quadmath.paths`,
`figure_scope` and `save_figure` patterns; keep resource closure correct on errors.
Use headless rendering and explicit seeded inputs where needed. Print actual
written paths for manifest collection. Never promise machine-independent timing
or cross-toolchain image/media bytes because the inputs are seeded.

Adding a generator requires its explicit `make_all_figures.py` inventory entry.
Adding a numbered chapter requires a `render_pdf.sh` module entry or a reasoned
exclusion. Actual mathematical content belongs in canonical chapters, not a
second operational derivation.

## Output and orchestration boundary

Use fresh `QUADMATH_OUTPUT_DIR` for isolated Python recipes. The shell renderer
and cleanup script fix their paths under checkout `quadmath/output/`; the
override does not isolate PDF/TeX writes or glossary updates. Use an isolated
checkout for full PDF work. Never invoke `clean_output.sh` as routine preparation.

The renderer generates figures, refreshes the glossary, validates Markdown and
compiles PDFs; it does not run pytest, coverage or Lean. `run_all.sh` is the
separate test/coverage wrapper, with PDF only under `--with-pdf`. All scientific
recipes explicitly select Python 3.13; preserve `.python-version`.

## Delegated implementation

Skip builds/tests/lint/formatters while editing. Isolated real consumer probes
are allowed when assigned, with actual output paths reported. The integration
owner runs combined checks and inspects visuals once edits settle. Preserve
immutable releases and user outputs; report cross-owner heading-link changes.
