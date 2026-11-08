# Development Checks: Numerical Evidence, Links and Render Boundaries

This page describes integration commands, not a current green-run receipt.
The [root workflow](../../WORKFLOW.md) assigns verification once concurrent edits
have settled. Workers skip builds/test suites/lint/formatters during implementation;
real isolated consumer probes are reported separately.

## Python environment and numerical gates

From the repository root, explicitly use scientific Python 3.13; do not change
`.python-version` merely to run a recipe.

```bash
uv sync --python 3.13
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 coverage run -m pytest -q
uv run --python 3.13 coverage report
```

`.coveragerc` enables branch measurement and an aggregate report floor of 100.
That configuration is not a current coverage result, a per-module proof or
proof of every mathematical claim. Numerical tests use real arithmetic/geometry;
external-tool fixtures do not establish an actual PDF build. See the
[test guide](../../tests/README.md) for scope, naming and resource regressions.

## Source and documentation checks

```bash
# Check without rewriting the canonical generated API block:
uv run --python 3.13 python quadmath/scripts/generate_glossary.py --check
# Canonical manuscript source rules:
uv run --python 3.13 python quadmath/scripts/validate_markdown.py --strict
# Operational docs and template-layout projection links:
uv run --python 3.13 python docs/development/check_links.py
# Explicit multi-root scan from the QuadMath checkout (siblings must exist):
uv run --python 3.13 python docs/development/check_links.py \
  docs ../symergetics/docs ../QuadCraft/docs
```

Without `--check`, glossary generation writes its marked block in
`quadmath/markdown/10_symbols_glossary.md`. Other prose is not automatically
synchronized with source. The manuscript validator owns its implemented
canonical-source rules. The checker CLI is `check_links.py [ROOT ...]`, where
roots are documentation directories; no arguments defaults to QuadMath `docs/`.
Missing or empty roots fail instead of producing a vacuous pass. Same-file
`#heading` and cross-file fragments are checked, including duplicate heading
slugs, URL-decoded targets and anchors in referenced Markdown outside scan roots.
Fenced/inline code is excluded and remote content is not validated.
Neither tool verifies every Markdown dialect, mathematical truth or rendered
legibility. Root/scoped guides outside selected docs roots still need review.

Keep explicit equation/code anchors stable and update inbound links when headings
change. Link mathematical derivations to canonical chapters rather than maintain
another inconsistent version under operational docs. The template-layout
projection is generated according to its [guide](../manuscript/README.md);
do not independently edit copied numbered sections or figures.

## Isolated figures versus checkout PDF writes

```bash
QUADMATH_OUTPUT_DIR="$(mktemp -d)/quadmath-results" MPLBACKEND=Agg \
  uv run --python 3.13 python quadmath/scripts/stats_gallery.py
```

Cooperating Python renderers use `quadmath.paths`, which honors the selected
output root. Installed-wheel consumers should set it explicitly to avoid
checkout-root discovery. Rendering a figure is a direct consumer action, not a
suite or PDF build; inspect actual panels, scale, units and clipping separately.

```bash
# Assigned integration workflows:
UV_PYTHON=3.13 bash run_all.sh --test-only
UV_PYTHON=3.13 bash run_all.sh
# Deliberate checkout PDF writes; use a separate checkout for isolation.
unset QUADMATH_OUTPUT_DIR
UV_PYTHON=3.13 bash run_all.sh --with-pdf
```

The default wrapper runs suite/coverage, figures, glossary freshness and strict
manuscript validation. `--with-pdf` adds the shell renderer with existing figures;
that renderer still regenerates the API block, validates and compiles PDFs.
It does not run Lean or the docs checker. Its PDF/TeX/temp paths are fixed under
`quadmath/output/`: the Python override does not isolate this entire pipeline.
See [build dependencies](../../quadmath/README.md#build-requirements).

## Separate Lean evidence and safe handoff

Lean elaboration (`lake build` from `lean/`) and declaration-level trust audits
are separate assigned gates; [the Lean guide](../lean/README.md) states their
limits. Do not infer universal shell counts from finite instances or a fresh
build from a source inventory.

Preserve generated user work. `clean_output.sh` removes the fixed checkout tree,
not the Python override; never advertise it as automatically safe. A handoff
records expanded changed paths, actual probe outputs, unrun gates and cross-owner
links. Source-linked corrections are useful evidence, not a substitute for the
integration owner's final combined checks and visual inspection.

Return to [documentation](../README.md), [architecture](../../ARCHITECTURE.md)
or [root agent guidance](../../AGENTS.md).
