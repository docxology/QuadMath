# QuadMath Agent Guide: Source Ownership and Evidence Boundaries

## Repository Architecture

- `src/quadmath/`: importable runtime library. Reusable numerical and rendering
  methods belong here; use concrete subpackage imports.
- `tests/unit/`: library behavior/resource regressions; `tests/tools/`:
  checkout-orchestration regressions.
- `quadmath/scripts/`: thin example/generation and manuscript/PDF orchestration.
- `quadmath/markdown/`: canonical mathematical chapters. Preserve explicit
  equation/code anchors; regenerate only the marked API-glossary block.
- `docs/`: operational guides; `docs/manuscript/` is a scripted projection,
  not a second source of mathematical prose or hand-edited figures.
- `quadmath/output/`: generated, partly tracked and possibly user-modified.
- `lean/`: independent handwritten Lean models, not Python extraction.

See [ARCHITECTURE.md](ARCHITECTURE.md), [WORKFLOW.md](WORKFLOW.md) and the
[documentation index](docs/README.md). Scoped `AGENTS.md` refine local guidance.

## Before Making Changes

Read the implementing source and its callers. Respect task ownership and
unexpected user changes. During concurrent/delegated implementation, do not
run builds, test suites, lint or formatters; the integration owner runs those
once edits settle. Isolated real consumer/render probes are distinct from
suite verification and must identify actual inputs, outputs and limitations.
Do not mutate Git branches, tags or generated release artifacts unless assigned.

## Source Code Standards (`src/`)

- Keep computation in existing modules unless a separate responsibility is real.
  No production dependency on symergetics, QuadCraft or checkout-only tools.
- Preserve exact integer/rational arithmetic until the declared floating boundary.
  Physical gauge classes are rank three; raw four-tuples need not compare equal
  merely because their physical positions coincide.
- Refer to [SPEC.md](SPEC.md) and canonical
  [methods](quadmath/markdown/03_quadray_methods.md) for embedding and volume
  contracts. Native volume may be `Fraction`; do not infer integrality from a name.
- Give public methods accurate annotations and docstrings, including units,
  gauge, shapes, errors and ownership of returned resources.
- Find exact-symbol references before changing signatures/exports; keep
  `__all__` and package exports intentional, accounting for colliding names.
- Use seeded RNG inputs where appropriate; do not call measured timings or
  arbitrary cross-platform image/media bytes deterministic.

## Test Standards (`tests/`)

Use real numerical/geometry examples, not mocked numerical implementations.
Temporary output redirection and controlled external-tool fixtures are different
from replacing the mathematics. Prefer regressions for uncertain observable
behavior, invariants, resource closure, errors and transitions; do not add tests
that only pin wording, test counts, default spellings or source text.
The branch-enabled coverage floor is a configured aggregate integration gate,
not proof of universal correctness or a current measured result.

## Documentation Standards (`quadmath/markdown/` and `docs/`)

Canonical chapters own mathematical explanations; operational guides link there.
Keep module recipes source-accurate and self-contained, distinguish validation
scores from independent tests, and describe Lean theorem/trust scope honestly.
No saved current line/test counts or promises of automatic semantic coherence.
Update inbound heading links when renaming; preserve explicit equation/code anchors.
Do not hand-edit generated manuscript projections, copied figures or user outputs.

## Build Pipeline Standards (`quadmath/scripts/`)

Reuse package renderers and `quadmath.paths`, not obsolete flat imports.
Headless recipes use `MPLBACKEND=Agg`; use existing `figure_scope`/`save_figure`
patterns to close resources on success and errors. Emit actual output paths.
`make_all_figures.py` owns its explicit generator list and manifest; adding a
chapter also requires `render_pdf.sh` module/exclusion inventory maintenance.
The shell renderer compiles documents but does not run pytest, coverage or Lean.

## Key Commands

From the repository root, scientific commands explicitly select Python 3.13;
leave `.python-version` unchanged. For an assigned integration owner:

```bash
uv sync --python 3.13
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 coverage run -m pytest -q
uv run --python 3.13 coverage report
uv run --python 3.13 python quadmath/scripts/generate_glossary.py --check
uv run --python 3.13 python quadmath/scripts/validate_markdown.py --strict
uv run --python 3.13 python docs/development/check_links.py
```

For isolated Python rendering, set `QUADMATH_OUTPUT_DIR` to a fresh external
root and use a specific generator. **This does not redirect the shell PDF
renderer or cleanup script.** PDF builds update checkout outputs and the
canonical API glossary; use an isolated checkout when needed. Full commands
and dependencies are in [WORKFLOW.md](WORKFLOW.md).

## Quality Gates and Handoff

The integration owner selects and reports numerical tests/coverage,
source/docs-link checks, actual visual inspection, PDF builds and Lean
elaboration separately. A successful image save or stubbed tool test does not
establish visual quality or a real PDF build. Report only exercised checks,
include expanded changed paths and actual probe outputs, and identify unresolved
cross-owner links. Never clean `quadmath/output/` merely because it is generated.

## Cross-References

- [Source guide](src/README.md)
- [Test guide](tests/README.md)
- [Figure/PDF guide](quadmath/README.md)
- [Learning operations](docs/learning/README.md)
- [Benchmark/statistics operations](docs/analysis/README.md)
- [Lean scope](docs/lean/README.md)
