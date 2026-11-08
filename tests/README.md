# QuadMath Test Suite: Numerical Behavior and Integration Boundaries

Tests exercise the runtime package and checkout tools. Results, collection
counts and coverage percentages are properties of a particular run, not
permanent guarantees. The [workflow](../WORKFLOW.md) assigns integrated
verification after concurrent implementation work has settled.

## Running Tests

From the repository root, when verification is explicitly assigned:

```bash
uv sync --python 3.13
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 coverage run -m pytest -q
uv run --python 3.13 coverage report
# Focus on one implementation surface:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 pytest tests/unit/core/test_quadray.py -v
# Obtain the actual collection inventory without maintaining saved counters:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 pytest --collect-only -q
```

Do not run these from every delegated worker. Report only exercised checks and
include their revision/environment when publishing a receipt.

## Test File Mapping

| Location | Scope |
|---|---|
| `unit/core/` | Coordinates, exact/rational metric and volume behavior, linear algebra, geometry and information diagnostics |
| `unit/lattice/` | Enumeration/search/conversions, field learning, graph update laws and trajectory identification |
| `unit/learn/` | Fold selection, chronological holdout and linear fitting |
| `unit/optimize/`, `unit/inference/` | Optimization and information/free-energy helpers |
| `unit/stats/` | Statistical behavior and timing-harness structure, not fixed timing thresholds |
| `unit/viz/` | Plot/animation output contracts, geometry and figure lifecycle/error behavior |
| `unit/validate/`, `unit/tools/` | Numerical validation reports and runtime utilities |
| `unit/test_paths.py`, `unit/test_pipeline.py` | Output policy and typed composition |
| `tools/` | Figure/glossary/Markdown orchestration and shell-renderer contracts |

The usual correspondence is `src/quadmath/<area>/<module>.py` to
`tests/unit/<area>/test_<module>.py`, with supplemental `*_cov.py` files in
some areas. Consult the live tree rather than a duplicated exhaustive file list.

## Test Configuration

`conftest.py` selects the Agg backend and makes checkout package/scripts paths
available. Runtime imports use `quadmath.*`; checkout script modules may be
loaded directly. Installed-wheel consumers do not inherit these test path
bootstraps, so package-consumer evidence is separate from source-tree tests.

The root [.coveragerc](../.coveragerc) enables branch measurement over `src`
and sets `fail_under = 100`. This is an aggregate coverage-report gate with
configured rounding, not independent proof of 100% behavior coverage for each
module. `coverage run` and `coverage report` are separate commands; passing
pytest alone does not establish that the reporting gate passed.

## Testing Patterns

Use real arithmetic and geometry examples, including gauge aliases, degeneracy,
large exact values and declared error conditions where relevant. For example:

```python
from fractions import Fraction
from quadmath.core.quadray import Quadray, integer_tetra_volume


def test_primitive_native_volume_is_fractional():
    vertices = [Quadray(0, 0, 0, 0), Quadray(1, 0, 0, 0),
                Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0)]
    assert integer_tetra_volume(*vertices) == Fraction(1, 4)
```

Do not mock numerical implementations or internal arithmetic. Temporary-output
redirection, controlled external-program fixtures and wrapping a real helper
are different from replacing the mathematical result. Seed random workloads
with a local generator; test meaningful invariants and results, not a mandate
that every function use the same global seed.

For plotting, a saved file alone is weak evidence: relevant regressions should
exercise geometry, labels derived from data, output semantics and closure on
success/error. Caller-owned axes must not be closed by a primitive. Use
`tmp_path`/explicit output locations, and do not overwrite tracked user figures.
Avoid permanent tests that only pin wording, exports' source spelling, default
text, current counts or existence without consumer behavior.

## What the suite does not establish

- Stubbed Pandoc/XeLaTeX contract tests are not actual PDF builds or font checks.
- Seeded input tests do not promise machine-independent benchmark times or
  byte-identical figures/media across different libraries and encoders.
- Python tests do not elaborate Lean theorem sources or prove all manuscript claims.
- Script tests and Markdown checks do not inspect every rendered panel for legibility.

These require separate assigned gates and honest run receipts. See
[build guide](../quadmath/README.md), [Lean scope](../docs/lean/README.md),
[source guide](../src/README.md) and [agent testing guidance](AGENTS.md).
