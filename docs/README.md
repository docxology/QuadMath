# QuadMath Documentation: Operational Guides and Canonical Mathematics

This index describes the development checkout, not a frozen package release or
a new receipt for the [archived manuscript DOI](https://zenodo.org/records/16887791).

## Choose a task

| Need | Guide |
|---|---|
| Install, import and orient yourself | [Root README](../README.md), [repository tour](overview.md) |
| Understand runtime modules and exactness | [Source guide](../src/README.md), [architecture](../ARCHITECTURE.md) |
| Change code/docs safely, then integrate | [Workflow](../WORKFLOW.md), [development checks](development/README.md), [tests](../tests/README.md) |
| Fit fields, extend boundary values or identify trajectories | [Learning operations](learning/README.md) |
| Measure runtime, analyze samples or render statistics figures | [Analysis operations](analysis/README.md) |
| Understand Lean declarations and trust limits | [Lean scope](lean/README.md) |
| Generate figures, PDF or TeX | [Manuscript tools](../quadmath/README.md) |
| Understand the template-layout projection | [Projection guide](manuscript/README.md), [manuscript status](MANUSCRIPT_STATUS.md) |

## Canonical mathematical reading

[`quadmath/markdown/`](../quadmath/markdown/README.md) is the editing source of
truth. Start with [SPEC.md](../SPEC.md),
[quadray methods](../quadmath/markdown/03_quadray_methods.md), and
[conversions](../quadmath/markdown/14_conversions_spec.md). Specialized treatments:
[field learning](../quadmath/markdown/11_ivm_field_learning.md),
[dynamics](../quadmath/markdown/12_ivm_dynamics.md),
[evaluation](../quadmath/markdown/15_learning_evaluation.md),
[benchmarks/statistics](../quadmath/markdown/17_benchmarks_statistics.md), and
[API glossary](../quadmath/markdown/10_symbols_glossary.md).

`docs/manuscript/` is a scripted projection with rewritten layout/figure paths;
its numbered chapters and figures must not be independently edited.
Operational pages link to canonical explanations rather than maintaining
conflicting mathematical summaries or current test/file counts.

## Scientific commands and output safety

Run from the repository root, explicitly selecting Python 3.13 while preserving
the existing `.python-version` pin:

```bash
uv sync --python 3.13
QUADMATH_OUTPUT_DIR="$(mktemp -d)/quadmath-results" MPLBACKEND=Agg \
  uv run --python 3.13 python quadmath/scripts/stats_gallery.py
```

This isolates cooperating **Python** generator outputs. The shell PDF renderer
and cleanup script use fixed checkout paths; the override does not isolate their
writes. Preserve tracked and user-generated outputs and use a separate checkout
for isolated PDF work. Validation/build recipes belong to the
[integration workflow](../WORKFLOW.md); workers do not run suites/builds merely
because they edited documentation. A green command is not semantic or visual
review of every mathematical claim.
