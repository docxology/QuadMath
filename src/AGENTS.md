# Source Agent Guide: Runtime Contracts and Figure Ownership

`src/quadmath/` is the wheel's runtime package. See the [module map](README.md),
[root rules](../AGENTS.md) and [workflow](../WORKFLOW.md).

## Before changing a public method

Read its implementation and exact-symbol callers; inspect concrete module
`__all__` and package initializers before modifying exports. Top-level exports
are selected, not mandatory for every new API. Preserve concrete imports for
colliding gallery/ball names. Internal imports use `quadmath.*`, not obsolete
flat `quadray`/`paths` imports. Do not add dependencies on checkout scripts or
sibling suite packages.

## Mathematical methods

Follow [SPEC.md](../SPEC.md) and the canonical
[methods chapter](../quadmath/markdown/03_quadray_methods.md). Preserve common-offset
invariance, rational exactness and explicit floating boundaries. Native tetra-volume
is not always integral. Shell-norm, Euclidean-radius and BFS enumerations have
different contracts. Hard-boundary harmonic extension differs from soft fitting;
conservative transport differs from symmetric-normalized heat. Do not broaden
claims beyond the actual algorithm or theorem assumptions.

## Rendering and resource lifecycle

Reusable renderers belong in `viz`; orchestration scripts select inputs and
print paths. `viz.abstracts.render_quadray_abstract` is a module-qualified API,
not a required root-package export. Use `paths` output helpers and existing
`viz._common.figure_scope`/`save_figure` patterns. Close owned figures on success
and save failure; axes primitives must respect caller ownership. Document exact
units/counts and make visual data informative without oversized exact labels.
Set Agg before Matplotlib imports in headless recipes. Never overwrite tracked
user artifacts during a probe: select a fresh `QUADMATH_OUTPUT_DIR`.

## Behavior regressions and integration

Use real numerical examples and local seeded generators. Cover uncertain consumer
geometry, extreme exact values, errors and resource/transition behavior; do not
add source-text, wording, saved-count or existence-only regressions. Type hints
and docstrings must state shapes, gauge, units and resource ownership accurately.
Coverage is a configured integration gate, not proof of all behavior.

During delegated implementation skip builds/tests/lint/formatters. Supply changed
paths and actual isolated probe evidence; the integration owner runs selected
verification once edits settle. Commands are in the [test guide](../tests/README.md).
