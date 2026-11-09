# Learning Documentation Agent Guide: Method and Evaluation Boundaries

This folder documents `lattice.ivm_field`, `lattice.ivm_dynamics`,
`learn.learning_eval` and related `viz.vis_lattice` recipes. Read
[operations](README.md), [docs guidance](../AGENTS.md) and
[development workflow](../development/README.md).

## Keep method claims source-bound

Distinguish shell-radius field balls, Euclidean-radius dynamics balls and
BFS enumeration.
`learn` uses soft observations at positive regularization; zero regularization
evaluates its kernel target directly. `harmonic_extension` instead fixes hard
boundary values, with an anchor in every connected component.
Symmetric-normalized heat, conservative concentration transport and integer
rounded averaging have different invariants. Do not promise mass conservation
for normalized heat or squared-norm contraction for majority updates.

Random site k-folds are not spatially blocked. Selection-fold scores and refit
errors are not untouched final-test estimates. Temporal forecasts require a
chronological prefix/suffix and must distinguish continuation from teacher-forced
one-step error. `fit_geometry` is a linear least-squares map, not automatically
a constrained rigid fit. Keep example inputs complete and scale/gauge explicit.

## Canonical navigation and ownership

Link derivations to canonical field/dynamics/evaluation/gallery chapters under
`quadmath/markdown/`, not copied projection prose. Do not hand-edit
`docs/manuscript/` numbered sections or figures. No source changes from this
documentation task. Explicit Python 3.13 and isolated Python output roots are
required in rendering recipes; shell PDFs do not honor the override.

During delegated work defer suites/builds/lint/formatters to integration; report
only actual consumer probes. Preserve anchors and report cross-owner link changes.
