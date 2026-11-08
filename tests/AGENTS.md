# Test Agent Guide: Real Consumer Behavior, Not Incidental Counters

Read [root guidance](../AGENTS.md), [test operations](README.md) and
[the workflow](../WORKFLOW.md). `unit/` mirrors runtime package responsibilities;
`tools/` tests checkout orchestrators. Do not maintain exhaustive saved inventories
or current test counts in documentation.

## Selecting useful regressions

Use real arithmetic/geometry and declared invariants, including failure/edge
conditions, gauge aliases and resource transitions where consumer behavior is
uncertain. Never mock numerical results. Temporary output redirection, controlled
external-program fixtures and wrapping a real helper do not substitute for real
mathematics; external-tool fixtures do not establish an actual PDF build.

Prefer local seeded random generators. A fixed seed does not guarantee stable
elapsed timings or identical image/video bytes across dependency/tool versions.
Do not add tests that only pin wording, default spellings, source text, current
counts or file existence. Existing incidental assertions should be removed rather
than re-pinned when their subject changes.

## Output/resource safety

Use `tmp_path` or explicit isolated output roots. Never overwrite user-generated
or tracked figures. Test the observable geometry or data alongside a relevant
output contract, not just a nonempty path. Owned figures must close on success
and failure; axes primitives must respect caller-owned figures. Cover actual
resource errors without suppressing them.

## Coverage and handoff

`.coveragerc` enables branch coverage and sets an aggregate report floor.
Configuration is not a current measured percentage or proof of correctness.
The test guide provides explicit Python 3.13 commands for assigned verification.
During delegated implementation do not run suites/builds/lint/formatters; the
integration owner verifies once all edits settle. Report real probes separately,
including inputs, output locations and limitations. Python tests do not elaborate
Lean, inspect every panel's legibility or prove every canonical chapter claim.
