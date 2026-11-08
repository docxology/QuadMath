# Analysis Documentation Agent Guide: Measurement and Plot Provenance

Document actual `stats.benchmarks`, `stats.statistics` and `viz.vis_stats`
contracts. Read [operations](README.md), [docs guidance](../AGENTS.md), and
[development checks](../development/README.md).

## Evidence boundaries

Timing workload seeds do not make wall-clock measurements deterministic.
State workload sizes, environment and summary policy rather than current speed
promises. A log-log slope is a fit, not a complexity proof. Bootstrap,
permutation and multiple-testing methods retain their scientific assumptions.
The stats gallery's seeded illustrative data are not fresh benchmark results.

Link methodology to canonical `17_benchmarks_statistics.md` and
`18_stats_gallery.md` under `quadmath/markdown/`; do not maintain a second
inconsistent derivation or edit generated projection copies. Runtime APIs are
owned by source implementers, not this docs task.

## Recipes and maintenance

Use explicit Python 3.13. Isolate Python rendering with a fresh
`QUADMATH_OUTPUT_DIR` and preserve tracked/user outputs. Caller-owned axes and
complete gallery figures have different lifecycle responsibilities; describe the
actual one. Do not promise byte identity across plotting/font/tool versions.
Keep heading links navigable; no current coverage/test-count claims.

During delegated edits skip builds/tests/lint/formatters and report direct probes
separately. The integration owner checks combined docs and inspects actual figures.
