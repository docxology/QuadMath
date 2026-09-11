# analysis/ — benchmarks, statistics, and stats visualization

Documentation for the measurement and analysis modules over the manuscript's
numerical surfaces — all landed on disk (checked 2026-09-11):

- `src/benchmarks.py` — `perf_counter` timing harness over the core
  surfaces (`BenchRow`, `time_callable`, the four `bench_*` constructors,
  `run_all`, `summary_table`)
- `src/statistics.py` — resampling-based inference on standalone numpy
  (`summarize`, `bootstrap_ci`, `permutation_test`, `cohens_d`,
  `p_adjust_bonferroni`, `scaling_fit`)
- `src/vis_stats.py` — the input-agnostic matplotlib rendering primitives
  behind the statistics gallery (see section `18_stats_gallery.md`)

Design lineage: measured surfaces (`src/quadray.py`, `src/omni_numbering.py`,
`src/lattice_search.py`, `src/ivm_field.py`) → timing harness → statistical
analysis → rendering, all numpy + matplotlib only and deterministic (no new
dependencies; seeded `numpy.random.default_rng` everywhere).

Manuscript treatments (source of truth under `quadmath/markdown/`):
`17_benchmarks_statistics.md` (methodology and a reproducible numeric
example) and `18_stats_gallery.md` (the four-figure gallery).

## `src/benchmarks.py` — perf_counter timing harness

Public API: `BenchRow` (frozen dataclass: `name`, `n`, `trials`, `total_s`,
`mean_s`, `median_s`, `p95_s`, `ops_per_s`, with `as_dict()`),
`time_callable(fn, *, trials=5, warmup=1)`, `bench_conversions(n=200,
trials=5)`, `bench_shell_enumeration(k_max=4, trials=5)`,
`bench_lattice_search(n_sites=200, queries=50, trials=5)`,
`bench_field_fit(n_sites=64, trials=3)`, `summary_table(rows)`, `run_all()`,
`BENCH_DEFAULTS`.

- **Method**: `time.perf_counter` wall-clock timing; one warmup call is run
  and discarded, then `trials` timed calls produce `mean_s`, `median_s`,
  and `p95_s` summaries plus throughput `ops_per_s = n / mean_s`.
- **Constructors**: `bench_conversions` times quadray/embedding round-trips,
  `bench_shell_enumeration` times shell enumeration, `bench_lattice_search`
  times nearest-site queries over a random ball, `bench_field_fit` times a
  synthetic `IVMField.learn` fit.  `run_all()` returns the `BenchRow` list;
  `summary_table(rows)` renders a fixed-width text table.
- **The harness measures only** — it never asserts; timing claims live in
  the manuscript, not in test assertions.

## `src/statistics.py` — honest analysis methodology

Public API: `summarize(x)`, `bootstrap_ci(x, stat=np.mean, *, iters=2000,
seed=0, alpha=0.05)`, `permutation_test(a, b, *, iters=2000, seed=0,
alternative="two-sided")`, `cohens_d(a, b)`, `p_adjust_bonferroni(pvals)`,
`scaling_fit(sizes, times) -> (slope, intercept, r2)`.

- **`bootstrap_ci`**: percentile bootstrap over vectorized resampling
  indices (`rng.integers(0, n, size=(iters, n))`); no distributional
  assumption beyond the resampling approximation.
- **`permutation_test`**: pooled labels, permuted and split at `len(a)` per
  iteration; p-values use the add-one convention
  `(count + 1) / (iters + 1)` — never zero, conservative by construction.
  Alternatives `greater`/`less` reuse the same convention with signed
  comparisons.
- **`cohens_d`** — standardized mean difference with pooled standard
  deviation; **`p_adjust_bonferroni`** — family-wise correction
  `min(1, m·p_i)`; **`scaling_fit`** — ordinary least squares on log-log
  data returning the complexity exponent, intercept, and `r2`.
- All numpy-only and exactly reproducible for fixed seeds: every draw comes
  from `numpy.random.default_rng(seed)`.

## `src/vis_stats.py` — input-agnostic stats rendering

Public API: `plot_latency_hist(times, ax=None, *, bins=20, title=...)`,
`plot_scaling_loglog(sizes, times, ax=None, *, title=...)`,
`plot_ci_bars(labels, means, lows, highs, ax=None, *, title=...)`,
`plot_ecdf(values, ax=None, *, title=...)`.

- Every primitive takes plain numpy arrays and an explicit axes object (or
  creates one); none imports from the other `src/` modules — the panels
  compose inside caller-owned figures and nothing renders at import time.
- The gallery function writes exactly four PNGs —
  `stats_gallery_latency.png`, `stats_gallery_scaling.png`,
  `stats_gallery_ci.png`, `stats_gallery_ecdf.png` — in that order into the
  figure directory, creating it when missing.

## Running the surface

```bash
# Regenerate the four gallery figures (sets MPLBACKEND=Agg and seed=32;
# stdout is exactly the four written paths)
uv run python quadmath/scripts/stats_gallery.py

# Run the benchmark suite and print its summary table
uv run python -c "import sys, os; sys.path.insert(0, os.path.join('src')); from benchmarks import run_all, summary_table; print(summary_table(run_all()))"
```

## Invariants

- **Determinism**: fixed seeds everywhere (`seed=0` defaults in
  `bootstrap_ci` / `permutation_test`, `seed=32` in the gallery script);
  re-renders are byte-identical and the numbers in
  `17_benchmarks_statistics.md` reproduce exactly.
- **Headless rendering**: scripts set `MPLBACKEND=Agg` themselves; nothing
  touches a display.
- **No timing assertions in tests**: `tests/` pins structure and
  determinism (artist placement, ECDF monotonicity, exact statistics), but
  never asserts wall-clock performance values.
- **100% coverage**: `tests/test_benchmarks.py`, `tests/test_statistics.py`,
  and `tests/test_vis_stats.py` fully cover the three modules (the repo's
  `test_<module>.py` convention; workflow in
  [development](../development/README.md)).

## Reading order

1. `quadmath/markdown/17_benchmarks_statistics.md` — the methodology and
   the reproducible numeric example
2. `quadmath/markdown/18_stats_gallery.md` — the four-figure gallery
3. This README — API map
4. `tests/test_benchmarks.py`, `tests/test_statistics.py`,
   `tests/test_vis_stats.py` — the contracts as executable assertions
   (fixed seeds, no mocks — see [development](../development/README.md))