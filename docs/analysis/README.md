# Analysis Operations: Measured Timings, Resampling and Statistics Figures

Keep numerical inference, timing measurements and illustrative rendering separate.
The canonical treatments are [benchmarks/statistics](../../quadmath/markdown/17_benchmarks_statistics.md)
and the [statistics gallery](../../quadmath/markdown/18_stats_gallery.md).
This page maps methods and practical recipes, not fixed performance claims.

## Timing harness and workload scope

[`stats/benchmarks.py`](../../src/quadmath/stats/benchmarks.py) provides:

- `time_callable(fn, trials=..., warmup=...)`: discarded warmup calls followed
  by a list of `time.perf_counter` wall-time samples.
- `bench_conversions`, `bench_shell_enumeration`, `bench_lattice_search`,
  `bench_field_fit`: seeded workload constructors returning `BenchRow` lists.
- `run_all`: use the module's `BENCH_DEFAULTS` at call time.
- `summary_table`: format measured rows; `BenchRow.as_dict` exposes row fields.

Workload seeds stabilize inputs, not timings. Record hardware, interpreter,
dependency versions, workload sizes, warmup and trial policy before comparing
runs. `ops_per_s` is based on the constructor's declared work count divided by
mean time; it is not an independent measure of mathematical operations. The
harness does not enforce a speed threshold or prove asymptotic complexity.

## Statistical methods and assumptions

[`stats/statistics.py`](../../src/quadmath/stats/statistics.py) supplies NumPy-based
methods without a SciPy dependency:

| Methods | Interpretation |
|---|---|
| `sample_mean`, `summarize`, `cohens_d` | Range-safe mean, descriptive summaries and pooled-standard-deviation effect size |
| `bootstrap_ci` | Streamed seeded percentile bootstrap, defaulting to `sample_mean`; resampling assumptions still matter |
| `permutation_test` | Pooled-label mean test with add-one p-values and centered range-based tie tolerance; exchangeability remains an assumption |
| `p_adjust_bonferroni`, `benjamini_hochberg` | Multiple-comparison adjustments with different error-control meanings |
| `scaling_fit`, `scaling_predict` | Log-ratio OLS reporting tuple and stable anchored predictions, not a complexity proof |
| `jackknife_ci` | Leave-one-out normal interval and bias, defaulting to `sample_mean`; unrepresentable results raise |
| `welch_t_test` | Unequal-variance two-sample test implemented locally |
| `rotation_stats` | Circular summaries for angles in radians |

Seeded resampling is repeatable within its numerical environment. Neither a
bootstrap nor random site folds automatically correct spatial/temporal dependence.
Use [learning operations](../learning/README.md) for selection/test boundaries.

Sample arguments are nonempty finite one-dimensional vectors. Resampling
callbacks receive one vector and must return a finite real scalar.
`sample_mean` uses compensated summation with an exact overflow-only fallback;
local moment scales and exact quantile interpolation avoid unnecessary range
loss. Singleton descriptive standard deviation is undefined (`NaN`); truly
out-of-range descriptive spread may be infinite. Welch uses numerical
Student-t tails, not a large-degree normal approximation.

## Composable statistics plotting

[`viz/vis_stats.py`](../../src/quadmath/viz/vis_stats.py) exposes
`plot_latency_hist`, `plot_scaling_loglog`, `plot_ci_bars` and `plot_ecdf`.
They accept supplied data and optional caller-owned axes rather than running
the benchmark suite themselves. Manage figures created by a caller explicitly.
`gallery(out_dir)` owns its complete saved figures and writes:

- `stats_gallery_latency.png`
- `stats_gallery_scaling.png`
- `stats_gallery_ci.png`
- `stats_gallery_ecdf.png`

The gallery's illustrative seeded data are **not a fresh measurement of this
checkout's performance**. A successful save also does not establish correct
labels, clipping-free layout or cross-platform byte identity.

## Isolated consumer recipes

From the repository root after `uv sync --python 3.13`:

```bash
# Four gallery figures, all outside tracked outputs:
QUADMATH_OUTPUT_DIR="$(mktemp -d)/quadmath-results" MPLBACKEND=Agg \
  uv run --python 3.13 python quadmath/scripts/stats_gallery.py
# A small real wall-clock workload, without modifying default benchmark sizes:
uv run --python 3.13 python -c "from quadmath.stats.benchmarks import bench_conversions, summary_table; print(summary_table(bench_conversions(n=20, trials=2)))"
```

A small resampling example, run with `uv run --python 3.13 python`:

```python
import numpy as np
from quadmath.stats.statistics import bootstrap_ci, permutation_test

sample = np.array([1.0, 2.0, 3.0, 4.0])
print("mean interval:", bootstrap_ci(sample, iters=200, seed=7))
print("permutation p:", permutation_test(sample, sample + 1, iters=200, seed=7))
```

These are consumer recipes, not test-suite gates or evidence of real-world
effectiveness. See [development checks](../development/README.md),
[output policy](../../WORKFLOW.md#output-management) and the
[documentation index](../README.md).
