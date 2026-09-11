# Statistics and Scaling Gallery

## Overview

This section is the figure surface of the benchmarks-and-statistics layer:
every rendering primitive used by section `17_benchmarks_statistics.md`
lives in `src/quadmath/viz/vis_stats.py` (`plot_latency_hist`, `plot_scaling_loglog`,
`plot_ci_bars`, `plot_ecdf`).  The primitives are input-agnostic — plain
numpy arrays in, artists out; they import nothing from the other `src/`
modules and receive their matplotlib axes explicitly, so panels compose
inside caller-owned figures and nothing renders at import time.  The
command-line entry is `quadmath/scripts/stats_gallery.py` — a thin
orchestrator that sets a headless backend and a fixed seed (`seed=32`),
delegates to the gallery function in `src/quadmath/viz/vis_stats.py` (thin-orchestrator
contract of `quadmath/scripts/AGENTS.md`), and prints each written path on
its own line: those stdout lines are the `make_all_figures` manifest
contract.  The gallery writes exactly four PNGs —
`stats_gallery_latency.png`, `stats_gallery_scaling.png`,
`stats_gallery_ci.png`, `stats_gallery_ecdf.png` — and every random draw
comes from the fixed seed, so the whole set is deterministic.

## Latency histogram

`plot_latency_hist(times, ax=None, *, bins=20, title=...)` renders the
distribution of per-call wall-clock durations measured by
`src/quadmath/stats/benchmarks.py::time_callable` (trials summary in
`17_benchmarks_statistics.md`) as a histogram with `bins` bins on the given
axes.  The shape of the distribution carries what the mean alone hides: a
tight spike means the harness saw a stable workload, while a heavy right
tail is exactly what the `median_s` / `p95_s` fields of `BenchRow` are
there to expose.  The gallery panel times a representative benchmark
constructor and histograms its per-trial durations.

![**Latency distribution of a timed benchmark.** Histogram of per-call wall-clock times measured with `time.perf_counter` (one warmup call discarded), rendered by `plot_latency_hist`; the spread and right tail complement the `mean_s`, `median_s`, and `p95_s` fields of `BenchRow`.](../output/figures/stats_gallery_latency.png)

## Scaling fit

`plot_scaling_loglog(sizes, times, ax=None, *, title=...)` plots measured
workload sizes against durations on log-log axes and overlays the
least-squares power law of `src/quadmath/stats/statistics.py::scaling_fit` — the slope is
the empirical complexity exponent $\beta_1$ of \eqref{eq:stat-scaling}.
On log-log axes a power law is a straight line, so the panel shows at a
glance whether a lattice routine scales linearly, quadratically, or worse,
and how tightly the data follow the law ($r^2$).  The gallery panel fits a
representative size sweep from the benchmark suite.

![**Log-log scaling fit.** Measured durations against workload sizes with the least-squares power law from `scaling_fit` overlaid, rendered by `plot_scaling_loglog`; the fitted slope is the empirical complexity exponent.](../output/figures/stats_gallery_scaling.png)

## Confidence intervals

`plot_ci_bars(labels, means, lows, highs, ax=None, *, title=...)` renders
paired estimates as a bar-and-whisker chart: one bar per label at its
point estimate, with error bars spanning the low and high ends of a
percentile bootstrap interval `src/quadmath/stats/statistics.py::bootstrap_ci`
(\eqref{eq:stat-bootstrap}).  Overlapping intervals visually encode the
same information a permutation test quantifies
(\eqref{eq:stat-permutation}): two conditions whose intervals do not
overlap are the ones the pooled test flags.  The gallery panel shows
bootstrap intervals for several resampling conditions computed at
deterministic seeds.

![**Bootstrap confidence intervals by condition.** Point estimates with 95% percentile bootstrap intervals computed by `bootstrap_ci` (seeded, 2000 resamples), rendered by `plot_ci_bars`; non-overlapping intervals mark the conditions a permutation test would separate.](../output/figures/stats_gallery_ci.png)

## Empirical CDF

`plot_ecdf(values, ax=None, *, title=...)` renders the empirical
distribution function

\begin{equation}
\label{eq:vis-ecdf}
\hat{F}(t) \;=\; \frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}\{t_i \le t\},
\end{equation}

a monotone step function that jumps by $1/n$ at every observed value —
the exact distribution of the sample, with no binning choices at all.  For
timing data the ECDF answers directly what a percentile summary rounds
off: the curve passes through the empirical median at height $0.5$ and
through the 95th percentile at height $0.95$, so `median_s` and `p95_s`
are readable off the plot.  The gallery panel draws the ECDF of a seeded
timing sample next to the same information as the histogram panel above.

![**Empirical cumulative distribution of measured times.** Exact step-function ECDF $\hat{F}(t)$ of a seeded per-call timing sample, rendered by `plot_ecdf`; the empirical median (height 0.5) and 95th percentile (height 0.95) are read directly off the curve.](../output/figures/stats_gallery_ecdf.png)

## Reproducibility and test contract

- The gallery function in `src/quadmath/viz/vis_stats.py` writes exactly the four files
  listed above, in that order, into the output directory (creating it when
  missing) and returns their paths.  Every panel is fully seeded — the
  script fixes `seed=32` — and the PNGs carry no timestamp metadata, so
  re-renders are byte-identical.
- Tests: `tests/test_vis_stats.py` pins the contract with deterministic
  assertions, no mocks, and no pixel diffs — artist placement on
  caller-provided axes, input-agnostic behavior over plain arrays, and the
  invariants of each primitive (monotone non-decreasing ECDF steps, CI
  whisker endpoints matching the lows/highs inputs).  Consistent with
  `17_benchmarks_statistics.md`, no test asserts wall-clock timing
  values.
- Figure regeneration:
  `uv run python quadmath/scripts/stats_gallery.py` (the script sets
  `MPLBACKEND=Agg` itself); stdout is exactly the four written paths, the
  same manifest contract as `quadmath/scripts/lattice_gallery.py`.
- Markdown: `uv run python quadmath/scripts/validate_markdown.py`.

## Cross-references

- The measured quantities behind every panel: `17_benchmarks_statistics.md`
  (`src/quadmath/stats/benchmarks.py`, `src/quadmath/stats/statistics.py`).
- The sibling gallery of the lattice layer, same thin-orchestrator and
  determinism contract: `16_lattice_gallery.md` (`src/quadmath/viz/vis_lattice.py`).