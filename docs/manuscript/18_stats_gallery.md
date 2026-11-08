# Statistics and Scaling Gallery

## Synthetic examples of statistical plotting primitives

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
comes from a fixed seed. These are synthetic plotting examples, not benchmark measurements; byte reproducibility assumes a fixed rendering environment.

## Latency histogram

`plot_latency_hist(times, ax=None, *, bins=20, title=...)` renders the
distribution of per-call wall-clock durations measured by
`src/quadmath/stats/benchmarks.py::time_callable` (trials summary in
`17_benchmarks_statistics.md`) as a histogram with `bins` bins on the given
axes.  The shape of the distribution carries what the mean alone hides: a
tight spike means the harness saw a stable workload, while a heavy right
tail is exactly what the `median_s` / `p95_s` fields of `BenchRow` are
there to expose.  The gallery panel histograms a deterministic synthetic
stand-in for such a sample: 256 draws of a lognormal distribution (mean 0,
$\sigma = 0.6$, clipped to $[0.05, 5.0]$ in seconds-scale units) taken from the
single fixed-seed `numpy.random.default_rng(32)` generator, so re-renders are
byte-identical.

![**Latency distribution.** Histogram (20 bins) of a synthetic 256-draw lognormal latency sample (mean 0, $\sigma = 0.6$) clipped to [0.05, 5.0] in seconds-scale units, drawn from the fixed-seed `numpy.random.default_rng(32)` generator of `stats_gallery.py`; the dashed line marks the sample mean. Rendered by `plot_latency_hist`; reproduced by `uv run python quadmath/scripts/stats_gallery.py`. The spread and right tail complement the `mean_s`, `median_s`, and `p95_s` fields of `BenchRow`, whose per-call times `time_callable` measures with `time.perf_counter` after one discarded warmup call.](figures/stats_gallery_latency.png)

## Scaling fit

`plot_scaling_loglog(sizes, times, ax=None, *, title=...)` plots measured
workload sizes against durations on log-log axes and overlays the
least-squares power law of `src/quadmath/stats/statistics.py::scaling_fit` — the slope is
the empirical complexity exponent $\beta_1$ of \eqref{eq:stat-scaling}.
On log-log axes a power law is a straight line, so the panel shows at a
glance whether a lattice routine scales linearly, quadratically, or worse,
and how tightly the data follow the law ($r^2$).  The gallery panel fits a
synthetic sweep with known ground truth: durations $t \approx 3\,n^{1.35}\,(1 + \varepsilon)$
with $\varepsilon \sim \mathcal{N}(0, 0.02^{2})$ relative noise over sizes
$n \in \{8, 16, 32, 64, 128, 256\}$, drawn from the same fixed-seed generator
(times rounded to four decimals for byte stability), so the fit should
recover the planted exponent 1.35.

![**Log-log scaling fit.** Workload sizes $n \in \{8, 16, 32, 64, 128, 256\}$ (dimensionless units) against synthetic durations $t \approx 3\,n^{1.35}\,(1 + \varepsilon)$ with $\varepsilon \sim \mathcal{N}(0, 0.02^{2})$ relative noise, times rounded to four decimals, all drawn from the fixed-seed `numpy.random.default_rng(32)` generator; the least-squares power law from `scaling_fit` is overlaid by `plot_scaling_loglog`, and its fitted slope — annotated in the legend — should recover the planted exponent 1.35, the empirical complexity exponent $\beta_1$ of \eqref{eq:stat-scaling}.](figures/stats_gallery_scaling.png)

## Confidence intervals

`plot_ci_bars(labels, means, lows, highs, ax=None, *, title=...)` renders
the supplied estimates and interval endpoints. It does not calculate
confidence levels or hypothesis tests. The gallery uses three API-name
labels with synthetic estimates and seeded arbitrary half-widths on
$[0.05,0.25]$, not measured timings or bootstrap intervals.

In a real analysis, callers may supply bootstrap bounds from
[the statistical module](17_benchmarks_statistics.md). Overlap of two
individual confidence intervals is not equivalent to a test of their
difference; the relevant comparison depends on sampling design, dependence,
and confidence level.

![**Synthetic interval-rendering example.** API-name labels carry illustrative estimates and seeded symmetric half-widths, not measured latency or calculated 95% coverage. `plot_ci_bars` displays the supplied endpoints; neither overlap nor non-overlap substitutes for a specified test of differences.](figures/stats_gallery_ci.png)

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
timing data the ECDF displays the empirical fraction at or below a duration.
Quantiles can be read through a chosen inverse-ECDF convention; NumPy's
interpolated percentile summaries need not sit exactly on its step heights.
The gallery panel draws the ECDF of the same
256-draw clipped lognormal latency sample as the histogram panel above, so
the two panels read as one distribution in two renderings.

![**Empirical cumulative distribution of the synthetic latency sample.** Step-function ECDF of the same 256-draw clipped lognormal sample as the histogram, seed 32. It displays sample fractions, not a future-latency guarantee; interpolated percentile conventions may differ from inverse-step quantiles.](figures/stats_gallery_ecdf.png)

## Reproducibility and test contract

- The gallery function in `src/quadmath/viz/vis_stats.py` writes exactly the four files
  listed above, in that order, into the output directory (creating it when
  missing) and returns their paths.  Every panel is fully seeded — the
  script fixes `seed=32` — and the PNGs carry no timestamp metadata, so
  re-renders are byte-identical.
- Tests: `tests/unit/viz/test_vis_stats.py` pins the contract with deterministic
  assertions, no mocks, and no pixel diffs — artist placement on
  caller-provided axes, input-agnostic behavior over plain arrays, and the
  invariants of each primitive (monotone non-decreasing ECDF steps, CI
  whisker endpoints matching the lows/highs inputs).  Consistent with
  `17_benchmarks_statistics.md`, no test asserts wall-clock timing
  values.
- Figure regeneration:
  `uv run --python 3.13 python quadmath/scripts/stats_gallery.py` (the script sets
  `MPLBACKEND=Agg` itself); stdout is exactly the four written paths, the
  same manifest contract as `quadmath/scripts/lattice_gallery.py`.
- Markdown: `uv run --python 3.13 python quadmath/scripts/validate_markdown.py`.

## Cross-references

- The measured quantities behind every panel: `17_benchmarks_statistics.md`
  (`src/quadmath/stats/benchmarks.py`, `src/quadmath/stats/statistics.py`).
- The sibling gallery of the lattice layer, same thin-orchestrator and
  determinism contract: `16_lattice_gallery.md` (`src/quadmath/viz/vis_lattice.py`).