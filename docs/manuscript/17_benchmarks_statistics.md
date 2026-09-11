# Benchmarks and Statistics

## Overview

This section documents the measurement surface behind the preceding
chapters.  It is split across three modules with one job each:
`src/benchmarks.py` measures (a small `perf_counter` timing harness over
the core numerical surfaces), `src/statistics.py` analyzes (resampling
inference, effect sizes, multiplicity correction, and scaling fits, on
standalone numpy), and `src/vis_stats.py` renders (input-agnostic
matplotlib primitives; the figures they produce are collected in
`18_stats_gallery.md`).  The measured workloads are the four surfaces the
manuscript already treats: quadray conversions (`quadray.py`, conventions
in `SPEC.md`), shell enumeration (`omni_numbering.py`, `13_lattice_tooling.md`),
nearest-site search (`lattice_search.py`), and field fitting
(`ivm_field.py`, `11_ivm_field_learning.md`).  Everything here is
deterministic: every random draw comes from a seeded
`numpy.random.default_rng`, and every timing routine is fully specified by
its arguments, so the numbers below reproduce bit-for-bit.

## Timing methodology

`time_callable(fn, *, trials=5, warmup=1)` measures wall-clock duration
with `time.perf_counter`, the highest-resolution monotonic clock available.
It runs one warmup call first and discards it (first-call effects: module
imports, key caches), then times `trials` calls and reports the summary
statistics

\begin{equation}
\label{eq:bench-mean}
\bar{t} \;=\; \frac{1}{T}\sum_{i=1}^{T} t_i ,
\qquad t_m \;=\; \operatorname{median}(t_1,\dots,t_T),
\end{equation}

\begin{equation}
\label{eq:bench-percentile}
t_{95} \;=\; Q_{0.95}(t_1,\dots,t_T),
\end{equation}

where $T$ is the number of trials and $Q_p$ the empirical $p$-quantile.
The median is robust to a single outlier trial; the 95th percentile bounds
the tail that a user would actually feel.  Results are returned as
`BenchRow`, a frozen dataclass with fields `name`, `n`, `trials`, `total_s`,
`mean_s`, `median_s`, `p95_s`, and `ops_per_s`, plus `as_dict()` for plain
data.  Here `n` is the per-call workload size (conversions per call, shell
depth, sites or queries per call), and throughput is

\begin{equation}
\label{eq:bench-ops}
\text{ops\_per\_s} \;=\; \frac{n}{\bar{t}} .
\end{equation}

Four constructors wrap the core surfaces with these defaults:
`bench_conversions(n=200, trials=5)` times round-trip quadray/embedding
conversions, `bench_shell_enumeration(k_max=4, trials=5)` times shell
enumeration through `omni_numbering.generate_shell`, `bench_lattice_search(n_sites=200, queries=50, trials=5)` times
`lattice_search` nearest-site queries over a random ball of sites, and
`bench_field_fit(n_sites=64, trials=3)` times a synthetic field fit through
`IVMField.learn`.  `run_all()` executes all four with the shared defaults
collected in the module constant `BENCH_DEFAULTS` and returns the list of
`BenchRow` records; `summary_table(rows)` renders them as a fixed-width
text table.  The harness measures only; it asserts nothing — timing
distributions belong to analysis, not to test assertions.

## Percentile bootstrap

`bootstrap_ci(x, stat=np.mean, *, iters=2000, seed=0, alpha=0.05)` gives a
confidence interval for any statistic $\hat{\theta}$ by resampling the data
with replacement.  With $n = |x|$ and a seeded
`numpy.random.default_rng(seed)`, iteration $b$ draws resampling indices
$i_{bj} \sim \mathrm{Uniform}\{0,\dots,n-1\}$ in one vectorized call
(`rng.integers(0, n, size=(iters, n))`) and evaluates

\begin{equation}
\label{eq:stat-bootstrap}
\hat{\theta}^{*}_{b} \;=\; \hat{\theta}\big(x_{i_{b1}},\dots,x_{i_{bn}}\big),
\qquad
\mathrm{CI}_{1-\alpha}
\;=\;
\Big[\, Q_{\alpha/2}\big(\hat{\theta}^{*}_{1..B}\big),\;
Q_{1-\alpha/2}\big(\hat{\theta}^{*}_{1..B}\big) \Big],
\end{equation}

with $B$ = `iters`.  The percentile interval needs no distributional
assumption about $\hat{\theta}$ — only that the empirical resampling
distribution approximates its sampling distribution.  All randomness is
consumed from the seeded generator, so the interval is exactly
reproducible.

## Pooled permutation tests

`permutation_test(a, b, *, iters=2000, seed=0, alternative="two-sided")`
tests whether two samples differ in location without assuming a
distribution.  Both samples are pooled; the observed statistic is
$o = \big|\bar{a} - \bar{b}\big|$; each of the `iters` iterations draws
`rng.permutation` of the pool, splits it at $\lvert a\rvert$ into a fake
$a$ and $b$, and recomputes the difference.  The two-sided p-value uses the
add-one convention

\begin{equation}
\label{eq:stat-permutation}
p \;=\;
\frac{1 + \#\big\{ r :\; |d_r| \,\ge\, o \big\}}{\text{iters} + 1},
\end{equation}

where $d_r$ is the difference of permutation $r$.  The $+1$ in numerator
and denominator counts the observed arrangement itself and guarantees
$p \ge 1/(\text{iters}+1) > 0$: a permutation test can never report an
impossible zero p-value, and the estimate is conservative.
`alternative="greater"` and `"less"` use the same convention with signed
comparisons of $d_r$ against the signed observed difference.

## Effect size, multiplicity, and scaling

`cohens_d(a, b)` reports the standardized mean difference

\begin{equation}
\label{eq:stat-cohens}
d \;=\; \frac{\bar{a} - \bar{b}}{s_p},
\qquad
s_p \;=\;
\sqrt{\frac{(n_a - 1)\,s_a^2 + (n_b - 1)\,s_b^2}{n_a + n_b - 2}},
\end{equation}

with $s_a, s_b$ the unbiased sample standard deviations.  A p-value says
whether a difference is distinguishable from noise; $d$ says how large the
difference is, which is the question that matters when comparing benchmark
configurations.

`p_adjust_bonferroni(pvals)` controls the family-wise error rate when $m$
hypotheses are tested at once:

\begin{equation}
\label{eq:stat-bonferroni}
p'_i \;=\; \min\!\big(1,\; m\, p_i\big).
\end{equation}

Each $p'_i$ can be read as a valid p-value at level $\alpha$ for the whole
family — conservative, but assumption-free.

`scaling_fit(sizes, times)` fits ordinary least squares on the log-log
data,

\begin{equation}
\label{eq:stat-scaling}
\log t \;=\; \beta_1 \log n + \beta_0,
\end{equation}

and returns `(slope, intercept, r2)`.  The slope is the empirical
complexity exponent: $\beta_1 \approx 1$ indicates linear work in the
input size $n$, $\beta_1 \approx 2$ quadratic, and the coefficient of
determination $r^2$ measures how well the power law describes the
measurements.

## A reproducible example

Both workhorses above are fully determined by their arguments, so the
following numbers are exact for any reader.  The first call computes the
95% bootstrap confidence interval of the mean of the first seven Fibonacci
numbers; the second asks whether the shifted ranges $1..10$ and $11..20$
differ in location:

```python
import numpy as np

from statistics import bootstrap_ci, permutation_test

x = [2, 3, 5, 8, 13, 21, 34]
lo, hi = bootstrap_ci(x, np.mean, iters=2000, seed=0, alpha=0.05)
# (5.142857142857143, 20.571428571428573)

a = list(range(1, 11))
b = list(range(11, 21))
p = permutation_test(a, b, iters=2000, seed=0, alternative="two-sided")
# 0.0014992503748125937
```

With `bootstrap_ci(x, np.mean, iters=2000, seed=0, alpha=0.05)` the mean of
$x$ is $86/7 \approx 12.285714$ and the 95% percentile bootstrap interval
is $[5.142857142857143,\; 20.571428571428573]$.  Both endpoints are exact
multiples of $1/7$: the mean of any 7-value resample with repetition is a
multiple of $1/7$, so the percentile grid is discrete rather than dense.
With `permutation_test(a, b, iters=2000, seed=0, alternative="two-sided")`
on $a = 1,\dots,10$ and $b = 11,\dots,20$, the observed difference is
$o = |\bar{a} - \bar{b}| = 10$ exactly, and exactly 2 of the 2000 seeded
permutations reach it, giving

\begin{equation}
\label{eq:stat-example-p}
p \;=\; \frac{2 + 1}{2000 + 1} \;=\; \frac{3}{2001}
\;\approx\; 0.00149925 .
\end{equation}

The count of 2 deserves a comment.  Among the
$\binom{20}{10} = 184{,}756$ possible splits of the pool, exactly two
achieve $|d_r| = 10$ — the observed arrangement ($1..10$ against $11..20$)
and its complement — so the expected hit count over 2000 random draws is
only $2000 \cdot 2/184756 \approx 0.02$.  The add-one floor of
\eqref{eq:stat-permutation} is $1/2001 \approx 0.0005$, which is what a
typical seed reports here; the stream seeded with `seed=0` happens to draw
both exact-maximum arrangements (at iterations 70 and 1146 of the 2000), so
the count is 2 and the reported p-value is $3/2001$.  The example
illustrates the convention rather than a typical run: with any other seed
the same call reproduces the floor value, and in either case the test
correctly never reports the impossible $p = 0$.

## Cross-references

- Timing harness, `BenchRow`, and the four benchmark constructors:
  `src/benchmarks.py`.
- Bootstrap, permutation tests, `cohens_d`, `p_adjust_bonferroni`,
  `scaling_fit`: `src/statistics.py`.
- The figure primitives built on these results: `18_stats_gallery.md`.
- Measured surfaces: `quadray.py` (conventions in `SPEC.md`),
  `omni_numbering.py` and `lattice_search.py` (`13_lattice_tooling.md`),
  `ivm_field.py` (`11_ivm_field_learning.md`).
- Sibling gallery of the lattice layer: `16_lattice_gallery.md`.