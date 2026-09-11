#!/usr/bin/env python3
"""Bootstrap-vs-jackknife confidence-interval comparison (thin orchestrator).

Fixed-seed diagnostics example for section ``17_benchmarks_statistics.md``:
two synthetic lattice-error samples with a planted true mean shift.  The
95% percentile bootstrap interval (:func:`quadmath.stats.statistics.bootstrap_ci`,
seeded) and the normal-approximation jackknife interval
(:func:`quadmath.stats.statistics.jackknife_ci`, deterministic) are computed
for the mean of each sample, and all four intervals are drawn in ONE
``plot_ci_bars`` call from ``src/quadmath/viz/vis_stats.py``.  The single
PNG ``stats_ci_comparison.png`` is written under ``quadmath/output/figures/``
and the output path is printed on its own line — the ``make_all_figures``
manifest contract (only path lines on stdout).  Every random draw comes
from ``numpy.random.default_rng`` with the fixed seed below, so the figure
is reproducible byte-for-byte.
"""
from __future__ import annotations

import os
import sys

#: Seed for the synthetic samples and for the bootstrap resampling.
SEED = 17
#: Sample size per lattice-error group.
N_PER_GROUP = 64
#: Planted mean shift between the two lattice-error populations.
TRUE_SHIFT = 0.08
#: Tail probability; the intervals cover 1 - ALPHA.
ALPHA = 0.05
#: Output file name of the comparison figure.
OUTPUT_NAME = "stats_ci_comparison.png"


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    import matplotlib.pyplot as plt  # noqa: WPS433
    import numpy as np  # noqa: WPS433

    from quadmath.paths import get_figure_dir  # noqa: WPS433
    from quadmath.stats.statistics import bootstrap_ci, jackknife_ci  # noqa: WPS433
    from quadmath.viz.vis_stats import plot_ci_bars  # noqa: WPS433

    rng = np.random.default_rng(SEED)
    # Two lattice-error samples: routine A is unbiased (mean 0.0), routine
    # B carries a planted mean shift TRUE_SHIFT; both share scale 0.15.
    sample_a = rng.normal(0.0, 0.15, size=N_PER_GROUP)
    sample_b = rng.normal(TRUE_SHIFT, 0.15, size=N_PER_GROUP)

    labels: list[str] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for name, sample in (("A", sample_a), ("B", sample_b)):
        mean = float(np.mean(sample))
        boot_lo, boot_hi = bootstrap_ci(
            sample, np.mean, iters=2000, seed=SEED, alpha=ALPHA
        )
        jack_lo, jack_hi, _bias = jackknife_ci(sample, np.mean, alpha=ALPHA)
        labels.extend((f"{name} bootstrap", f"{name} jackknife"))
        means.extend((mean, mean))
        lows.extend((float(boot_lo), float(jack_lo)))
        highs.extend((float(boot_hi), float(jack_hi)))

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    plot_ci_bars(
        labels,
        np.array(means, dtype=float),
        np.array(lows, dtype=float),
        np.array(highs, dtype=float),
        ax,
        title="95% CIs for the mean: bootstrap vs jackknife",
    )
    # Reference line at 0 (the unbiased population mean): B's intervals
    # exclude it, A's hug it so tightly that bootstrap lands just below
    # and jackknife just above the line.
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    fig.tight_layout()
    out_path = os.path.join(get_figure_dir(), OUTPUT_NAME)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(os.path.abspath(out_path))


if __name__ == "__main__":
    main()