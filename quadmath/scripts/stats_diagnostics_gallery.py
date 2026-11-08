#!/usr/bin/env python3
"""Bootstrap-vs-jackknife confidence-interval comparison (thin orchestrator).

Fixed-seed diagnostics example for section ``17_benchmarks_statistics.md``:
two synthetic normal samples with a planted population mean shift. The
95% percentile bootstrap interval (:func:`quadmath.stats.statistics.bootstrap_ci`,
seeded) and the normal-approximation jackknife interval
(:func:`quadmath.stats.statistics.jackknife_ci`, deterministic) are computed
for the mean of each sample, and all four intervals are drawn in ONE
``plot_ci_bars`` call from ``src/quadmath/viz/vis_stats.py``.  The single
PNG ``stats_ci_comparison.png`` is written under ``quadmath/output/figures/``
and the output path is printed on its own line — the ``make_all_figures``
manifest contract (only path lines on stdout).  Every random draw comes
from ``numpy.random.default_rng`` with a fixed seed, so numerical inputs are
reproducible; encoded PNG bytes depend on the rendering environment.
"""
from __future__ import annotations

import os
import sys

#: Seed for the synthetic samples and for the bootstrap resampling.
SEED = 17
#: Sample size per synthetic group.
N_PER_GROUP = 64
#: Planted difference between the generating population means.
TRUE_SHIFT = 0.08
#: Nominal two-sided error level, not a guarantee for one realized interval.
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

    import numpy as np  # noqa: WPS433

    from quadmath.paths import get_figure_dir  # noqa: WPS433
    from quadmath.stats.statistics import bootstrap_ci, jackknife_ci  # noqa: WPS433
    from quadmath.viz.vis_stats import plot_ci_bars  # noqa: WPS433
    from quadmath.viz._common import figure_scope, save_figure  # noqa: WPS433

    rng = np.random.default_rng(SEED)
    # These are simulated groups, not measured errors from package routines.
    # Their generating means are 0 and TRUE_SHIFT, with common scale 0.15.
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

    out_path = os.path.join(get_figure_dir(), OUTPUT_NAME)
    with figure_scope(figsize=(7.2, 5.0)) as fig:
        ax = fig.add_subplot(1, 1, 1)
        plot_ci_bars(
            labels,
            np.array(means, dtype=float),
            np.array(lows, dtype=float),
            np.array(highs, dtype=float),
            ax,
            title=f"Synthetic group means: bootstrap vs jackknife\nNominal {100 * (1 - ALPHA):g}% intervals; n={N_PER_GROUP} per group",
        )
        ax.scatter(
            np.arange(len(labels)), np.repeat([0.0, TRUE_SHIFT], 2),
            marker="x", color="gray", label="generating population mean", zorder=3,
        )
        ax.set_ylabel("synthetic mean (arbitrary units)")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out_path, dpi=160)
    print(os.path.abspath(out_path))


if __name__ == "__main__":
    main()