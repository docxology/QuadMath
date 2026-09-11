#!/usr/bin/env python3
"""Learning evaluation figure (thin orchestrator).

Constructs a deterministic synthetic linear-regression problem, fits
``GradientDescentTrainer`` from ``src/quadmath/learn/learning_eval.py``, and
delegates rendering to ``src/quadmath/viz/plots.py::plot_loss_history`` per the
thin-orchestrator contract in ``quadmath/scripts/AGENTS.md``.  The builder's
``save=True`` path writes to the fixed name ``loss_history.png``, but the
manuscript pins this figure as ``learn_loss_history.png``; so the builder is
called with ``save=False`` and the open figure is saved here under the pinned
name.  Sets a headless backend and fixed seeds throughout; writes
``learn_loss_history.png`` under ``quadmath/output/figures/`` and prints the
output path (the stdout path line is the ``make_all_figures`` manifest
contract -- only path lines on stdout).
"""
from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def synthetic_linear_data(
    n_samples: int = 40, n_features: int = 3, seed: int = 0
):
    """Build a well-conditioned synthetic linear problem, fixed seed.

    Returns ``(features, target, true_coef)``: the design matrix is standard
    normal, the true coefficients and intercept are fixed constants, and the
    target adds a small fixed-seed Gaussian observation noise.
    """
    import numpy as np  # noqa: WPS433

    rng = np.random.default_rng(seed)
    features = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    true_coef = np.array([1.5, -2.0, 0.75])
    intercept = 0.5
    target = features @ true_coef + intercept + rng.normal(
        loc=0.0, scale=0.05, size=n_samples
    )
    return features, target, true_coef


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    import matplotlib.pyplot as plt  # noqa: WPS433

    from quadmath.learn.learning_eval import GradientDescentTrainer  # noqa: WPS433
    from quadmath.paths import get_figure_dir  # noqa: WPS433
    from quadmath.viz.plots import plot_loss_history  # noqa: WPS433

    features, target, _ = synthetic_linear_data(n_samples=40, n_features=3, seed=0)
    trainer = GradientDescentTrainer(lr=0.05, max_iters=300).fit(features, target)

    plot_loss_history(trainer.loss_history, save=False)
    fig = plt.gcf()
    out_path = f"{get_figure_dir()}/learn_loss_history.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
