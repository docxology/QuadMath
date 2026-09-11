"""Tests for the statistics visualization gallery (src/vis_stats.py).

All assertions are deterministic and pixel-free: artist placement on
caller-provided or freshly created axes, exact fit/step values on small
known inputs, and byte-identical PNG re-renders for a fixed seed.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import paths as paths_module
import stats_gallery
from vis_stats import (
    GALLERY_FILES,
    gallery,
    plot_ci_bars,
    plot_ecdf,
    plot_latency_hist,
    plot_scaling_loglog,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure a test created (headless Agg via conftest)."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_latency_hist
# ---------------------------------------------------------------------------


def test_latency_hist_creates_figure_with_mean_line():
    before = plt.get_fignums()
    ax = plot_latency_hist(np.array([1.0, 2.0, 2.0, 3.0, 4.0]), bins=10)
    assert isinstance(ax, plt.Axes)
    assert plt.get_fignums() == before + [ax.figure.number]
    assert ax.get_title() == "Latency distribution"
    assert len(ax.patches) == 10
    assert ax.get_xlabel() == "latency" and ax.get_ylabel() == "count"
    (mean_line,) = ax.lines
    assert mean_line.get_linestyle() == "--"
    assert 2.4 in np.asarray(mean_line.get_xdata(), dtype=float)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("mean" in label for label in legend_labels)


def test_latency_hist_draws_on_caller_axes():
    fig = plt.figure()
    caller_ax = fig.add_subplot(1, 1, 1)
    before = plt.get_fignums()
    ax = plot_latency_hist(np.array([0.5, 1.5]), ax=caller_ax, title="custom title")
    assert ax is caller_ax
    assert plt.get_fignums() == before  # no new figure created
    assert ax.get_title() == "custom title"


# ---------------------------------------------------------------------------
# plot_scaling_loglog
# ---------------------------------------------------------------------------


def test_scaling_loglog_default_axes_and_fit_slope():
    sizes = np.array([8.0, 16.0, 32.0])
    times = np.array([10.0, 30.0, 95.0])
    ax = plot_scaling_loglog(sizes, times)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert ax.get_title() == "Scaling (log-log)"
    (expected_slope, _) = np.polyfit(np.log(sizes), np.log(times), 1)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("slope" in label for label in legend_labels)
    assert any(f"slope = {expected_slope:.3f}" in label for label in legend_labels)


def test_scaling_loglog_draws_on_caller_axes():
    fig = plt.figure()
    caller_ax = fig.add_subplot(1, 1, 1)
    before = plt.get_fignums()
    ax = plot_scaling_loglog(
        np.array([1.0, 2.0]), np.array([1.0, 4.0]), ax=caller_ax, title="power law"
    )
    assert ax is caller_ax
    assert plt.get_fignums() == before
    assert ax.get_title() == "power law"
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"


# ---------------------------------------------------------------------------
# plot_ci_bars
# ---------------------------------------------------------------------------


def test_ci_bars_default_axes_and_rotated_labels():
    labels = ["a", "bb", "ccc"]
    means = np.array([1.0, 2.0, 3.0])
    lows = np.array([0.9, 1.8, 2.7])
    highs = np.array([1.1, 2.2, 3.3])
    ax = plot_ci_bars(labels, means, lows, highs)
    assert ax.get_title() == "Estimates with CI"
    assert ax.get_ylabel() == "estimate"
    tick_labels = [text.get_text() for text in ax.get_xticklabels()]
    assert tick_labels == labels
    for text in ax.get_xticklabels():
        assert text.get_rotation() == 30.0
    assert len(ax.containers) == 1
    # capsize > 0 draws cap lines above and below each estimate; the cap
    # Line2D artists live in the ErrorbarContainer, not in ax.lines.
    assert len(ax.lines) >= len(labels)
    # Caps render as Line2Ds (legacy) or one LineCollection (modern
    # matplotlib); either way the container carries them.
    assert len(ax.containers[0][2]) >= 1


def test_ci_bars_draws_on_caller_axes():
    fig = plt.figure()
    caller_ax = fig.add_subplot(1, 1, 1)
    before = plt.get_fignums()
    ax = plot_ci_bars(
        ["x"], np.array([1.0]), np.array([0.9]), np.array([1.1]),
        ax=caller_ax, title="with CI",
    )
    assert ax is caller_ax
    assert plt.get_fignums() == before
    assert ax.get_title() == "with CI"


def test_ci_bars_rejects_mismatched_lengths():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    with pytest.raises(ValueError, match="equal length"):
        plot_ci_bars(["a", "b"], np.array([1.0, 2.0]), np.array([0.9, 1.8]), np.array([1.1]))
    with pytest.raises(ValueError, match="equal length"):
        plot_ci_bars(["a", "b"], np.array([1.0, 2.0]), np.array([0.9]), np.array([1.1, 2.2]))
    with pytest.raises(ValueError, match="equal length"):
        plot_ci_bars(["a"], np.array([1.0, 2.0]), np.array([0.9, 1.8]), np.array([1.1, 2.2]))


# ---------------------------------------------------------------------------
# plot_ecdf
# ---------------------------------------------------------------------------


def test_ecdf_step_values_on_known_sample():
    ax = plot_ecdf(np.array([3.0, 1.0, 2.0, 2.0]))
    assert ax.get_title() == "Empirical CDF"
    (step_line,) = ax.lines
    np.testing.assert_allclose(step_line.get_xdata(), [1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(step_line.get_ydata(), [0.25, 0.5, 0.75, 1.0])
    assert ax.get_xlabel() == "value" and ax.get_ylabel() == "P(X <= x)"


def test_ecdf_draws_on_caller_axes():
    fig = plt.figure()
    caller_ax = fig.add_subplot(1, 1, 1)
    before = plt.get_fignums()
    ax = plot_ecdf(np.array([1.0]), ax=caller_ax, title="sample ECDF")
    assert ax is caller_ax
    assert plt.get_fignums() == before
    assert ax.get_title() == "sample ECDF"


# ---------------------------------------------------------------------------
# gallery: composition, determinism, reproducibility
# ---------------------------------------------------------------------------


def test_gallery_writes_reproducible_figures(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    paths1 = stats_gallery.main(str(run1))
    paths2 = stats_gallery.main(str(run2))
    assert [os.path.basename(p) for p in paths1] == list(GALLERY_FILES)
    for path in paths1:
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
    for first, second in zip(paths1, paths2):
        with open(first, "rb") as fh_a, open(second, "rb") as fh_b:
            assert fh_a.read() == fh_b.read()
    # Every figure is closed after each run.
    assert plt.get_fignums() == []


def test_gallery_existing_dir_rerun_identical(tmp_path):
    run_dir = tmp_path / "shared"
    paths_a = stats_gallery.main(str(run_dir))  # creates the directory
    paths_b = stats_gallery.main(str(run_dir))  # reuses the existing directory
    assert paths_a == paths_b
    for first, second in zip(paths_a, paths_b):
        with open(first, "rb") as fh_a, open(second, "rb") as fh_b:
            assert fh_a.read() == fh_b.read()


def test_gallery_module_entrypoint_matches_script(tmp_path):
    paths = gallery(str(tmp_path / "direct"), seed=32)
    assert [os.path.basename(p) for p in paths] == list(GALLERY_FILES)
    for path in paths:
        assert os.path.isfile(path)


# ---------------------------------------------------------------------------
# thin script contract (quadmath/scripts/stats_gallery.py)
# ---------------------------------------------------------------------------


def test_stats_gallery_script_prints_paths(tmp_path, monkeypatch, capsys):
    figures_dir = tmp_path / "figures"
    monkeypatch.setattr(paths_module, "get_figure_dir", lambda: str(figures_dir))
    assert stats_gallery.GALLERY_FILES == GALLERY_FILES  # manifest consistency
    written = stats_gallery.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines == written
    assert [os.path.basename(line) for line in lines] == list(GALLERY_FILES)
    for line in lines:
        assert os.path.isabs(line)
        assert os.path.isfile(line)
        assert os.path.getsize(line) > 0
