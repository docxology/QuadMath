"""Tests for the standalone figure builders (quadmath/viz/plots.py).

Each save path is redirected into a temporary directory by monkeypatching
``quadmath.viz.plots.get_figure_dir``; determinism is verified by re-rendering
into separate directories and byte-comparing the PNGs, mirroring the gallery
reproducibility tests. Headless Agg backend is set in tests/conftest.py.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import quadmath.viz.plots as plots_module
from quadmath.core.quadray import Quadray
from quadmath.lattice.ivm_field import shell_cardinalities, shell_sites
from quadmath.viz.plots import (
    plot_error_histogram,
    plot_lattice_shell_3d,
    plot_loss_history,
    plot_shell_growth,
    plot_slerp_path,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure a test created (headless Agg via conftest)."""
    yield
    plt.close("all")


@pytest.fixture()
def _figures_dir(tmp_path, monkeypatch):
    """Redirect quadmath.viz.plots figure output into a fresh tmp directory."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(figures_dir))
    return figures_dir


# ---------------------------------------------------------------------------
# plot_loss_history
# ---------------------------------------------------------------------------


def test_loss_history_saves_png_and_returns_path(_figures_dir):
    outpath = plot_loss_history([1.0, 0.6, 0.4, 0.3, 0.25])
    assert outpath == str(_figures_dir / "loss_history.png")
    assert os.path.isfile(outpath)
    assert os.path.getsize(outpath) > 0


def test_loss_history_accepts_numpy_and_iterator_inputs(_figures_dir):
    path_a = plot_loss_history(np.array([1.0, 0.5, 0.2]))
    path_b = plot_loss_history(iter([1.0, 0.5, 0.2]))
    assert os.path.isfile(path_a)
    assert os.path.isfile(path_b)
    with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
        assert fh_a.read() == fh_b.read()


def test_loss_history_is_byte_reproducible(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(tmp_path / "a"))
    path_a = plot_loss_history([2.0, 1.0, 0.7, 0.5])
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(tmp_path / "b"))
    path_b = plot_loss_history([2.0, 1.0, 0.7, 0.5])
    with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
        assert fh_a.read() == fh_b.read()


def test_loss_history_styling_and_labels():
    plot_loss_history([1.0, 0.5], save=False)
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Training loss history"
    assert ax.get_xlabel() == "iteration"
    assert ax.get_ylabel() == "loss"
    assert len(ax.lines) == 1
    assert ax.lines[0].get_xdata().tolist() == [0, 1]


def test_loss_history_rejects_empty_sequence():
    with pytest.raises(ValueError):
        plot_loss_history([])


# ---------------------------------------------------------------------------
# plot_shell_growth
# ---------------------------------------------------------------------------


def test_shell_growth_saves_png_and_returns_path(_figures_dir):
    outpath = plot_shell_growth(3)
    assert outpath == str(_figures_dir / "shell_growth.png")
    assert os.path.isfile(outpath)
    assert os.path.getsize(outpath) > 0


def test_shell_growth_plots_cardinalities_and_uses_default_k_max():
    plot_shell_growth(save=False)
    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "shell index k"
    assert ax.get_ylabel() == "sites in shell"
    assert list(ax.lines[0].get_ydata()) == [float(c) for c in shell_cardinalities(6)]


def test_shell_growth_rejects_negative_k_max():
    with pytest.raises(ValueError):
        plot_shell_growth(-1)


# ---------------------------------------------------------------------------
# plot_error_histogram
# ---------------------------------------------------------------------------


def test_error_histogram_saves_png_and_returns_path(_figures_dir):
    outpath = plot_error_histogram([0.1, 0.2, 0.15, 0.05, 0.2], bins=5)
    assert outpath == str(_figures_dir / "error_histogram.png")
    assert os.path.isfile(outpath)
    assert os.path.getsize(outpath) > 0


def test_error_histogram_draws_mean_line():
    errors = [1.0, 2.0, 3.0]
    plot_error_histogram(errors, bins=3, save=False)
    ax = plt.gcf().axes[0]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_xdata()[0] == pytest.approx(2.0)
    assert ax.get_xlabel() == "error" and ax.get_ylabel() == "count"


def test_error_histogram_rejects_empty_sequence():
    with pytest.raises(ValueError):
        plot_error_histogram([])


def test_error_histogram_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        plot_error_histogram([0.1, float("nan"), 0.3])
    with pytest.raises(ValueError):
        plot_error_histogram([0.1, float("inf"), 0.3])


def test_error_histogram_rejects_nonpositive_bins():
    with pytest.raises(ValueError):
        plot_error_histogram([0.1, 0.2], bins=0)


# ---------------------------------------------------------------------------
# plot_lattice_shell_3d
# ---------------------------------------------------------------------------


def test_lattice_shell_3d_saves_png_and_returns_path(_figures_dir):
    outpath = plot_lattice_shell_3d(1)
    assert outpath == str(_figures_dir / "lattice_shell_3d.png")
    assert os.path.isfile(outpath)
    assert os.path.getsize(outpath) > 0


def test_lattice_shell_3d_scatters_all_shell_sites():
    plot_lattice_shell_3d(1, save=False)
    ax = plt.gcf().axes[0]
    assert ax.get_title() == f"IVM lattice shell k=1 ({len(shell_sites(1))} sites)"
    assert len(ax.collections[0]._offsets3d[0]) == len(shell_sites(1))


def test_lattice_shell_3d_is_byte_reproducible(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(tmp_path / "a"))
    path_a = plot_lattice_shell_3d(1)
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(tmp_path / "b"))
    path_b = plot_lattice_shell_3d(1)
    with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
        assert fh_a.read() == fh_b.read()


def test_lattice_shell_3d_rejects_negative_shell():
    with pytest.raises(ValueError):
        plot_lattice_shell_3d(-2)


# ---------------------------------------------------------------------------
# plot_slerp_path
# ---------------------------------------------------------------------------


#: Identity quaternion and the unit quaternion for a right-handed pi/2
#: rotation about z (the half-angle form rotate_about_axis builds).
_Q0 = (1.0, 0.0, 0.0, 0.0)
_Q1 = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))


def test_slerp_path_saves_png_and_returns_path(_figures_dir):
    outpath = plot_slerp_path(_Q0, _Q1)
    assert outpath == str(_figures_dir / "quaternion_slerp_path.png")
    assert os.path.isfile(outpath)
    assert os.path.getsize(outpath) > 0


def test_slerp_path_traces_default_site_with_exact_endpoints():
    plot_slerp_path(_Q0, _Q1, n_frames=9, save=False)
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Slerp rotation path of site (2, 0, 0, 0)"
    assert len(ax.lines) == 1 and len(ax.collections) == 2
    xs, ys, zs = ax.lines[0].get_data_3d()
    assert len(xs) == 9
    # The default (2, 0, 0, 0) site starts at its embedded position (2, 2, 2)
    # and ends rotated by pi/2 about z at (-2, 2, 2); z stays constant because
    # the interpolated quaternions all rotate about the z axis.
    assert (xs[0], ys[0], zs[0]) == (2.0, 2.0, 2.0)
    assert (xs[-1], ys[-1], zs[-1]) == pytest.approx((-2.0, 2.0, 2.0))
    assert all(z == pytest.approx(2.0) for z in zs)
    assert len(ax.collections[0]._offsets3d[0]) == 1  # start marker
    assert len(ax.collections[1]._offsets3d[0]) == 1  # end marker


def test_slerp_path_accepts_explicit_site():
    plot_slerp_path(_Q0, _Q1, n_frames=5, site=Quadray(1, 0, 0, 0), save=False)
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Slerp rotation path of site (1, 0, 0, 0)"
    xs, ys, zs = ax.lines[0].get_data_3d()
    assert (xs[0], ys[0], zs[0]) == (1.0, 1.0, 1.0)
    assert (xs[-1], ys[-1], zs[-1]) == pytest.approx((-1.0, 1.0, 1.0))


def test_slerp_path_is_byte_reproducible(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(tmp_path / "a"))
    path_a = plot_slerp_path(_Q0, _Q1)
    (tmp_path / "b").mkdir()
    monkeypatch.setattr(plots_module, "get_figure_dir", lambda: str(tmp_path / "b"))
    path_b = plot_slerp_path(_Q0, _Q1)
    with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
        assert fh_a.read() == fh_b.read()


def test_slerp_path_rejects_non_unit_q0():
    with pytest.raises(ValueError):
        plot_slerp_path((2.0, 0.0, 0.0, 0.0), _Q1)


def test_slerp_path_rejects_non_unit_q1():
    with pytest.raises(ValueError):
        plot_slerp_path(_Q0, (1.0, 1.0, 0.0, 0.0))


def test_slerp_path_rejects_too_few_frames():
    with pytest.raises(ValueError):
        plot_slerp_path(_Q0, _Q1, n_frames=1)
