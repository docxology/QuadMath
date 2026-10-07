from __future__ import annotations

import os
from fractions import Fraction

import matplotlib.pyplot as plt
import pytest
from matplotlib import animation
from matplotlib.figure import Figure

import quadmath.viz.animations as animations
import quadmath.viz.plots as plots
import quadmath.viz.vis_lattice as vis_lattice
import quadmath.viz.vis_stats as vis_stats
import quadmath.viz.visualize as visualize
from quadmath.core.quadray import Quadray
from quadmath.optimize.discrete_variational import discrete_ivm_descent
from quadmath.optimize.nelder_mead_quadray import SimplexState


@pytest.fixture(autouse=True)
def _isolate_output_dirs(tmp_path, monkeypatch):
    fig_dir = tmp_path / "figures"
    data_dir = tmp_path / "data"
    fig_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(visualize, "get_figure_dir", lambda: str(fig_dir))
    monkeypatch.setattr(visualize, "get_data_dir", lambda: str(data_dir))
    monkeypatch.setattr(plots, "get_figure_dir", lambda: str(fig_dir))
    plt.close("all")
    yield
    plt.close("all")


def _descent_path():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    return discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=3)


def _simplex_state() -> SimplexState:
    return SimplexState(
        vertices=[],
        values=[],
        volume=Fraction(1),
        history=[],
        best_values=[1.0, 0.5, 0.25],
        worst_values=[2.0, 1.0, 0.75],
        spreads=[1.0, 0.5, 0.5],
        volumes=[Fraction(1), Fraction(2), Fraction(2)],
    )


def _unit_quat(deg: float) -> tuple[float, float, float, float]:
    import math

    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


PUBLIC_FIGURE_CALLS = {
    "plot_ivm_neighbors": lambda save: visualize.plot_ivm_neighbors(save=save),
    "plot_partition_tetrahedron": lambda save: visualize.plot_partition_tetrahedron(
        (2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 2, 0), (2, 2, 1, 1), save=save
    ),
    "animate_simplex": lambda save: visualize.animate_simplex(
        [[Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(0, 0, 0, 1)]], save=save
    ),
    "animate_discrete_path": lambda save: visualize.animate_discrete_path(_descent_path(), save=save),
    "plot_simplex_trace": lambda save: visualize.plot_simplex_trace(_simplex_state(), save=save),
    "plot_loss_history": lambda save: plots.plot_loss_history([1.0, 0.5, 0.25], save=save),
    "plot_shell_growth": lambda save: plots.plot_shell_growth(3, save=save),
    "plot_error_histogram": lambda save: plots.plot_error_histogram([0.1, 0.2, 0.15, 0.05, 0.2], bins=5, save=save),
    "plot_lattice_shell_3d": lambda save: plots.plot_lattice_shell_3d(1, save=save),
    "plot_slerp_path": lambda save: plots.plot_slerp_path(_unit_quat(0.0), _unit_quat(90.0), save=save),
}


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("name", sorted(PUBLIC_FIGURE_CALLS))
def test_public_figure_function_leaves_no_open_figure(name, save):
    PUBLIC_FIGURE_CALLS[name](save)
    assert plt.get_fignums() == []


@pytest.fixture
def failing_save(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(Figure, "savefig", boom)
    monkeypatch.setattr(animation.FuncAnimation, "save", boom)


@pytest.mark.parametrize("name", sorted(PUBLIC_FIGURE_CALLS))
def test_public_figure_function_closes_figure_when_save_fails(name, failing_save):
    with pytest.raises(RuntimeError, match="save failed"):
        PUBLIC_FIGURE_CALLS[name](True)
    assert plt.get_fignums() == []


GALLERY_CALLS = {
    "vis_lattice.gallery": lambda out: vis_lattice.gallery(out),
    "vis_stats.gallery": lambda out: vis_stats.gallery(out),
    "frames_strip": lambda out: animations.frames_strip(
        animations.lattice_frames(shells=2, n=4), out_path=str(out) + "/strip.png"
    ),
}


@pytest.mark.parametrize("name", sorted(GALLERY_CALLS))
def test_gallery_leaves_no_open_figure(name, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    GALLERY_CALLS[name](str(out))
    assert plt.get_fignums() == []


@pytest.mark.parametrize("name", sorted(GALLERY_CALLS))
def test_gallery_closes_figure_when_save_fails(name, tmp_path, failing_save):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match="save failed"):
        GALLERY_CALLS[name](str(out))
    assert plt.get_fignums() == []


def test_successful_save_leaves_only_final_files():
    visualize.plot_ivm_neighbors(save=True)
    assert sorted(os.listdir(visualize.get_figure_dir())) == ["ivm_neighbors.png"]
    assert sorted(os.listdir(visualize.get_data_dir())) == ["ivm_neighbors_data.csv", "ivm_neighbors_data.npz"]


def test_failed_save_leaves_no_partial_target(monkeypatch):
    def partial_then_fail(self, fname, *args, **kwargs):
        with open(fname, "wb") as fh:
            fh.write(b"partial")
        raise RuntimeError("save failed")

    monkeypatch.setattr(Figure, "savefig", partial_then_fail)
    with pytest.raises(RuntimeError, match="save failed"):
        visualize.plot_ivm_neighbors(save=True)
    assert os.listdir(visualize.get_figure_dir()) == []
