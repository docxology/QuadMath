import os

import pytest

import quadmath.viz.visualize as visualize
from quadmath.viz.visualize import plot_ivm_neighbors, plot_partition_tetrahedron, animate_discrete_path
from quadmath.core.quadray import Quadray
from quadmath.optimize.discrete_variational import discrete_ivm_descent


@pytest.fixture(autouse=True)
def _isolate_output_dirs(tmp_path, monkeypatch):
    """Redirect figure/data output to tmp dirs so tests never touch (or
    delete) the shared quadmath/output/ tree that the manuscript references."""
    fig_dir = tmp_path / "figures"
    data_dir = tmp_path / "data"
    fig_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(visualize, "get_figure_dir", lambda: str(fig_dir))
    monkeypatch.setattr(visualize, "get_data_dir", lambda: str(data_dir))


def test_plot_ivm_neighbors_saves_file():
    path = plot_ivm_neighbors(save=True)
    assert os.path.isfile(path)



def test_plot_ivm_neighbors_no_save():
    path = plot_ivm_neighbors(save=False)
    assert path == ""


def test_plot_partition_tetrahedron_saves_file():
    mu = (2, 1, 1, 0)
    s = (1, 2, 1, 0)
    a = (1, 1, 2, 0)
    psi = (2, 2, 1, 1)
    path = plot_partition_tetrahedron(mu, s, a, psi, save=True)
    assert os.path.isfile(path)



def test_plot_partition_tetrahedron_no_save():
    mu = (2, 1, 1, 0)
    s = (1, 2, 1, 0)
    a = (1, 1, 2, 0)
    psi = (2, 2, 1, 1)
    path = plot_partition_tetrahedron(mu, s, a, psi, save=False)
    assert path == ""


def test_animate_discrete_path_saves_file():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    dpath = discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=10)
    out = animate_discrete_path(dpath, save=True)
    assert os.path.isfile(out)



def test_animate_discrete_path_no_save():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    dpath = discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=3)
    out = animate_discrete_path(dpath, save=False)
    assert out == ""
