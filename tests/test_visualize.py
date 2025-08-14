import os

from visualize import plot_ivm_neighbors, plot_partition_tetrahedron, animate_discrete_path
from quadray import Quadray
from discrete_variational import discrete_ivm_descent


def test_plot_ivm_neighbors_saves_file():
    path = plot_ivm_neighbors(save=True)
    assert os.path.isfile(path)
    os.remove(path)


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
    os.remove(path)


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
    os.remove(out)


def test_animate_discrete_path_no_save():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    dpath = discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=3)
    out = animate_discrete_path(dpath, save=False)
    assert out == ""
