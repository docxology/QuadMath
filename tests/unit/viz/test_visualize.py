import os

import numpy as np
import pytest

import quadmath.viz.visualize as visualize
from quadmath.viz.visualize import animate_simplex, plot_ivm_neighbors, plot_partition_tetrahedron, animate_discrete_path
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray
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


PUBLIC_VISUALIZE_NAMES = [
    "plot_ivm_neighbors",
    "animate_simplex",
    "plot_simplex_trace",
    "plot_partition_tetrahedron",
    "animate_discrete_path",
]


def test_visualize_all_lists_exactly_the_public_functions():
    assert visualize.__all__ == PUBLIC_VISUALIZE_NAMES


def test_visualize_star_import_exports_only_public_functions():
    namespace: dict = {}
    exec("from quadmath.viz.visualize import *", namespace)
    exported = sorted(name for name in namespace if not name.startswith("__"))
    assert exported == sorted(PUBLIC_VISUALIZE_NAMES)


def test_viz_package_exports_are_explicit():
    import quadmath.viz as viz

    assert sorted(viz.__all__) == sorted(PUBLIC_VISUALIZE_NAMES + ["vis_lattice", "vis_stats"])
    for name in PUBLIC_VISUALIZE_NAMES:
        assert getattr(viz, name) is getattr(visualize, name)
    for leaked in ("plt", "np", "csv", "os", "animation", "Iterable", "Quadray", "to_xyz", "DEFAULT_EMBEDDING"):
        assert not hasattr(viz, leaked)


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


def _descent_path():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    return discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=3)


def _run_ivm(embedding):
    return plot_ivm_neighbors(embedding=embedding, save=True)


def _run_tetrahedron(embedding):
    return plot_partition_tetrahedron((2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 2, 0), (2, 2, 1, 1), embedding=embedding, save=True)


def _run_simplex(embedding):
    frame = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(0, 0, 0, 1)]
    return animate_simplex([frame], embedding=embedding, save=True)


def _run_path(embedding):
    return animate_discrete_path(_descent_path(), embedding=embedding, save=True)


@pytest.mark.parametrize(
    ("runner", "npz_name"),
    [
        (_run_ivm, "ivm_neighbors_data.npz"),
        (_run_tetrahedron, "partition_tetrahedron_data.npz"),
        (_run_simplex, "simplex_animation_vertices.npz"),
        (_run_path, "discrete_path.npz"),
    ],
)
def test_generator_embedding_matches_list_embedding(runner, npz_name):
    list_rows = [list(row) for row in DEFAULT_EMBEDDING]
    from_list = runner(list_rows)
    list_stored = np.load(os.path.join(visualize.get_data_dir(), npz_name))["embedding"]

    from_generator = runner(list(row) for row in list_rows)
    generator_stored = np.load(os.path.join(visualize.get_data_dir(), npz_name))["embedding"]

    assert from_generator == from_list
    np.testing.assert_array_equal(generator_stored, list_stored)
