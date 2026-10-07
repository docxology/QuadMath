from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import quadmath.viz.visualize as visualize
from quadmath.viz._common import set_axes_equal
from quadmath.viz.visualize import animate_discrete_path
from quadmath.optimize.discrete_variational import DiscretePath


@pytest.fixture(autouse=True)
def _isolate_output_dirs(tmp_path, monkeypatch):
    """Redirect figure/data output to tmp dirs so tests never touch the
    shared quadmath/output/ tree that the manuscript references."""
    fig_dir = tmp_path / "figures"
    data_dir = tmp_path / "data"
    fig_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(visualize, "get_figure_dir", lambda: str(fig_dir))
    monkeypatch.setattr(visualize, "get_data_dir", lambda: str(data_dir))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_set_axes_equal_box_aspect_is_proportional_to_data_spans():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0.0, 6.9], [0.0, 1.1], [0.0, 0.23])
    set_axes_equal(ax)
    spans = np.array([hi - lo for lo, hi in (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())])
    box = np.array(ax.get_box_aspect())
    np.testing.assert_allclose(box / box[0], spans / spans[0])


def test_animate_discrete_path_empty_returns_empty_string():
    # path with zero steps should early-return "" when save=False
    empty = DiscretePath(path=[], values=[])  # type: ignore[arg-type]
    out = animate_discrete_path(empty, save=False)
    assert out == ""


def test_animate_discrete_path_empty_save_true_returns_empty_string():
    # path with zero steps and save=True should also return "" via guard
    empty = DiscretePath(path=[], values=[])  # type: ignore[arg-type]
    out = animate_discrete_path(empty, save=True)
    assert out == ""
