from __future__ import annotations

import types

from visualize import _set_axes_equal, animate_discrete_path
from discrete_variational import DiscretePath


class DummyAxis:
    def __init__(self):
        self._calls = []

    # emulate get_* limits
    def get_xlim3d(self):
        return (0.0, 2.0)

    def get_ylim3d(self):
        return (0.0, 4.0)

    def get_zlim3d(self):
        return (0.0, 6.0)

    # record set_* calls to ensure fallback path executed
    def set_xlim3d(self, a, b):
        self._calls.append(("x", a, b))

    def set_ylim3d(self, a, b):
        self._calls.append(("y", a, b))

    def set_zlim3d(self, a, b):
        self._calls.append(("z", a, b))


def test__set_axes_equal_fallback_branch_executed(monkeypatch):
    # Force AttributeError on set_box_aspect by giving an object without it
    ax = DummyAxis()
    # Bind a function that raises to mimic missing attribute path
    def raiser(*args, **kwargs):
        raise AttributeError("no set_box_aspect")

    # Attach set_box_aspect that raises to trigger except path
    ax.set_box_aspect = raiser  # type: ignore[attr-defined]
    _set_axes_equal(ax)
    # Fallback should have set all three axis limits
    kinds = [k for (k, *_rest) in ax._calls]
    assert set(kinds) == {"x", "y", "z"}


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


def test__set_axes_equal_box_aspect_branch_executed():
    class AxisWithBox(DummyAxis):
        def set_box_aspect(self, tup):  # noqa: D401
            # emulate successful box-aspect path
            self._calls.append(("box",) + tuple(tup))

    ax = AxisWithBox()
    _set_axes_equal(ax)
    # Should have recorded box-aspect and not needed to set limits
    assert any(k == "box" for (k, *_rest) in ax._calls)


