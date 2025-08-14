import os

from nelder_mead_quadray import nelder_mead_quadray
from visualize import animate_simplex, plot_simplex_trace
from quadray import Quadray


def test_simplex_animation_saves_file():
    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [Quadray(5, 0, 0, 0), Quadray(4, 1, 0, 0), Quadray(0, 4, 1, 0), Quadray(1, 1, 1, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=8)
    path = animate_simplex(state.history, save=True)
    assert os.path.isfile(path)
    os.remove(path)


def test_simplex_animation_no_save():
    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(1, 1, 0, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=3)
    path = animate_simplex(state.history, save=False)
    assert path == ""


def test_on_step_callback_invoked():
    def f(q: Quadray) -> float:
        return (q.a) ** 2 + (q.b) ** 2 + (q.c) ** 2 + (q.d) ** 2

    steps = []

    def on_step(verts):
        steps.append(len(verts))

    initial = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(1, 1, 0, 0)]
    _ = nelder_mead_quadray(f, initial, max_iter=2, on_step=on_step)
    assert len(steps) >= 1


def test_simplex_trace_saves_file():
    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [Quadray(5, 0, 0, 0), Quadray(4, 1, 0, 0), Quadray(0, 4, 1, 0), Quadray(1, 1, 1, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=6)
    trace_png = plot_simplex_trace(state, save=True)
    assert os.path.isfile(trace_png)
    os.remove(trace_png)


def test_simplex_trace_no_save():
    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [Quadray(2, 0, 0, 0), Quadray(1, 1, 0, 0), Quadray(0, 2, 0, 0), Quadray(0, 1, 1, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=2)
    trace_png = plot_simplex_trace(state, save=False)
    assert trace_png == ""


def test_nelder_mead_zero_iter_diagnostics():
    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [Quadray(2, 0, 0, 0), Quadray(1, 1, 0, 0), Quadray(0, 2, 0, 0), Quadray(0, 1, 1, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=0)
    # No iterations: single diagnostic entry and single history snapshot
    assert len(state.volumes) == 1
    assert len(state.best_values) == 1
    assert len(state.worst_values) == 1
    assert len(state.spreads) == 1
    assert len(state.history) == 1
