import os
from typing import List

import pytest

import quadmath.viz.visualize as visualize
from quadmath.optimize.nelder_mead_quadray import nelder_mead_quadray
from quadmath.viz.visualize import animate_simplex, plot_simplex_trace
from quadmath.core.quadray import Quadray, DEFAULT_EMBEDDING, to_xyz


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


def test_simplex_animation_saves_file():
    def f(q: Quadray) -> float:
        return (q.a - 1) ** 2 + (q.b - 0) ** 2 + (q.c - 0) ** 2 + (q.d - 0) ** 2

    initial = [Quadray(5, 0, 0, 0), Quadray(4, 1, 0, 0), Quadray(0, 4, 1, 0), Quadray(1, 1, 1, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=8)
    path = animate_simplex(state.history, save=True)
    assert os.path.isfile(path)


def test_nelder_mead_reaches_lattice_optimum():
    # Convex bowl whose continuous minimum sits at (1, -0.5, 0.25) under the
    # default embedding; the true lattice argmin (verified by enumeration over
    # the neighborhood) is Quadray(1, 0, 1, 1) with f = 0.8125.
    def f(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        return (x - 1.0) ** 2 + (y + 0.5) ** 2 + (z - 0.25) ** 2

    initial = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0),
               Quadray(0, 0, 1, 0), Quadray(1, 1, 0, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=100)
    assert state.values[0] == pytest.approx(0.8125)
    assert state.vertices[0] == Quadray(1, 0, 1, 1)




def test_nelder_mead_outside_contraction_rejection():
    # Covers the classical NM path where the outside-contraction candidate is
    # worse than the reflection (spike at (12,0,0,0)), so the reflection is
    # kept instead.
    def f(q: Quadray) -> float:
        base = (q.a - 11) ** 2 + q.b ** 2 + q.c ** 2 + q.d ** 2
        return base + (50 if q.as_tuple() == (12, 0, 0, 0) else 0)

    initial = [Quadray(11, 0, 0, 0), Quadray(10, 1, 0, 0),
               Quadray(10, 0, 1, 0), Quadray(6, 0, 0, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=3)
    assert state.values[0] == 0.0
    assert state.vertices[0] == Quadray(11, 0, 0, 0)


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


def _distance_objective(target: Quadray):
    """Squared Euclidean distance (default embedding) to a lattice target."""
    tx, ty, tz = to_xyz(target, DEFAULT_EMBEDDING)

    def f(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        return (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2

    return f


def test_nelder_mead_degenerate_line_escapes_confinement():
    # Collinear initial simplex along the (1,1,1,0) direction: every classical
    # NM move is an affine combination, so without the CVP-style restart the
    # search stays confined to that line and terminates at the line optimum
    # (value 3.0), never reaching the true lattice argmin Quadray(4,1,0,0)
    # (verified by enumeration) with value 0.0.
    target = Quadray(4, 1, 0, 0)
    initial = [Quadray(0, 0, 0, 0), Quadray(1, 1, 1, 0), Quadray(2, 2, 2, 0), Quadray(3, 3, 3, 0)]
    steps: List[List[Quadray]] = []
    state = nelder_mead_quadray(_distance_objective(target), initial, max_iter=200, on_step=steps.append)
    assert state.vertices[0] == target
    assert state.values[0] == 0.0
    # The restart must have left the degenerate line: non-zero volumes appear.
    assert any(v != 0 for v in state.volumes)
    # The on_step callback observes the restart snapshots too.
    assert len(steps) >= 1


def test_nelder_mead_degenerate_plane_escapes_confinement():
    # Coplanar (but not collinear) initial simplex in the c=d=0 plane; the
    # true lattice argmin Quadray(1,0,3,0) (verified by enumeration) lies off
    # the plane, so only a restart can reach it.
    target = Quadray(1, 0, 3, 0)
    initial = [Quadray(0, 0, 0, 0), Quadray(2, 0, 0, 0), Quadray(0, 2, 0, 0), Quadray(2, 2, 0, 0)]
    state = nelder_mead_quadray(_distance_objective(target), initial, max_iter=200)
    assert state.vertices[0] == target
    assert state.values[0] == 0.0
    assert any(v != 0 for v in state.volumes)


def test_nelder_mead_collapsed_local_optimum_terminates():
    # A simplex collapsed exactly at the objective's lattice minimum: all
    # axial neighbor probes are worse, so the convergence branch returns
    # immediately without any restart.
    target = Quadray(1, 0, 0, 0)
    initial = [Quadray(1, 0, 0, 0)] * 4
    state = nelder_mead_quadray(_distance_objective(target), initial, max_iter=5)
    assert len(state.history) == 1
    assert state.volume == 0
    assert all(v == target for v in state.vertices)
    assert state.values[0] == 0.0


def test_nelder_mead_probes_escape_premature_collapse():
    # The simplex-animation demo objective: without the axial probe the NM
    # run collapsed at a premature local lattice point (value 3.5) although
    # the true minimum 0.6 at Quadray(2,0,0,0) — xyz (2,2,2) under the
    # default embedding — is reachable. The probe restart escapes the
    # premature collapse and converges to the optimum within the demo budget.
    def f(q: Quadray) -> float:
        x, y, z = to_xyz(q, DEFAULT_EMBEDDING)
        obj = (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2
        if x < 0 and y < 0:
            obj += 5.0
        if abs(x) > 4 or abs(y) > 4 or abs(z) > 4:
            obj += 10.0
        return obj + 0.1 * abs(x + y + z)

    initial = [Quadray(5, 0, 0, 0), Quadray(4, 1, 0, 0), Quadray(0, 4, 1, 0), Quadray(1, 1, 1, 0)]
    state = nelder_mead_quadray(f, initial, max_iter=20)
    assert state.vertices[0] == Quadray(2, 0, 0, 0)
    assert abs(state.values[0] - 0.6) < 1e-12
    assert state.volume == 0
