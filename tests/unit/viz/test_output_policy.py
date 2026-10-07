"""Output path policy shared by every viz saver.

A path that is absolute or has a directory component is used as given; a bare
file name (or bare directory name for galleries) resolves into the figure
directory.  Companion data files keep their fixed names under
``get_data_dir()``.  Every test here redirects both directories into ``tmp_path``.
"""
from __future__ import annotations

import os
import shutil
from fractions import Fraction

import pytest

import quadmath.viz._common as common
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
    figure_dir = lambda: str(fig_dir)  # noqa: E731
    for module in (animations, plots, vis_lattice, vis_stats, visualize):
        monkeypatch.setattr(module, "get_figure_dir", figure_dir)
    monkeypatch.setattr(visualize, "get_data_dir", lambda: str(data_dir))
    return fig_dir


def _ffmpeg_or_skip() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")


def test_resolve_keeps_absolute_path(tmp_path):
    target = str(tmp_path / "elsewhere" / "x.png")
    assert common.resolve_output_path(target, lambda: "/unused") == target


@pytest.mark.parametrize("path", ["sub/x.png", "./x.png", "../x.png", "nested/dir/"])
def test_resolve_keeps_path_with_directory_component(path):
    assert common.resolve_output_path(path, lambda: "/unused") == path


@pytest.mark.parametrize("path", [".", ".."])
def test_resolve_keeps_dot_entries_as_directories(path):
    assert common.resolve_output_path(path, lambda: "/unused") == path


@pytest.mark.parametrize("name", ["loss.png", "clip.mp4", "frames.gif"])
def test_resolve_places_bare_name_in_figure_dir(name, tmp_path):
    assert common.resolve_output_path(name, lambda: str(tmp_path)) == str(tmp_path / name)


def test_resolve_calls_figure_dir_only_for_bare_names(tmp_path):
    calls = []

    def figure_dir():
        calls.append(1)
        return str(tmp_path)

    common.resolve_output_path(str(tmp_path / "x.png"), figure_dir)
    assert calls == []
    common.resolve_output_path("x.png", figure_dir)
    assert calls == [1]


def test_resolve_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty"):
        common.resolve_output_path("", lambda: "/unused")


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


def _descent_path():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    return discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=3)


def _unit_quat(deg: float) -> tuple[float, float, float, float]:
    import math

    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


_SIMPLEX_FRAME = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(0, 0, 0, 1)]

# (saver name, needs ffmpeg, call with an output name, expected extension)
_SAVERS = [
    ("plot_loss_history", False, lambda n: plots.plot_loss_history([1.0, 0.5, 0.2], out_path=n), "png"),
    ("plot_shell_growth", False, lambda n: plots.plot_shell_growth(2, out_path=n), "png"),
    ("plot_error_histogram", False, lambda n: plots.plot_error_histogram([0.1, 0.2, 0.3], out_path=n), "png"),
    ("plot_lattice_shell_3d", False, lambda n: plots.plot_lattice_shell_3d(1, out_path=n), "png"),
    ("plot_slerp_path", False, lambda n: plots.plot_slerp_path(_unit_quat(0.0), _unit_quat(90.0), out_path=n), "png"),
    ("plot_ivm_neighbors", False, lambda n: visualize.plot_ivm_neighbors(out_path=n), "png"),
    (
        "plot_partition_tetrahedron",
        False,
        lambda n: visualize.plot_partition_tetrahedron((2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 2, 0), (2, 2, 1, 1), out_path=n),
        "png",
    ),
    ("plot_simplex_trace", False, lambda n: visualize.plot_simplex_trace(_simplex_state(), out_path=n), "png"),
    ("animate_simplex", True, lambda n: visualize.animate_simplex([_SIMPLEX_FRAME], out_path=n), "mp4"),
    ("animate_discrete_path", True, lambda n: visualize.animate_discrete_path(_descent_path(), out_path=n), "mp4"),
    ("frames_strip", False, lambda n: animations.frames_strip(animations.lattice_frames(shells=1, n=2), n), "png"),
    ("frames_to_gif", False, lambda n: animations.frames_to_gif(animations.lattice_frames(shells=1, n=2), n), "gif"),
]


@pytest.mark.parametrize(("name", "needs_ffmpeg", "call", "ext"), _SAVERS, ids=[s[0] for s in _SAVERS])
def test_saver_resolves_bare_name_into_figure_dir(name, needs_ffmpeg, call, ext, _isolate_output_dirs):
    if needs_ffmpeg:
        _ffmpeg_or_skip()
    out = call(f"policy_probe.{ext}")
    expected = _isolate_output_dirs / f"policy_probe.{ext}"
    assert out == str(expected)
    assert expected.is_file()


@pytest.mark.parametrize(("name", "needs_ffmpeg", "call", "ext"), _SAVERS, ids=[s[0] for s in _SAVERS])
def test_saver_uses_directory_path_verbatim(name, needs_ffmpeg, call, ext, tmp_path):
    if needs_ffmpeg:
        _ffmpeg_or_skip()
    target = tmp_path / "given" / f"verbatim.{ext}"
    target.parent.mkdir()
    assert call(str(target)) == str(target)
    assert target.is_file()


@pytest.mark.parametrize(
    ("module", "name"),
    [(vis_stats, "stats_gallery"), (vis_lattice, "lattice_gallery")],
    ids=["vis_stats.gallery", "vis_lattice.gallery"],
)
def test_gallery_bare_directory_resolves_into_figure_dir(module, name, _isolate_output_dirs):
    paths = module.gallery(name)
    target_dir = _isolate_output_dirs / name
    assert paths
    assert all(os.path.dirname(p) == str(target_dir) for p in paths)
    assert all(os.path.isfile(p) for p in paths)


@pytest.mark.parametrize("module", [vis_stats, vis_lattice], ids=["vis_stats", "vis_lattice"])
def test_gallery_directory_path_used_verbatim(module, tmp_path):
    target = tmp_path / "given"
    paths = module.gallery(str(target))
    assert all(os.path.dirname(p) == str(target) for p in paths)
