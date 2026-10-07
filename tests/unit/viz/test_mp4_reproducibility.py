"""MP4 outputs are byte-identical across repeated saves in one session.

The first save and the second save are separated by a wall-clock gap of more
than a second, so any creation timestamp the muxer embeds would differ between
them.  Tests skip when ffmpeg is not on PATH.
"""
from __future__ import annotations

import shutil
import time

import pytest

import quadmath.viz.visualize as visualize
from quadmath.core.quadray import Quadray
from quadmath.optimize.discrete_variational import discrete_ivm_descent

_SIMPLEX_FRAME = [Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(0, 0, 0, 1)]


@pytest.fixture(autouse=True)
def _isolate_output_dirs(tmp_path, monkeypatch):
    fig_dir = tmp_path / "figures"
    data_dir = tmp_path / "data"
    fig_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(visualize, "get_figure_dir", lambda: str(fig_dir))
    monkeypatch.setattr(visualize, "get_data_dir", lambda: str(data_dir))


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")


def _descent_path():
    def f(q: Quadray) -> float:
        return float((q.a - 2) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    return discrete_ivm_descent(f, Quadray(6, 0, 0, 0), max_iter=5)


def test_discrete_path_mp4_is_byte_reproducible(tmp_path):
    _require_ffmpeg()
    path = _descent_path()
    first = visualize.animate_discrete_path(path, out_path=str(tmp_path / "first.mp4"))
    time.sleep(1.1)
    second = visualize.animate_discrete_path(path, out_path=str(tmp_path / "second.mp4"))
    with open(first, "rb") as fh_a, open(second, "rb") as fh_b:
        assert fh_a.read() == fh_b.read()


def test_simplex_mp4_is_byte_reproducible(tmp_path):
    _require_ffmpeg()
    first = visualize.animate_simplex([_SIMPLEX_FRAME, _SIMPLEX_FRAME], out_path=str(tmp_path / "first.mp4"))
    time.sleep(1.1)
    second = visualize.animate_simplex([_SIMPLEX_FRAME, _SIMPLEX_FRAME], out_path=str(tmp_path / "second.mp4"))
    with open(first, "rb") as fh_a, open(second, "rb") as fh_b:
        assert fh_a.read() == fh_b.read()
