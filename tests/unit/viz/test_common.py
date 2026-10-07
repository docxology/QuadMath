from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import quadmath.viz.animations as animations
from quadmath.viz._common import atomic_target, save_figure

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_atomic_target_success_leaves_only_the_target(tmp_path):
    target = tmp_path / "out.txt"
    with atomic_target(str(target)) as tmp:
        with open(tmp, "w") as fh:
            fh.write("data")
    assert target.read_text() == "data"
    assert os.listdir(tmp_path) == ["out.txt"]


def test_atomic_target_failure_keeps_previous_target_and_removes_temp(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_target(str(target)) as tmp:
            with open(tmp, "w") as fh:
                fh.write("partial")
            raise RuntimeError("boom")
    assert target.read_text() == "old"
    assert os.listdir(tmp_path) == ["out.txt"]


def test_atomic_target_failure_creates_no_target(tmp_path):
    target = tmp_path / "out.txt"
    with pytest.raises(RuntimeError):
        with atomic_target(str(target)) as tmp:
            with open(tmp, "w") as fh:
                fh.write("partial")
            raise RuntimeError("boom")
    assert not target.exists()
    assert os.listdir(tmp_path) == []


def test_atomic_temp_path_is_a_sibling_that_keeps_the_extension(tmp_path):
    target = tmp_path / "frame.png"
    with atomic_target(str(target)) as tmp:
        assert os.path.dirname(tmp) == str(tmp_path)
        assert tmp.endswith(".png")
        assert tmp != str(target)
        with open(tmp, "wb") as fh:
            fh.write(PNG_SIGNATURE)
    assert target.read_bytes() == PNG_SIGNATURE


def test_save_figure_writes_png_and_leaves_no_temp_file(tmp_path):
    fig = plt.figure()
    fig.add_subplot(111).plot([0.0, 1.0])
    target = tmp_path / "plot.png"
    save_figure(fig, str(target), dpi=80)
    plt.close(fig)
    assert target.read_bytes().startswith(PNG_SIGNATURE)
    assert os.listdir(tmp_path) == ["plot.png"]


def test_save_figure_failure_leaves_no_partial_target(tmp_path, monkeypatch):
    fig = plt.figure()
    target = tmp_path / "plot.png"

    def partial_then_fail(self, fname, *args, **kwargs):
        with open(fname, "wb") as fh:
            fh.write(PNG_SIGNATURE + b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(Figure, "savefig", partial_then_fail)
    with pytest.raises(RuntimeError, match="disk full"):
        save_figure(fig, str(target))
    plt.close(fig)
    assert not target.exists()
    assert os.listdir(tmp_path) == []


def test_frames_to_gif_failure_leaves_no_partial_gif(tmp_path, monkeypatch):
    from PIL import Image

    def partial_then_fail(self, fp, *args, **kwargs):
        with open(fp, "wb") as fh:
            fh.write(b"GIF89a partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(Image.Image, "save", partial_then_fail)
    frames = animations.lattice_frames(shells=2, n=4)
    with pytest.raises(RuntimeError, match="disk full"):
        animations.frames_to_gif(frames, str(tmp_path / "anim.gif"))
    assert os.listdir(tmp_path) == []
