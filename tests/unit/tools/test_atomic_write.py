from __future__ import annotations

import csv
import os
import zipfile

import numpy as np
import pytest

from quadmath.tools.atomic_write import atomic_open, atomic_savez, atomic_write_text


def _names(directory) -> list[str]:
    return sorted(os.listdir(directory))


def test_atomic_write_text_creates_new_target(tmp_path):
    target = tmp_path / "data.txt"
    atomic_write_text(str(target), "alpha\n")
    assert target.read_text(encoding="utf-8") == "alpha\n"
    assert _names(tmp_path) == ["data.txt"]


def test_atomic_write_text_replaces_existing_target(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("old\n", encoding="utf-8")
    atomic_write_text(str(target), "new\n")
    assert target.read_text(encoding="utf-8") == "new\n"
    assert _names(tmp_path) == ["data.txt"]


def test_failed_write_keeps_previous_target_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "data.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_open(str(target)) as fh:
            fh.write("partial,")
            raise RuntimeError("boom")
    assert target.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert _names(tmp_path) == ["data.csv"]


def test_failed_write_creates_no_target_when_none_existed(tmp_path):
    target = tmp_path / "fresh.txt"
    with pytest.raises(TypeError):
        atomic_write_text(str(target), 42)  # type: ignore[arg-type]
    assert not target.exists()
    assert _names(tmp_path) == []


def test_failed_replace_removes_temp_file_and_keeps_target(tmp_path, monkeypatch):
    target = tmp_path / "data.txt"
    target.write_text("keep\n", encoding="utf-8")

    def _broken_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _broken_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(str(target), "new\n")
    assert target.read_text(encoding="utf-8") == "keep\n"
    assert _names(tmp_path) == ["data.txt"]


def test_atomic_open_passes_newline_through_for_csv(tmp_path):
    target = tmp_path / "rows.csv"
    with atomic_open(str(target), newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerow(["scale", "volume"])
    assert target.read_bytes() == b"scale,volume\n"


def test_atomic_open_accepts_pathlike_target(tmp_path):
    target = tmp_path / "pathlike.txt"
    atomic_write_text(target, "ok")  # type: ignore[arg-type]
    assert target.read_text(encoding="utf-8") == "ok"


def test_atomic_savez_round_trips_arrays(tmp_path):
    target = tmp_path / "data.npz"
    atomic_savez(str(target), xs=np.arange(3), ys=np.array([[1.5, 2.5]]))
    with np.load(target) as archive:
        assert sorted(archive.files) == ["xs", "ys"]
        assert archive["xs"].tolist() == [0, 1, 2]
        assert archive["ys"].tolist() == [[1.5, 2.5]]
    assert _names(tmp_path) == ["data.npz"]


def test_atomic_savez_uses_fixed_zip_timestamps_for_byte_reproducibility(tmp_path):
    target = tmp_path / "data.npz"
    atomic_savez(str(target), xs=np.arange(3))
    first = target.read_bytes()
    atomic_savez(str(target), xs=np.arange(3))
    assert target.read_bytes() == first
    with zipfile.ZipFile(target) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_failed_savez_keeps_previous_target_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "data.npz"
    atomic_savez(str(target), xs=np.arange(3))
    before = target.read_bytes()
    with pytest.raises(ValueError):
        atomic_savez(str(target), bad=np.array([object()], dtype=object))
    assert target.read_bytes() == before
    assert _names(tmp_path) == ["data.npz"]
