"""Tests for the figure-manifest contract (quadmath/scripts/make_all_figures.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import make_all_figures
from make_all_figures import extract_output_paths, require_existing_output

ROOT = "/work/QuadMath"
FIG = f"{ROOT}/quadmath/output/figures"
DATA = f"{ROOT}/quadmath/output/data"
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "quadmath" / "scripts"


def test_extract_takes_path_token_from_prose_prefix():
    stdout = (
        "4D trajectory figure saved: " f"{FIG}/figure_13_4d_trajectory.png\n"
        "Figure 14 saved to: " f"{FIG}/figure_14_free_energy_landscape.png\n"
    )
    assert extract_output_paths(stdout, ROOT) == [
        "quadmath/output/figures/figure_13_4d_trajectory.png",
        "quadmath/output/figures/figure_14_free_energy_landscape.png",
    ]


def test_extract_keeps_gif_and_txt_outputs():
    stdout = (
        f"{FIG}/animation_lattice.gif\n"
        f"{DATA}/sympy_symbolics.txt\n"
    )
    assert extract_output_paths(stdout, ROOT) == [
        "quadmath/output/figures/animation_lattice.gif",
        "quadmath/output/data/sympy_symbolics.txt",
    ]


def test_extract_dedupes_repeated_paths_in_first_seen_order():
    stdout = (
        f"{DATA}/volumes_scale_data.csv\n"
        f"{FIG}/volumes_scale.png\n"
        f"• Data: {DATA}/volumes_scale_data.csv\n"
    )
    assert extract_output_paths(stdout, ROOT) == [
        "quadmath/output/data/volumes_scale_data.csv",
        "quadmath/output/figures/volumes_scale.png",
    ]


def test_extract_ignores_progress_lines_without_output_suffix():
    stdout = (
        "Converged at step 12 with gradient norm 1.00e-03\n"
        "Optimization completed in 40 steps\n"
        "Final free energy: 2.31e-02\n"
    )
    assert extract_output_paths(stdout, ROOT) == []


def test_extract_rejects_paths_outside_repo_root():
    stale = "/Volumes/external_drive/Git/projects/QuadMath/quadmath/output/figures/x.png"
    with pytest.raises(ValueError, match="outside repo root"):
        extract_output_paths(f"saved: {stale}\n", ROOT)


def test_require_existing_output_passes_when_one_path_exists(tmp_path):
    (tmp_path / "quadmath" / "output" / "figures").mkdir(parents=True)
    (tmp_path / "quadmath" / "output" / "figures" / "x.png").write_bytes(b"png")
    require_existing_output(
        "demo.py",
        ["quadmath/output/figures/missing.png", "quadmath/output/figures/x.png"],
        str(tmp_path),
    )


def test_require_existing_output_fails_when_no_path_exists(tmp_path):
    with pytest.raises(RuntimeError, match="demo.py emitted no existing output path"):
        require_existing_output("demo.py", ["quadmath/output/figures/missing.png"], str(tmp_path))


def test_require_existing_output_fails_when_nothing_emitted(tmp_path):
    with pytest.raises(RuntimeError, match="demo.py emitted no existing output path"):
        require_existing_output("demo.py", [], str(tmp_path))


def _fake_repo(tmp_path, monkeypatch):
    scripts = tmp_path / "quadmath" / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(make_all_figures, "_repo_root", lambda: str(tmp_path))
    return scripts


def test_run_script_returns_repo_relative_paths_of_written_outputs(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path, monkeypatch)
    out = tmp_path / "quadmath" / "output" / "figures"
    body = (
        "import os\n"
        f"os.makedirs({str(out)!r}, exist_ok=True)\n"
        f"open(os.path.join({str(out)!r}, 'fake.png'), 'wb').write(b'png')\n"
        f"print('Figure saved: ' + os.path.join({str(out)!r}, 'fake.png'))\n"
    )
    script = scripts / "fake_demo.py"
    script.write_text(body)
    assert make_all_figures._run_script(str(script)) == [
        "quadmath/output/figures/fake.png",
    ]


def test_run_script_fails_when_printed_path_was_not_written(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path, monkeypatch)
    script = scripts / "fake_demo.py"
    script.write_text(
        "import os\n"
        f"print('Figure saved: ' + os.path.join({str(tmp_path)!r}, 'quadmath', 'output', 'ghost.png'))\n"
    )
    with pytest.raises(RuntimeError, match="fake_demo.py emitted no existing output path"):
        make_all_figures._run_script(str(script))


def test_run_script_fails_when_script_prints_no_output_path(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path, monkeypatch)
    script = scripts / "timing_only.py"
    script.write_text("print('timing complete')\n")
    with pytest.raises(RuntimeError, match="timing_only.py emitted no existing output path"):
        make_all_figures._run_script(str(script))


def test_figure_scripts_exclude_gpu_benchmark_and_all_exist():
    names = make_all_figures.FIGURE_SCRIPTS
    assert "gpu_acceleration_demo.py" not in names
    assert len(set(names)) == len(names)
    for name in names:
        assert os.path.isfile(os.path.join(SCRIPTS_DIR, name)), name


def test_write_manifest_writes_header_and_paths(tmp_path):
    target = tmp_path / "figure_manifest.txt"
    make_all_figures.write_manifest(["quadmath/output/figures/a.png"], str(target))
    assert target.read_text(encoding="utf-8") == (
        "# Generated figure/data paths (relative to repo root)\n"
        "quadmath/output/figures/a.png\n"
    )


def test_write_manifest_failure_keeps_previous_manifest_and_leaves_no_temp(tmp_path):
    target = tmp_path / "figure_manifest.txt"
    target.write_text("previous\n", encoding="utf-8")
    with pytest.raises(TypeError):
        make_all_figures.write_manifest(["ok.png", 3], str(target))  # type: ignore[list-item]
    assert target.read_text(encoding="utf-8") == "previous\n"
    assert os.listdir(tmp_path) == ["figure_manifest.txt"]
