"""Tests for the figure-manifest contract (quadmath/scripts/make_all_figures.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import make_all_figures
from make_all_figures import extract_output_paths, require_existing_output

ROOT = "/work/artifacts"
FIG = f"{ROOT}/figures"
DATA = f"{ROOT}/data"


def test_extract_takes_path_token_from_prose_prefix():
    stdout = (
        "4D trajectory figure saved: " f"{FIG}/figure_13_4d_trajectory.png\n"
        "Figure 14 saved to: " f"{FIG}/figure_14_free_energy_landscape.png\n"
    )
    assert extract_output_paths(stdout, ROOT) == [
        "figures/figure_13_4d_trajectory.png",
        "figures/figure_14_free_energy_landscape.png",
    ]


def test_extract_keeps_gif_and_txt_outputs():
    stdout = (
        f"{FIG}/animation_lattice.gif\n"
        f"{DATA}/sympy_symbolics.txt\n"
    )
    assert extract_output_paths(stdout, ROOT) == [
        "figures/animation_lattice.gif",
        "data/sympy_symbolics.txt",
    ]


def test_extract_dedupes_repeated_paths_in_first_seen_order():
    stdout = (
        f"{DATA}/volumes_scale_data.csv\n"
        f"{FIG}/volumes_scale.png\n"
        f"• Data: {DATA}/volumes_scale_data.csv\n"
    )
    assert extract_output_paths(stdout, ROOT) == [
        "data/volumes_scale_data.csv",
        "figures/volumes_scale.png",
    ]


def test_extract_ignores_progress_lines_without_output_suffix():
    stdout = (
        "Converged at step 12 with gradient norm 1.00e-03\n"
        "Optimization completed in 40 steps\n"
        "Final free energy: 2.31e-02\n"
    )
    assert extract_output_paths(stdout, ROOT) == []


def test_extract_rejects_paths_outside_configured_output_root():
    stale = "/Volumes/another_run/figures/x.png"
    with pytest.raises(ValueError):
        extract_output_paths(f"saved: {stale}\n", ROOT)


def test_extract_spaced_paths_keeps_suffix_containment_and_deduplication():
    root = "/work/Quad Math"
    path = f"{root}/figures/a figure.png"
    stdout = f"Generated: {path}\n{path}\nSaved: {root}/progress log\n"
    assert extract_output_paths(stdout, root) == [
        "figures/a figure.png",
    ]
    with pytest.raises(ValueError):
        extract_output_paths("Saved: /work/Quad Math Elsewhere/a figure.png", root)
    with pytest.raises(ValueError):
        extract_output_paths(f"Saved: {root}/../outside/a figure.png", root)


def test_require_existing_output_fails_when_no_path_exists(tmp_path):
    with pytest.raises(RuntimeError):
        require_existing_output("demo.py", ["figures/missing.png"], str(tmp_path))


def test_require_existing_output_fails_when_nothing_emitted(tmp_path):
    with pytest.raises(RuntimeError):
        require_existing_output("demo.py", [], str(tmp_path))


def _fake_repo(tmp_path, monkeypatch):
    scripts = tmp_path / "quadmath" / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(make_all_figures, "_repo_root", lambda: str(tmp_path))
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(tmp_path / "run_output"))
    return scripts


def test_run_script_resolves_artifacts_under_external_configured_root(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path / "checkout", monkeypatch)
    output = tmp_path / "isolated run"
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(output))
    script = scripts / "external_demo.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "out = Path(os.environ['QUADMATH_OUTPUT_DIR']) / 'data/result.csv'\n"
        "out.parent.mkdir(parents=True)\n"
        "out.write_text('value' + chr(10) + '7' + chr(10))\n"
        "print('Data saved: ' + str(out))\n",
        encoding="utf-8",
    )
    assert make_all_figures._run_script(str(script)) == ["data/result.csv"]
    assert (output / "data/result.csv").read_text() == "value\n7\n"


def test_run_script_returns_output_relative_paths_of_written_artifacts(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path, monkeypatch)
    out = tmp_path / "run_output" / "figures"
    body = (
        "import os\n"
        f"os.makedirs({str(out)!r}, exist_ok=True)\n"
        f"open(os.path.join({str(out)!r}, 'fake.png'), 'wb').write(b'png')\n"
        f"print('Figure saved: ' + os.path.join({str(out)!r}, 'fake.png'))\n"
    )
    script = scripts / "fake_demo.py"
    script.write_text(body)
    assert make_all_figures._run_script(str(script)) == [
        "figures/fake.png",
    ]


def test_run_script_accepts_real_spaced_output_from_unrelated_cwd(tmp_path, monkeypatch):
    root = tmp_path / "Quad Math"
    root.mkdir()
    scripts = _fake_repo(root, monkeypatch)
    script = scripts / "spaced_demo.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "out = Path(os.environ['QUADMATH_OUTPUT_DIR']) / 'figures/a figure.png'\n"
        "out.parent.mkdir(parents=True)\n"
        "out.write_bytes(b'png')\n"
        "print('Generated: ' + str(out))\n"
        "print(out)\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert make_all_figures._run_script(str(script)) == [
        "figures/a figure.png",
    ]


def test_run_script_fails_when_printed_path_was_not_written(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path, monkeypatch)
    script = scripts / "fake_demo.py"
    script.write_text(
        "import os\n"
        f"print('Figure saved: ' + os.path.join({str(tmp_path)!r}, 'run_output', 'ghost.png'))\n"
    )
    with pytest.raises(RuntimeError):
        make_all_figures._run_script(str(script))


def test_run_script_fails_when_script_prints_no_output_path(tmp_path, monkeypatch):
    scripts = _fake_repo(tmp_path, monkeypatch)
    script = scripts / "timing_only.py"
    script.write_text("print('timing complete')\n")
    with pytest.raises(RuntimeError):
        make_all_figures._run_script(str(script))


def test_write_manifest_failure_keeps_previous_manifest_and_leaves_no_temp(tmp_path):
    target = tmp_path / "figure_manifest.txt"
    target.write_text("previous\n", encoding="utf-8")
    with pytest.raises(TypeError):
        make_all_figures.write_manifest(["ok.png", 3], str(target))  # type: ignore[list-item]
    assert target.read_text(encoding="utf-8") == "previous\n"
    assert os.listdir(tmp_path) == ["figure_manifest.txt"]
