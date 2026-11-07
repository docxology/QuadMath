"""Exercise entrypoint cwd and coherence failures with executable process fixtures."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pipeline(tmp_path):
    root = tmp_path / "Quad Math"
    scripts = root / "quadmath" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "run_all.sh", root / "run_all.sh")
    (root / "project-input").write_text("QuadMath process fixture", encoding="utf-8")
    (root / "api.txt").write_text("current API\n", encoding="utf-8")
    (root / "glossary.txt").write_text("current API\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    coverage = bin_dir / "coverage_process.py"
    coverage.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "root = Path.cwd()\n"
        "value = (root / 'project-input').read_text()\n"
        "if sys.argv[1] == 'run':\n"
        "    (root / 'coverage-artifact').write_text(value)\n"
        "elif sys.argv[1] == 'report':\n"
        "    assert (root / 'coverage-artifact').read_text() == value\n"
        "else:\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    launcher = bin_dir / "uv"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:3] == ['run', 'coverage']:\n"
        "    command = str(Path(__file__).with_name('coverage_process.py'))\n"
        "elif sys.argv[1:3] == ['run', 'python']:\n"
        "    command = sys.argv[3]\n"
        "else:\n"
        "    raise SystemExit(2)\n"
        "args = sys.argv[3:] if sys.argv[2] == 'coverage' else sys.argv[4:]\n"
        "os.execv(sys.executable, [sys.executable, command, *args])\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    (scripts / "make_all_figures.py").write_text(
        "from pathlib import Path\n"
        "value = Path('project-input').read_text()\n"
        "Path('figure-artifact').write_text(value)\n",
        encoding="utf-8",
    )
    (scripts / "generate_glossary.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "current = Path('api.txt').read_text()\n"
        "if '--check' in sys.argv:\n"
        "    if Path('glossary.txt').read_text() != current:\n"
        "        raise SystemExit('glossary is stale')\n"
        "else:\n"
        "    Path('glossary.txt').write_text(current)\n",
        encoding="utf-8",
    )
    (scripts / "validate_markdown.py").write_text(
        "from pathlib import Path\n"
        "assert Path('figure-artifact').read_text() == Path('project-input').read_text()\n"
        "Path('validation-artifact').write_text('validated')\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "other project"
    elsewhere.mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return root, elsewhere, env


def test_nonroot_spaced_entrypoint_runs_processes_in_its_own_project(pipeline):
    root, elsewhere, env = pipeline
    result = subprocess.run(
        ["bash", str(root / "run_all.sh")], cwd=elsewhere, env=env,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "coverage-artifact").read_text() == "QuadMath process fixture"
    assert (root / "figure-artifact").read_text() == "QuadMath process fixture"
    assert (root / "validation-artifact").read_text() == "validated"
    assert list(elsewhere.iterdir()) == []


def test_nonpdf_gate_rejects_stale_glossary_without_rewriting_it(pipeline):
    root, elsewhere, env = pipeline
    glossary = root / "glossary.txt"
    glossary.write_text("stale API\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(root / "run_all.sh")], cwd=elsewhere, env=env,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "glossary is stale" in result.stderr
    assert glossary.read_text() == "stale API\n"
    assert not (root / "validation-artifact").exists()
