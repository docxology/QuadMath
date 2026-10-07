#!/usr/bin/env python3
"""Regenerate all manuscript figures deterministically.

Runs the individual figure scripts in sequence, ensuring headless plotting and
collecting the emitted output paths into a manifest file under quadmath/output/.
"""
from __future__ import annotations

import os
import re
import sys
import subprocess
from typing import Dict, List

OUTPUT_SUFFIXES = (".png", ".mp4", ".pdf", ".csv", ".npz", ".gif", ".txt")
# Scripts that print no output path by design; exempt from the existence check
NO_ARTIFACT_SCRIPTS = frozenset({"gpu_acceleration_demo.py"})
_PATH_TOKEN = re.compile(r"(?:^|\s)(/\S+)\s*$")


def extract_output_paths(stdout: str, repo_root: str) -> List[str]:
    """Return unique output paths named in stdout, relative to repo_root."""
    root = os.path.normpath(repo_root)
    found: Dict[str, None] = {}
    for line in stdout.splitlines():
        match = _PATH_TOKEN.search(line)
        if match is None or not match.group(1).endswith(OUTPUT_SUFFIXES):
            continue
        token = os.path.normpath(match.group(1))
        if not token.startswith(root + os.sep):
            raise ValueError(f"Output path outside repo root: {token}")
        found[os.path.relpath(token, root)] = None
    return list(found)


def require_existing_output(script: str, rel_paths: List[str], repo_root: str) -> None:
    """Raise unless at least one emitted path exists on disk."""
    if not any(os.path.exists(os.path.join(repo_root, p)) for p in rel_paths):
        raise RuntimeError(f"{script} emitted no existing output path")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_src_on_path() -> None:
    src_path = os.path.join(_repo_root(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _get_output_dir() -> str:
    _ensure_src_on_path()
    from quadmath.paths import get_output_dir  # noqa: WPS433

    return get_output_dir()


def _get_data_dir() -> str:
    _ensure_src_on_path()
    from quadmath.paths import get_data_dir  # noqa: WPS433

    return get_data_dir()


def _run_script(path: str) -> List[str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    # Some scripts rely on src/ on sys.path; drive them via their shebang/py
    proc = subprocess.run([
        sys.executable,
        path,
    ], capture_output=True, text=True, env=env, cwd=_repo_root())
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"Script failed: {path}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    root = _repo_root()
    paths = extract_output_paths(stdout, root)
    name = os.path.basename(path)
    if name not in NO_ARTIFACT_SCRIPTS:
        require_existing_output(name, paths, root)
    return paths


def main() -> None:
    scripts = [
        os.path.join(_repo_root(), "quadmath", "scripts", "information_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "active_inference_figures.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "volumes_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "ivm_neighbors.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "quadray_clouds.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "simplex_animation.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "graphical_abstract_quadray.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "polyhedra_quadray_constructions.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "discrete_variational_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "sympy_formalisms.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "gpu_acceleration_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "ivm_field_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "ivm_dynamics_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "lattice_gallery.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "stats_gallery.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "animation_gallery.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "learning_gallery.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "quaternion_gallery.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "animation_stills.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "stats_diagnostics_gallery.py"),
    ]

    all_paths: Dict[str, None] = {}
    for script in scripts:
        if not os.path.exists(script):
            raise FileNotFoundError(f"Missing script: {script}")
        for p in _run_script(script):
            all_paths[p] = None

    data_dir = _get_data_dir()
    manifest_path = os.path.join(data_dir, "figure_manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("# Generated figure/data paths (relative to repo root)\n")
        for p in all_paths:
            f.write(p + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(manifest_path)


if __name__ == "__main__":
    main()


