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
# Figure/data generators run by main(), in order. gpu_acceleration_demo.py is
# deliberately absent: it is a timing benchmark that writes no artifact.
FIGURE_SCRIPTS = (
    "information_demo.py",
    "active_inference_figures.py",
    "volumes_demo.py",
    "ivm_neighbors.py",
    "quadray_clouds.py",
    "simplex_animation.py",
    "graphical_abstract_quadray.py",
    "polyhedra_quadray_constructions.py",
    "discrete_variational_demo.py",
    "sympy_formalisms.py",
    "ivm_field_demo.py",
    "ivm_dynamics_demo.py",
    "lattice_gallery.py",
    "stats_gallery.py",
    "animation_gallery.py",
    "learning_gallery.py",
    "quaternion_gallery.py",
    "animation_stills.py",
    "stats_diagnostics_gallery.py",
)
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
    require_existing_output(os.path.basename(path), paths, root)
    return paths


def write_manifest(rel_paths: List[str], manifest_path: str) -> None:
    """Atomically write the figure manifest: a header line, then one path per line."""
    _ensure_src_on_path()
    from quadmath.tools.atomic_write import atomic_write_text  # noqa: WPS433

    body = "# Generated figure/data paths (relative to repo root)\n"
    body += "".join(p + "\n" for p in rel_paths)
    atomic_write_text(manifest_path, body)


def main() -> None:
    root = _repo_root()
    all_paths: Dict[str, None] = {}
    for name in FIGURE_SCRIPTS:
        script = os.path.join(root, "quadmath", "scripts", name)
        if not os.path.exists(script):
            raise FileNotFoundError(f"Missing script: {script}")
        for p in _run_script(script):
            all_paths[p] = None

    data_dir = _get_data_dir()
    manifest_path = os.path.join(data_dir, "figure_manifest.txt")
    write_manifest(list(all_paths), manifest_path)
    print(f"Wrote manifest: {manifest_path}")
    print(manifest_path)


if __name__ == "__main__":
    main()


