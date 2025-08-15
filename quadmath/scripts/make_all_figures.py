#!/usr/bin/env python3
"""Regenerate all manuscript figures deterministically.

Runs the individual figure scripts in sequence, ensuring headless plotting and
collecting the emitted output paths into a manifest file under quadmath/output/.
"""
from __future__ import annotations

import os
import sys
import subprocess
from typing import List


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_src_on_path() -> None:
    src_path = os.path.join(_repo_root(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _get_output_dir() -> str:
    _ensure_src_on_path()
    from paths import get_output_dir  # noqa: WPS433

    return get_output_dir()


def _get_data_dir() -> str:
    _ensure_src_on_path()
    from paths import get_data_dir  # noqa: WPS433

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
    # Collect any printed output paths from the script
    paths: List[str] = []
    for line in stdout.splitlines():
        if any(line.endswith(ext) for ext in (".png", ".mp4", ".pdf", ".csv", ".npz")):
            paths.append(line.strip())
    return paths


def main() -> None:
    scripts = [
        os.path.join(_repo_root(), "quadmath", "scripts", "information_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "active_inference_figures.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "volumes_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "ivm_neighbors.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "quadray_clouds.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "simplex_animation.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "discrete_variational_demo.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "sympy_formalisms.py"),
        os.path.join(_repo_root(), "quadmath", "scripts", "gpu_acceleration_demo.py"),
    ]

    all_paths: List[str] = []
    for script in scripts:
        if not os.path.exists(script):
            raise FileNotFoundError(f"Missing script: {script}")
        out_paths = _run_script(script)
        all_paths.extend(out_paths)

    data_dir = _get_data_dir()
    manifest_path = os.path.join(data_dir, "figure_manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("# Generated figure/data paths\n")
        for p in all_paths:
            f.write(p + "\n")
    print(f"Wrote manifest: {manifest_path}")
    print(manifest_path)


if __name__ == "__main__":
    main()


