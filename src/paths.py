from __future__ import annotations

import os


def get_repo_root(start: str | None = None) -> str:
    """Heuristically find repository root by walking up from `start`.

    Stops at the first directory containing either `.git/` or `README.md`.
    If neither is encountered before filesystem root, returns the last checked
    path (a safe terminal fallback).
    """
    path = os.path.abspath(start or os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(path, ".git")) or os.path.exists(os.path.join(path, "README.md")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return path
        path = parent


def get_output_dir() -> str:
    """Return `quadmath/output` path at the repo root and ensure it exists."""
    root = get_repo_root()
    out = os.path.join(root, "quadmath", "output")
    os.makedirs(out, exist_ok=True)
    return out


def get_data_dir() -> str:
    """Return `quadmath/output/data` path and ensure it exists."""
    out = get_output_dir()
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_figure_dir() -> str:
    """Return `quadmath/output/figures` path and ensure it exists."""
    out = get_output_dir()
    figure_dir = os.path.join(out, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir
