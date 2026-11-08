#!/usr/bin/env python3
"""Render the native quadray basis and thirteen-site FCC neighborhood as PNG/SVG."""
from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    from quadmath.viz.abstracts import render_quadray_abstract  # noqa: WPS433

    for path in render_quadray_abstract():
        print(path)


if __name__ == "__main__":
    main()



