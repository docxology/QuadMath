#!/usr/bin/env python3
"""Lattice visualization gallery (thin orchestrator).

Delegates to ``src/vis_lattice.py::gallery`` per the thin-orchestrator
contract in ``quadmath/scripts/AGENTS.md``; sets a headless backend and a
fixed seed, writes the three gallery PNGs under ``quadmath/output/figures/``
and prints each output path on its own line.  The stdout path lines are the
``make_all_figures`` manifest contract (only path lines on stdout).
"""
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

    from paths import get_figure_dir  # noqa: WPS433

    from vis_lattice import gallery  # noqa: WPS433

    for out_path in gallery(get_figure_dir()):
        print(out_path)


if __name__ == "__main__":
    main()
