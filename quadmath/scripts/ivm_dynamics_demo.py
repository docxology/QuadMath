#!/usr/bin/env python3
"""Generate the IVM lattice dynamics demo figure (multi-snapshot).

Delegates to `render_dynamics_demo` in `src/ivm_dynamics.py` (thin
orchestrator contract) and prints the saved PNG path.
"""
from __future__ import annotations

import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_src_on_path() -> None:
    src_path = os.path.join(_repo_root(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    from ivm_dynamics import render_dynamics_demo  # noqa: E402  (deferred import)

    print(render_dynamics_demo())


if __name__ == "__main__":
    main()