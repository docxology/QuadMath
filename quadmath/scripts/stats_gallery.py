#!/usr/bin/env python3
"""Statistics visualization gallery (thin orchestrator).

Delegates to ``src/vis_stats.py::gallery`` per the thin-orchestrator
contract in ``quadmath/scripts/AGENTS.md``; sets a headless backend and a
fixed seed, writes the four statistics gallery PNGs under
``quadmath/output/figures/`` and prints each output path on its own line.
The stdout path lines are the ``make_all_figures`` manifest contract
(only path lines on stdout).
"""
from __future__ import annotations

import os
import sys

#: Output file names of the four gallery figures, in composition order.
#: Must equal ``src/vis_stats.py::GALLERY_FILES`` (enforced by
#: ``tests/test_vis_stats.py``).
GALLERY_FILES = (
    "stats_gallery_latency.png",
    "stats_gallery_scaling.png",
    "stats_gallery_ci.png",
    "stats_gallery_ecdf.png",
)


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main(out_dir: str | None = None) -> list[str]:
    """Compose the four statistics gallery figures and print their paths.

    Parameters
    - out_dir: Output directory; ``None`` resolves to
      ``<repo>/quadmath/output/figures`` via ``src/paths.py``.

    Returns
    - list[str]: The four absolute written paths, in
      :data:`GALLERY_FILES` order.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    from paths import get_figure_dir  # noqa: WPS433

    from vis_stats import gallery  # noqa: WPS433

    if out_dir is None:
        out_dir = get_figure_dir()
    out_paths: list[str] = []
    for out_path in gallery(out_dir):
        absolute = os.path.abspath(out_path)
        out_paths.append(absolute)
        print(absolute)
    return out_paths


if __name__ == "__main__":
    main()
