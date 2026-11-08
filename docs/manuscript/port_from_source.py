#!/usr/bin/env python3
"""Project canonical manuscript chapters and required figure assets into docs/.

The authoritative source is ``quadmath/markdown/``. The port rewrites figure
and license paths, maps resource/glossary chapters to the template's 99/98
buckets and copies the closed canonical preamble unchanged. It does not repair
obsolete source defects or maintain a divergent manuscript.

Usage from the repository root::

    python docs/manuscript/port_from_source.py
    python docs/manuscript/port_from_source.py --figures-dir /tmp/run/figures

Missing required assets fail. The shared documentation gate checks relative
targets and Markdown fragments after porting.
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "quadmath" / "markdown"
FIGS = ROOT / "quadmath" / "output" / "figures"
DST = ROOT / "docs" / "manuscript"
DST_FIGS = DST / "figures"

RENAMES = {
    "07_resources.md": "99_resources.md",
    "10_symbols_glossary.md": "98_symbols_glossary.md",
}
SECTIONS = [
    "01_introduction.md",
    "02_4d_namespaces.md",
    "03_quadray_methods.md",
    "04_optimization_in_4d.md",
    "05_extensions.md",
    "06_discussion.md",
    "07_resources.md",
    "08_equations_appendix.md",
    "09_free_energy_active_inference.md",
    "10_symbols_glossary.md",
    "11_ivm_field_learning.md",
    "12_ivm_dynamics.md",
    "13_lattice_tooling.md",
    "14_conversions_spec.md",
    "15_learning_evaluation.md",
    "16_lattice_gallery.md",
    "17_benchmarks_statistics.md",
    "18_stats_gallery.md",
]
FIGURE_NAMES = [
    "bridging_vs_native.png",
    "figure_13_4d_trajectory.png",
    "figure_14_free_energy_landscape.png",
    "fisher_information_eigenspectrum.png",
    "fisher_information_matrix.png",
    "free_energy_curve.png",
    "graphical_abstract_quadray.png",
    "graphical_abstract_quadray.svg",
    "ivm_dynamics_demo.png",
    "ivm_field_demo.png",
    "ivm_neighbors_edges.png",
    "natural_gradient_path.png",
    "partition_tetrahedron.png",
    "polyhedra_quadray_constructions.png",
    "quadray_clouds.png",
    "simplex_final.png",
    "simplex_trace.png",
    "vis_gallery_dynamics.png",
    "vis_gallery_field.png",
    "vis_gallery_shell.png",
    "simplex_trace_visualization.png",
    "stats_gallery_ci.png",
    "stats_gallery_ecdf.png",
    "stats_gallery_latency.png",
    "stats_gallery_scaling.png",
    "volumes_scale_plot.png",
    "quaternion_slerp_path.png",
    "learn_loss_history.png",
    "animation_frames_strip.png",
    "stats_ci_comparison.png",
]


def rewrite(text: str) -> str:
    """Apply the template-layout rewrites to one section's markdown text.

    Parameters
    ----------
    text : str
        Raw markdown of a source section.

    Returns
    -------
    str
        Markdown with figure, license and resource/glossary links rewritten.
    """
    text = text.replace("](../output/figures/", "](figures/")
    text = text.replace("](../LICENSE)", "](../../LICENSE)")
    for old, new in RENAMES.items():
        text = text.replace(f"]({old}", f"]({new}")
    return text


def port(figures_dir: Path = FIGS) -> None:
    """Copy canonical content and selected generated figures with rewrites."""
    DST.mkdir(parents=True, exist_ok=True)
    DST_FIGS.mkdir(parents=True, exist_ok=True)

    preamble = (SRC / "00_preamble.md").read_text(encoding="utf-8")
    (DST / "preamble.md").write_text(preamble, encoding="utf-8")

    for name in FIGURE_NAMES:
        src, dst = figures_dir / name, DST_FIGS / name
        if not src.is_file():
            raise FileNotFoundError(f"missing figure in source tree: {src}")
        shutil.copyfile(src, dst)

    for name in SECTIONS:
        text = (SRC / name).read_text(encoding="utf-8")
        (DST / RENAMES.get(name, name)).write_text(rewrite(text), encoding="utf-8")


def verify() -> list[str]:
    """Validate the projection through the same multi-root documentation gate."""
    checker = runpy.run_path(str(ROOT / "docs/development/check_links.py"))
    return checker["check_docs"](DST)


def main() -> int:
    """Run the port and verification.

    Returns
    -------
    int
        0 when the tree ports and verifies clean, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGS,
        help="Generated figure directory, including a QUADMATH_OUTPUT_DIR run's figures/.",
    )
    args = parser.parse_args()
    port(args.figures_dir)
    problems = verify()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"ported {len(SECTIONS)} sections, {len(FIGURE_NAMES)} figure assets, preamble.md; relative links and fragments resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
