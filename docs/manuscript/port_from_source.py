#!/usr/bin/env python3
"""Port quadmath/markdown/ sections into this template-layout tree.

Idempotent re-port: copies the numbered sections and referenced figures from
the editing source of truth (``quadmath/markdown/`` + ``quadmath/output/figures/``)
into ``docs/manuscript/``, applying the layout rewrites this tree requires:

1. images ``../output/figures/X`` -> ``figures/X``
2. LICENSE link ``../LICENSE`` -> ``../../LICENSE``
3. section renumbering to template buckets: ``07_resources.md`` ->
   ``99_resources.md`` (references render last), ``10_symbols_glossary.md`` ->
   ``98_symbols_glossary.md`` (glossary bucket)
4. inherited broken code-anchor links repointed: 09 links {#code:*} targets at
   ``08_equations_appendix.md`` (which defines only {#eq:*} anchors); the
   targets live in ``03_quadray_methods.md``
5. ``00_preamble.md`` -> ``preamble.md``; upstream the ``​```latex`` fence is
   never closed (local render_pdf.sh tolerates this; the template's
   ``extract_preamble`` does not), so the closing fence is appended when absent

Usage (from the repository root)::

    uv run python docs/manuscript/port_from_source.py

Exits 0 when the tree is in sync, 1 when a figure or link is missing.
"""

from __future__ import annotations

import re
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
]
# Broken inherited code-anchor links (upstream defect): the {#code:*} targets
# live in 03_quadray_methods.md, not 08_equations_appendix.md.
REPOINT_FROM = "08_equations_appendix.md"
REPOINT_TO = "03_quadray_methods.md"
REPOINT_NAMES = ("free_energy", "fisher_information_matrix", "natural_gradient_step")


def rewrite(text: str) -> str:
    """Apply the template-layout rewrites to one section's markdown text.

    Parameters
    ----------
    text : str
        Raw markdown of a source section.

    Returns
    -------
    str
        Markdown with figures, LICENSE, section links, and code anchors
        rewritten.
    """
    text = text.replace("](../output/figures/", "](figures/")
    text = text.replace("](../LICENSE)", "](../../LICENSE)")
    for old, new in RENAMES.items():
        text = text.replace(f"]({old}", f"]({new}")
    for name in REPOINT_NAMES:
        text = text.replace(
            f"]({REPOINT_FROM}#code:{name})",
            f"]({REPOINT_TO}#code:{name})",
        )
    return text


def port() -> None:
    """Copy preamble, figures, and sections with rewrites applied."""
    DST.mkdir(parents=True, exist_ok=True)
    DST_FIGS.mkdir(parents=True, exist_ok=True)

    preamble = (SRC / "00_preamble.md").read_text(encoding="utf-8")
    if not re.search(r"```\s*latex\s*\n.*\n\s*```\s*$", preamble, re.DOTALL):
        preamble = preamble.rstrip("\n") + "\n```\n"
    (DST / "preamble.md").write_text(preamble, encoding="utf-8")

    for name in FIGURE_NAMES:
        src, dst = FIGS / name, DST_FIGS / name
        if not src.is_file():
            raise FileNotFoundError(f"missing figure in source tree: {src}")
        shutil.copyfile(src, dst)

    for name in SECTIONS:
        text = (SRC / name).read_text(encoding="utf-8")
        (DST / RENAMES.get(name, name)).write_text(rewrite(text), encoding="utf-8")


def verify() -> list[str]:
    """Verify every relative link/image in the ported sections resolves.

    Returns
    -------
    list of str
        Human-readable problems; empty when clean.
    """
    problems: list[str] = []
    for md in sorted(DST.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for target in re.findall(r"\]\(([^)\s]+)\)", line):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (md.parent / target.split("#")[0]).resolve()
                if not resolved.exists():
                    problems.append(f"{md.name}:{lineno} unresolved link {target}")
            if "](07_resources.md" in line or "](10_symbols_glossary.md" in line:
                problems.append(f"{md.name}:{lineno} stale section link")
            if "](08_equations_appendix.md#code:" in line:
                problems.append(f"{md.name}:{lineno} un-repointed code-anchor link")
    return problems


def main() -> int:
    """Run the port and verification.

    Returns
    -------
    int
        0 when the tree ports and verifies clean, 1 otherwise.
    """
    port()
    problems = verify()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"ported {len(SECTIONS)} sections, {len(FIGURE_NAMES)} figures, preamble.md; all links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
