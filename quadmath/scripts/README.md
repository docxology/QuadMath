# QuadMath Scripts

Thin orchestrators for the QuadMath manuscript pipeline. Each script only
bootstraps paths, sets up the environment, and delegates to importable modules
under `src/` — figure/data/analysis logic lives in `src/` and is tested under
`tests/`.

## Script Inventory

| Script | Purpose | Delegates to | Run command |
|--------|---------|--------------|-------------|
| `render_pdf.sh` | Full build: figures, glossary, validation, per-chapter + combined PDFs, LaTeX export | every script below (its `scripts` array) + `pandoc`/`xelatex` | `bash quadmath/scripts/render_pdf.sh` |
| `clean_output.sh` | Remove all regenerable build output (`quadmath/output/`, `quadmath/latex/`) | filesystem only | `bash quadmath/scripts/clean_output.sh` |
| `make_all_figures.py` | Run the 9 figure generators in sequence; write `figure_manifest.txt` | the 9 figure scripts below (subprocess); `src/paths.py` | `uv run python quadmath/scripts/make_all_figures.py` |
| `validate_markdown.py` | Check image refs, internal links/anchors, equation-label uniqueness | none (stdlib only) | `uv run python quadmath/scripts/validate_markdown.py` |
| `generate_glossary.py` | Regenerate `quadmath/markdown/10_symbols_glossary.md` from the `src/` API | `src/glossary_gen.py` | `uv run python quadmath/scripts/generate_glossary.py` |
| `information_demo.py` | Information geometry: Fisher curvature, natural-gradient descent, free energy (figures + CSV) | `src/information.py`, `src/discrete_variational.py`, `src/metrics.py`, `src/visualize.py`, `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/information_demo.py` |
| `active_inference_figures.py` | Active-inference figures: free energy, perception-action loop | `src/information.py`, `src/paths.py` | `uv run python quadmath/scripts/active_inference_figures.py` |
| `simplex_animation.py` | Nelder-Mead simplex animation (MP4) and trace figures | `src/nelder_mead_quadray.py`, `src/visualize.py`, `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/simplex_animation.py` |
| `discrete_variational_demo.py` | Discrete IVM-lattice descent demo (figure + data) | `src/discrete_variational.py`, `src/visualize.py`, `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/discrete_variational_demo.py` |
| `volumes_demo.py` | Tetrahedron volume scaling: integer vs Cayley-Menger (figure + CSV) | `src/cayley_menger.py`, `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/volumes_demo.py` |
| `ivm_neighbors.py` | 12-neighbor IVM lattice structure (figure + NPZ data) | `src/visualize.py`, `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/ivm_neighbors.py` |
| `quadray_clouds.py` | 3D scatter of deterministic quadray point clouds | `src/conversions.py`, `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/quadray_clouds.py` |
| `polyhedra_quadray_constructions.py` | Platonic solids and quadray constructions (figures) | `src/paths.py` (figure construction inline) | `uv run python quadmath/scripts/polyhedra_quadray_constructions.py` |
| `graphical_abstract_quadray.py` | Journal graphical-abstract overview figure | `src/quadray.py`, `src/paths.py` | `uv run python quadmath/scripts/graphical_abstract_quadray.py` |
| `sympy_formalisms.py` | Symbolic Cayley-Menger / IVM volume formalisms (SymPy figure + CSV) | `src/symbolic.py`, `src/quadray.py` | `uv run python quadmath/scripts/sympy_formalisms.py` |
| `gpu_acceleration_demo.py` | Integer tetra-volume benchmark at scale (CPU timing figure) | `src/quadray.py` | `uv run python quadmath/scripts/gpu_acceleration_demo.py` |

All commands run from the repository root.

### Pipeline composition

- `render_pdf.sh` runs, in order: `ivm_neighbors.py`, `quadray_clouds.py`,
  `volumes_demo.py`, `simplex_animation.py`, `graphical_abstract_quadray.py`,
  `polyhedra_quadray_constructions.py`, `sympy_formalisms.py`,
  `information_demo.py`, `active_inference_figures.py`, `generate_glossary.py`,
  `validate_markdown.py`, `make_all_figures.py`.
- `make_all_figures.py` runs: `information_demo.py`, `active_inference_figures.py`,
  `volumes_demo.py`, `ivm_neighbors.py`, `quadray_clouds.py`,
  `simplex_animation.py`, `discrete_variational_demo.py`, `sympy_formalisms.py`,
  `gpu_acceleration_demo.py`.
- Consequence: `discrete_variational_demo.py` and `gpu_acceleration_demo.py` run
  only via `make_all_figures.py`; `graphical_abstract_quadray.py` and
  `polyhedra_quadray_constructions.py` run only directly in `render_pdf.sh`.

## Script Development

### Template

```python
#!/usr/bin/env python
"""Generate [description] figures.

Outputs:
    quadmath/output/figures/name.png
    quadmath/output/data/name.csv
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
    os.environ.setdefault("MPLBACKEND", "Agg")  # before importing matplotlib
    _ensure_src_on_path()

    import numpy as np  # noqa: WPS433
    import matplotlib.pyplot as plt  # noqa: WPS433

    # Import the factored src/ package (there is no installed distribution)
    from quadmath.core.quadray import Quadray, to_xyz  # noqa: WPS433
    from quadmath.paths import get_figure_dir, get_data_dir  # noqa: WPS433

    np.random.seed(42)  # deterministic

    fig, ax = plt.subplots(figsize=(8, 6))
    # ... plotting code ...

    fig_path = os.path.join(get_figure_dir(), "output_name.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {fig_path}")  # stdout paths feed the manifest

    data_path = get_data_dir() / "output_name.csv"
    # ... save data ...
    print(f"Generated: {data_path}")


if __name__ == "__main__":
    main()
```

### Requirements

1. **Thin orchestrator**: bootstrap + environment setup + delegated call into
   `src/` entrypoints; logic belongs in `src/` (tested there, 100% coverage gate)
2. **Path bootstrap**: `_ensure_src_on_path()` inserting `<repo>/src`, then
   import the factored package — `from quadmath.core.quadray import ...`,
   `from quadmath.paths import ...` — NOT installed-package style
3. **Headless mode**: `MPLBACKEND=Agg` before importing matplotlib
4. **Fixed seeds**: `np.random.seed(42)` for reproducibility
5. **Print outputs**: print every generated path; stdout lines ending in
   `.png/.mp4/.pdf/.csv/.npz` feed the `make_all_figures.py` manifest
6. **Close figures + main guard**: `plt.close(fig)`; all entry logic behind
   `if __name__ == "__main__":` so scripts stay import-safe
   (`tests/test_sympy_formalisms.py` imports `sympy_formalisms.py` directly)

## Output Locations

```python
from quadmath.paths import get_figure_dir, get_data_dir, get_output_dir  # after bootstrap

get_figure_dir()  # -> quadmath/output/figures/
get_data_dir()    # -> quadmath/output/data/
get_output_dir()  # -> quadmath/output/
```

## Adding New Scripts

1. Follow the template above
2. If it is a figure generator, add it to the `scripts` list in
   `make_all_figures.py` `main()`
3. If it must run in full builds, add it to the `scripts` array in
   `render_pdf.sh`
4. Test individually: `uv run python quadmath/scripts/new_script.py`
5. Run the full pipeline: `bash quadmath/scripts/render_pdf.sh`

## Cross-References

- [Repository overview and quick start](../../README.md) — repo-root `README.md`
- [Repository-wide agent guidelines](../../AGENTS.md) — repo-root `AGENTS.md`
- [Build-system overview](../README.md) — `quadmath/README.md`
- [Build-system directory agent notes](../AGENTS.md) — `quadmath/AGENTS.md`
- [Path utilities](../../src/paths.py) — `src/paths.py`
