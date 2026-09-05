# QuadMath Generation Scripts

This directory contains all scripts for generating figures, data, and building the manuscript.

## Script Categories

### Orchestration Scripts

| Script | Purpose |
|--------|---------|
| `render_pdf.sh` | Main build orchestrator - runs all scripts, validates, builds PDFs |
| `clean_output.sh` | Remove all generated outputs (safe - everything is regeneratable) |
| `make_all_figures.py` | Run all figure generation scripts |
| `validate_markdown.py` | Validate image refs, links, equation labels |
| `generate_glossary.py` | Auto-generate API documentation from `src/` |

### Figure Generation Scripts

| Script | Generates | Description |
|--------|-----------|-------------|
| `information_demo.py` | Fisher curvature, natural gradient | Information geometry visualizations |
| `simplex_animation.py` | Nelder-Mead animation | Simplex optimization MP4 |
| `ivm_neighbors.py` | IVM lattice neighbors | 12-neighbor structure visualization |
| `quadray_clouds.py` | Quadray point clouds | 3D scatter of quadray points |
| `volumes_demo.py` | Volume scaling | Tetrahedron volume calculations |
| `discrete_variational_demo.py` | Descent paths | IVM lattice optimization |
| `graphical_abstract_quadray.py` | Journal abstract figure | High-level overview graphic |
| `polyhedra_quadray_constructions.py` | Polyhedra | Platonic solids in quadray |
| `active_inference_figures.py` | Active inference | Free energy, perception-action |
| `gpu_acceleration_demo.py` | Performance plots | GPU vs CPU benchmarks |
| `sympy_formalisms.py` | Symbolic algebra | SymPy equation rendering |

## Running Scripts

### Individual Scripts

```bash
# Run any script individually
uv run python quadmath/scripts/information_demo.py
uv run python quadmath/scripts/simplex_animation.py
```

### All Figures

```bash
# Run all figure generators
uv run python quadmath/scripts/make_all_figures.py
```

### Full Pipeline

```bash
# Complete build (figures + validation + PDFs)
bash quadmath/scripts/render_pdf.sh
```

## Script Development

### Template

```python
#!/usr/bin/env python
"""Generate [description] figures.

This script creates:
    - quadmath/output/figures/name.png
    - quadmath/output/data/name.csv

Usage:
    uv run python quadmath/scripts/this_script.py
"""
from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "Agg"  # BEFORE matplotlib import

import matplotlib.pyplot as plt
import numpy as np

# Import from src/ modules
from quadray import Quadray, to_xyz
from paths import get_figure_dir, get_data_dir


def main():
    """Generate all outputs for this script."""
    np.random.seed(42)  # Deterministic
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ... plotting code ...
    
    # Save figure
    fig_path = os.path.join(get_figure_dir(), "output_name.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {fig_path}")
    
    # Save data
    data_path = get_data_dir() / "output_name.csv"
    # ... save data ...
    print(f"Generated: {data_path}")


if __name__ == "__main__":
    main()
```

### Requirements

1. **Headless mode**: Set `MPLBACKEND=Agg` before importing matplotlib
2. **Import from src/**: Use `from quadray import ...`, `from paths import ...`
3. **Fixed seeds**: `np.random.seed(42)` for reproducibility
4. **Print outputs**: Print paths of all generated files
5. **Close figures**: `plt.close(fig)` to prevent memory leaks

## Script Details

### `render_pdf.sh`

Main orchestrator with phases:

1. Check dependencies (pandoc, xelatex)
2. Setup output directories
3. Run generation scripts
4. Validate markdown
5. Generate glossary
6. Build individual PDFs
7. Build combined PDF
8. Export LaTeX

Options:

```bash
# Skip figure generation (faster rebuild)
bash quadmath/scripts/render_pdf.sh --skip-figures

# Verbose output
LOG_LEVEL=0 bash quadmath/scripts/render_pdf.sh
```

### `validate_markdown.py`

Checks:

- All image references resolve to existing files
- Internal links have valid anchors
- Equation labels are unique
- No bare URLs (use descriptive text)

```bash
# Standard validation
uv run python quadmath/scripts/validate_markdown.py

# Strict mode (warnings become errors)
uv run python quadmath/scripts/validate_markdown.py --strict
```

### `generate_glossary.py`

Auto-generates `quadmath/markdown/10_symbols_glossary.md` from:

- All modules in `src/`
- All public (non-underscore) top-level functions, classes, and ALL-CAPS constants
- Docstrings and type hints

```bash
uv run python quadmath/scripts/generate_glossary.py
```

### `clean_output.sh`

Removes all regeneratable content:

- `quadmath/output/figures/`
- `quadmath/output/data/`
- `quadmath/output/pdf/`
- `quadmath/output/tex/`
- `quadmath/output/latex_temp/`

```bash
bash quadmath/scripts/clean_output.sh
```

## Output Locations

Scripts should use path utilities:

```python
from paths import get_figure_dir, get_data_dir, get_output_dir

# Figures go here
get_figure_dir()  # -> quadmath/output/figures/

# Data files go here
get_data_dir()    # -> quadmath/output/data/

# General output
get_output_dir()  # -> quadmath/output/
```

## Adding New Scripts

1. Create script following template above
2. Add to `make_all_figures.py`:

   ```python
   SCRIPTS = [
       # ...
       "new_script.py",
   ]
   ```

3. Test individually: `uv run python quadmath/scripts/new_script.py`
4. Run full pipeline: `bash quadmath/scripts/render_pdf.sh`

## Cross-References

- [../README.md](../README.md) - Build system overview
- [../AGENTS.md](../AGENTS.md) - Agent guidelines for scripts
- [../../src/paths.py](../../src/paths.py) - Path utility source
