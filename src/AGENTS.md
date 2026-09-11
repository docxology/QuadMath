# AGENTS.md - Source Code Directory

## Purpose

This directory contains the core Python implementation of QuadMath, packaged
as `src/quadmath/`. All modules here must maintain **100% test coverage** and
follow strict quality standards.

## Package Layout

```
src/quadmath/
├── __init__.py        # re-exports the historical top-level public API
├── paths.py           # output-directory helpers
├── pipeline.py        # typed, composable pipeline layer
├── core/              # quadray, linalg_utils, cayley_menger, geometry,
│                      # metrics, symbolic, examples
├── lattice/           # omni_numbering, ivm_field, ivm_dynamics,
│                      # lattice_search, conversions
├── optimize/          # nelder_mead_quadray, discrete_variational
├── inference/         # information (Fisher, free energy, Active Inference)
├── stats/             # statistics, benchmarks
├── learn/             # learning_eval
├── viz/               # visualize, vis_lattice, vis_stats
└── tools/             # glossary_gen
```

## Agent Guidelines

### Before Modifying Any Module

1. **Run the test for that module first** (tests mirror the package tree):

   ```bash
   uv run pytest tests/unit/core/test_quadray.py -v
   ```

2. **Check current coverage**:

   ```bash
   uv run coverage run -m pytest tests/unit/core/test_quadray.py
   uv run coverage report -m --include="src/quadmath/core/quadray.py"
   ```

3. **Understand the module's role** in the dependency graph (see below)

### After Modifying Any Module

1. **Run full test suite**: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q`
2. **Verify 100% coverage**: `uv run coverage report`
3. **Update docstrings** if function signatures change
4. **Update tests** to cover new branches

## Coding Standards

### Type Hints (Required)

```python
# ✅ Correct
def function(param: np.ndarray, eps: float = 1e-6) -> float:
    ...

# ❌ Wrong - missing type hints
def function(param, eps=1e-6):
    ...
```

### Docstrings (Required)

```python
def function_name(param1: Type) -> ReturnType:
    """Short one-line description.

    Longer description explaining the mathematical context,
    algorithmic approach, or connection to the paper.

    Parameters
    - param1: Description with units if applicable

    Returns
    - ReturnType: Description of what is returned
    """
```

### Import Organization

```python
from __future__ import annotations  # Always first

# Standard library
from dataclasses import dataclass
from typing import Callable, List, Tuple

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Local - absolute package imports only (never bare module names)
from quadmath.core.quadray import Quadray
from quadmath.paths import get_output_dir
```

All intra-package imports use the absolute form
`from quadmath.<sub>.<mod> import ...` — even within the same subpackage.
There are no relative imports and no compatibility shims at the old flat
paths (`src/quadray.py` and friends no longer exist).

### Subpackage Re-Exports

Each subpackage `__init__.py` star-re-exports its modules
(`from .<mod> import *`). Before adding a module to a subpackage, check its
public names against the sibling modules: if two modules in the same
subpackage export the same top-level name, import the colliding modules
explicitly in that `__init__` instead of star-importing both. Known
collisions (already handled): `ball_sites` (`lattice.ivm_field` vs
`lattice.ivm_dynamics`), `MAX_SHELL` (`lattice.omni_numbering` vs
`lattice.lattice_search`), `gallery`/`GALLERY_FILES`
(`viz.vis_lattice` vs `viz.vis_stats`). New public names in a module must be
added to the re-export list in `src/quadmath/__init__.py` (and its
`__all__`) if they belong to the top-level API.

### No Mocks in Tests

Tests must use real numerical examples:

```python
# ✅ Correct - real computation
def test_volume():
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 1, 1, 0)
    vol = integer_tetra_volume(p0, p1, p2, p3)
    assert vol == 1  # Known value

# ❌ Wrong - mocking
def test_volume():
    with patch('quadmath.core.quadray.bareiss_determinant_int', return_value=4):
        ...
```

## Module Categories

### Core Mathematical Modules (`quadmath/core/`)

- `quadray.py` - Quadray vector class and operations
- `linalg_utils.py` - Exact integer linear algebra
- `cayley_menger.py` - Cayley-Menger determinants
- `geometry.py` - Minkowski/Lorentz helpers
- `metrics.py` - Entropy and Fisher-matrix diagnostics
- `symbolic.py` - SymPy integration
- `examples.py` - Pre-built examples

### Lattice Modules (`quadmath/lattice/`)

- `omni_numbering.py` - Shell enumeration and site indexing
- `ivm_field.py` - Field learning on the lattice
- `ivm_dynamics.py` - Dynamics and trajectory identification
- `lattice_search.py` - Fast nearest-site queries
- `conversions.py` - Coordinate system conversions

### Optimization Modules (`quadmath/optimize/`)

- `nelder_mead_quadray.py` - Simplex method on lattice
- `discrete_variational.py` - Greedy IVM descent

### Information Geometry Modules (`quadmath/inference/`, `quadmath/stats/`)

- `information.py` - Fisher information, free energy, Active Inference
- `statistics.py` / `benchmarks.py` - Deterministic statistics and timing

### Learning Evaluation (`quadmath/learn/`)

- `learning_eval.py` - k-fold, temporal splits, learning curves

### Visualization Modules (`quadmath/viz/`)

- `visualize.py` - 3D plotting and animation
- `vis_lattice.py` - Lattice gallery figures
- `vis_stats.py` - Statistics gallery figures
- `paths.py` (top level) - Output path management

### Utility Modules (`quadmath/tools/`, top level)

- `glossary_gen.py` - API documentation generation
- `pipeline.py` (top level) - Typed composable pipeline layer

## Dependency Graph

```
quadmath/core/linalg_utils.py
      │
      ▼
quadmath/core/quadray.py ──────────────────────────────┐
      │                                                │
      ├──► quadmath/optimize/nelder_mead_quadray.py    │
      │                                                │
      └──► quadmath/optimize/discrete_variational.py   │
                   │                                   │
                   ▼                                   ▼
              quadmath/viz/visualize.py ◄──────── quadmath/paths.py

quadmath/core/cayley_menger.py (standalone, lazy to_xyz import)
quadmath/inference/information.py ──► quadmath/core/quadray.py
quadmath/core/metrics.py (standalone)
quadmath/lattice/omni_numbering.py ──► quadmath/core/quadray.py
quadmath/lattice/{ivm_field,ivm_dynamics}.py ──► quadmath/core/quadray.py
quadmath/lattice/lattice_search.py ──► quadmath/lattice/omni_numbering.py
quadmath/{stats/benchmarks,learn/learning_eval,pipeline}.py ──► lattice layer
```

## Common Patterns

### Creating New Quadray Functions

```python
from quadmath.core.quadray import Quadray

def new_quadray_operation(q: Quadray) -> Quadray:
    """One-line description.
    
    Parameters
    - q: Input quadray vector
    
    Returns
    - Quadray: Transformed quadray
    """
    # Always normalize output to canonical form
    return Quadray(q.a * 2, q.b * 2, q.c * 2, q.d * 2).normalize()
```

### Adding Visualization Functions

```python
import os

from quadmath.viz.visualize import _set_axes_equal
from quadmath.paths import get_figure_dir
import matplotlib.pyplot as plt

def plot_new_visualization(data, save: bool = True) -> str:
    """Generate visualization and optionally save.
    
    Returns
    - str: Output path if saved, else ""
    """
    fig, ax = plt.subplots()
    # ... plotting code ...
    
    if save:
        out_path = os.path.join(get_figure_dir(), "new_viz.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)
    
    plt.close(fig)
    return ""
```

## Quality Checklist

Before committing changes to any module:

- [ ] All tests pass: `pytest -q`
- [ ] Coverage is 100%: `coverage report`
- [ ] Type hints are complete
- [ ] Docstrings are present and accurate
- [ ] No circular imports introduced
- [ ] `__all__` exports updated if adding public functions
- [ ] New module registered in its subpackage `__init__.py` (and in
      `src/quadmath/__init__.py` if it belongs to the top-level API)