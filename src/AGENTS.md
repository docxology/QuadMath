# AGENTS.md - Source Code Directory

## Purpose

This directory contains the core Python implementation of QuadMath. All modules here must maintain **100% test coverage** and follow strict quality standards.

## Agent Guidelines

### Before Modifying Any Module

1. **Run the test for that module first**:

   ```bash
   uv run pytest tests/test_<module>.py -v
   ```

2. **Check current coverage**:

   ```bash
   uv run coverage run -m pytest tests/test_<module>.py
   uv run coverage report -m --include="src/<module>.py"
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

# Local
from quadray import Quadray
from paths import get_output_dir
```

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
    with patch('quadray.bareiss_determinant_int', return_value=4):
        ...
```

## Module Categories

### Core Mathematical Modules

- `quadray.py` - Quadray vector class and operations
- `linalg_utils.py` - Exact integer linear algebra
- `cayley_menger.py` - Cayley-Menger determinants

### Optimization Modules

- `nelder_mead_quadray.py` - Simplex method on lattice
- `discrete_variational.py` - Greedy IVM descent

### Information Geometry Modules

- `information.py` - Fisher information, free energy, Active Inference
- `metrics.py` - Entropy, eigenspectrum analysis

### Visualization Modules

- `visualize.py` - 3D plotting and animation
- `paths.py` - Output path management

### Utility Modules

- `conversions.py` - Coordinate system conversions
- `geometry.py` - Basic geometric functions
- `examples.py` - Pre-built examples
- `symbolic.py` - SymPy integration
- `glossary_gen.py` - API documentation generation

## Dependency Graph

```
linalg_utils.py
      │
      ▼
quadray.py ──────────────────────────────┐
      │                                  │
      ├──► nelder_mead_quadray.py        │
      │                                  │
      └──► discrete_variational.py       │
                   │                     │
                   ▼                     ▼
              visualize.py ◄──────── paths.py

cayley_menger.py ◄── linalg_utils.py

information.py (standalone)
metrics.py (standalone)
```

## Common Patterns

### Creating New Quadray Functions

```python
from quadray import Quadray

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
from visualize import _set_axes_equal
from paths import get_figure_dir
import matplotlib.pyplot as plt

def plot_new_visualization(data, save: bool = True) -> str:
    """Generate visualization and optionally save.
    
    Returns
    - str: Output path if saved, else ""
    """
    fig, ax = plt.subplots()
    # ... plotting code ...
    
    if save:
        out_path = get_figure_dir() / "new_viz.png"
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
