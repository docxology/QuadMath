# QuadMath Source Modules

This directory contains the core Python implementation of QuadMath mathematical functionality. All modules maintain **100% test coverage** with real numerical examples (no mocks).

## Module Overview

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `quadray.py` | Core Quadray coordinate system | `Quadray`, `to_xyz`, `integer_tetra_volume`, `ace_tetravolume_5x5` |
| `cayley_menger.py` | Cayley-Menger determinant methods | `tetra_volume_cayley_menger`, `ivm_tetra_volume_cayley_menger`, `squared_distances_from_quadrays`, `tetra_circumradius`, `tetra_inradius` |
| `information.py` | Information geometry & Active Inference | `fisher_information_matrix`, `free_energy`, `active_inference_step` |
| `metrics.py` | Information-theoretic metrics | `shannon_entropy`, `fim_eigenspectrum`, `fisher_curvature_analysis` |
| `discrete_variational.py` | IVM lattice optimization | `neighbor_moves_ivm`, `discrete_ivm_descent`, `DiscretePath` |
| `nelder_mead_quadray.py` | Nelder-Mead on quadray lattice | `nelder_mead_quadray`, `SimplexState` |
| `visualize.py` | Plotting and animation | `plot_ivm_neighbors`, `animate_simplex`, `plot_simplex_trace` |
| `linalg_utils.py` | Linear algebra utilities | `bareiss_determinant_int`, `bareiss_rank`, `integer_adjugate` |
| `conversions.py` | Coordinate conversions | Quadray ↔ XYZ conversions |
| `geometry.py` | Geometric utilities | Basic geometry functions |
| `paths.py` | Path management | `get_output_dir`, `get_figure_dir`, `get_data_dir` |
| `examples.py` | Example configurations | Pre-built Quadray examples |
| `symbolic.py` | Symbolic math (SymPy) | Symbolic Quadray operations |
| `glossary_gen.py` | API glossary generation | Auto-document src/ modules |

## Core Modules

### `quadray.py` - Quadray Coordinate System

The foundational module implementing Fuller.4D Quadray coordinates:

```python
from quadray import Quadray, to_xyz, DEFAULT_EMBEDDING

# Create a quadray vector
q = Quadray(2, 1, 1, 0)

# Normalize to canonical form (at least one zero component)
q_norm = q.normalize()

# Convert to XYZ coordinates
xyz = to_xyz(q, DEFAULT_EMBEDDING)

# Compute integer tetra-volume
from quadray import integer_tetra_volume
vol = integer_tetra_volume(p0, p1, p2, p3)
```

### `information.py` - Information Geometry

Implements Fisher information, variational free energy, and Active Inference:

```python
from information import (
    fisher_information_matrix,
    free_energy,
    natural_gradient_step,
    active_inference_step,
    expected_free_energy
)

# Compute Fisher information from gradients
F = fisher_information_matrix(gradients)

# Compute variational free energy
F_val = free_energy(log_p_o_given_s, q, p)

# Natural gradient update
delta = natural_gradient_step(gradient, F)
```

### `nelder_mead_quadray.py` - Lattice Optimization

Nelder-Mead simplex optimization adapted for the integer quadray lattice:

```python
from nelder_mead_quadray import nelder_mead_quadray, SimplexState
from quadray import Quadray

def objective(q: Quadray) -> float:
    return sum(q.as_tuple())  # Example objective

initial = [Quadray(0,0,0,0), Quadray(1,0,0,0), 
           Quadray(0,1,0,0), Quadray(0,0,1,0)]

result: SimplexState = nelder_mead_quadray(objective, initial)
```

### `discrete_variational.py` - Discrete Descent

Greedy descent over the IVM lattice using 12 canonical neighbor moves:

```python
from discrete_variational import discrete_ivm_descent, DiscretePath
from quadray import Quadray

path: DiscretePath = discrete_ivm_descent(
    objective=my_function,
    start=Quadray(5, 3, 2, 0),
    max_iter=100
)
```

## Module Dependencies

```
quadray.py ◄── linalg_utils.py
    │
    ├──► nelder_mead_quadray.py
    └──► discrete_variational.py

visualize.py
    ├──► quadray.py
    ├──► nelder_mead_quadray.py
    ├──► discrete_variational.py
    └──► paths.py

information.py
    └──► quadray.py (DEFAULT_EMBEDDING for the Quadray Fisher pullback)
metrics.py (standalone, numpy only)
cayley_menger.py (standalone, numpy only; imports quadray.to_xyz lazily)
```

## Development Standards

### Type Hints

All public functions must have complete type annotations:

```python
def example_function(
    param1: np.ndarray,
    param2: float = 1.0
) -> Tuple[np.ndarray, float]:
    """Docstring here."""
    ...
```

### Docstrings

Follow NumPy-style docstrings:

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """Short description.

    Longer description if needed.

    Parameters
    - param1: Description of param1
    - param2: Description of param2

    Returns
    - ReturnType: Description of return value
    """
```

### Testing

Every module `module.py` has corresponding test file `tests/test_module.py`:

```bash
# Run tests for specific module
uv run pytest tests/test_quadray.py -v

# Check coverage for specific module
uv run coverage run -m pytest tests/test_quadray.py
uv run coverage report -m --include="src/quadray.py"
```

## Cross-References

- [AGENTS.md](AGENTS.md) - Agent-specific guidance for source modifications
- [../tests/README.md](../tests/README.md) - Test suite documentation
- [../ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture overview
