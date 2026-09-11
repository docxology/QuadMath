# QuadMath Source Modules

This directory contains the core Python implementation of QuadMath mathematical functionality, packaged as the importable `quadmath` package (`src/quadmath/`). All modules maintain **100% test coverage** with real numerical examples (no mocks).

## Package Layout

```
src/quadmath/
├── __init__.py        # re-exports the historical top-level public API
├── paths.py           # output-directory helpers
├── pipeline.py        # typed, composable pipeline layer
├── core/              # quadray coordinates, exact linear algebra, volumes
├── lattice/           # IVM numbering, fields, dynamics, search, conversions
├── optimize/          # Nelder-Mead on the lattice, discrete variational descent
├── inference/         # information geometry & Active Inference
├── stats/             # deterministic statistics & benchmarking
├── learn/             # train/test methodology for the lattice learners
├── viz/               # plotting primitives & deterministic galleries
└── tools/             # auto-documentation (API glossary generation)
```

## Module Map

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `quadmath/core/quadray.py` | Core Quadray coordinate system | `Quadray`, `to_xyz`, `integer_tetra_volume`, `ace_tetravolume_5x5` |
| `quadmath/core/cayley_menger.py` | Cayley-Menger determinant methods | `tetra_volume_cayley_menger`, `ivm_tetra_volume_cayley_menger`, `squared_distances_from_quadrays`, `tetra_circumradius`, `tetra_inradius` |
| `quadmath/inference/information.py` | Information geometry & Active Inference | `fisher_information_matrix`, `free_energy`, `active_inference_step` |
| `quadmath/core/metrics.py` | Information-theoretic metrics | `shannon_entropy`, `fim_eigenspectrum`, `fisher_curvature_analysis` |
| `quadmath/optimize/discrete_variational.py` | IVM lattice optimization | `neighbor_moves_ivm`, `discrete_ivm_descent`, `DiscretePath` |
| `quadmath/optimize/nelder_mead_quadray.py` | Nelder-Mead on quadray lattice | `nelder_mead_quadray`, `SimplexState` |
| `quadmath/viz/visualize.py` | Plotting and animation | `plot_ivm_neighbors`, `animate_simplex`, `plot_simplex_trace` |
| `quadmath/core/linalg_utils.py` | Linear algebra utilities | `bareiss_determinant_int`, `bareiss_rank`, `integer_adjugate` |
| `quadmath/lattice/conversions.py` | Coordinate conversions | Quadray ↔ XYZ conversions |
| `quadmath/core/geometry.py` | Geometric utilities | Basic geometry functions |
| `quadmath/paths.py` | Path management | `get_output_dir`, `get_figure_dir`, `get_data_dir` |
| `quadmath/core/examples.py` | Example configurations | Pre-built Quadray examples |
| `quadmath/core/symbolic.py` | Symbolic math (SymPy) | Symbolic Quadray operations |
| `quadmath/tools/glossary_gen.py` | API glossary generation | Auto-document src/ modules |
| `quadmath/lattice/omni_numbering.py` | IVM shell enumeration & site indexing | `NEIGHBOR_MOVES`, `MAX_SHELL`, `sites_through_shell` |
| `quadmath/lattice/ivm_field.py` | Field learning on the IVM lattice | `IVMField`, `ball_sites`, `quadray_shell_norm` |
| `quadmath/lattice/ivm_dynamics.py` | Dynamics & trajectory identification | `DynamicsParams`, `simulate`, `fit_trajectory` |
| `quadmath/lattice/lattice_search.py` | Fast nearest-site queries | `nearest`, `squared_distance`, `within_radius` |
| `quadmath/stats/statistics.py` | Descriptive stats & resampling | `bootstrap_ci`, `permutation_test`, `cohens_d` |
| `quadmath/stats/benchmarks.py` | Timing harness & scenarios | `BenchRow`, `run_all`, `time_callable` |
| `quadmath/learn/learning_eval.py` | Honest train/test methodology | `cross_validate_field`, `learning_curve` |
| `quadmath/viz/vis_lattice.py` | Lattice gallery figures | `gallery`, `shell_scatter`, `field_slice` |
| `quadmath/viz/vis_stats.py` | Statistics gallery figures | `gallery`, `plot_ci_bars`, `plot_ecdf` |
| `quadmath/pipeline.py` | Typed composable pipeline | `Pipeline`, `Step`, `sites_step`, `learn_step` |

## Core Modules

### `quadmath/core/quadray.py` - Quadray Coordinate System

The foundational module implementing Fuller.4D Quadray coordinates:

```python
from quadmath.core.quadray import Quadray, to_xyz, DEFAULT_EMBEDDING

# Create a quadray vector
q = Quadray(2, 1, 1, 0)

# Normalize to canonical form (at least one zero component)
q_norm = q.normalize()

# Convert to XYZ coordinates
xyz = to_xyz(q, DEFAULT_EMBEDDING)

# Compute integer tetra-volume
from quadmath.core.quadray import integer_tetra_volume
vol = integer_tetra_volume(p0, p1, p2, p3)
```

### `quadmath/inference/information.py` - Information Geometry

Implements Fisher information, variational free energy, and Active Inference:

```python
from quadmath.inference.information import (
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

### `quadmath/optimize/nelder_mead_quadray.py` - Lattice Optimization

Nelder-Mead simplex optimization adapted for the integer quadray lattice:

```python
from quadmath.optimize.nelder_mead_quadray import nelder_mead_quadray, SimplexState
from quadmath.core.quadray import Quadray

def objective(q: Quadray) -> float:
    return sum(q.as_tuple())  # Example objective

initial = [Quadray(0,0,0,0), Quadray(1,0,0,0), 
           Quadray(0,1,0,0), Quadray(0,0,1,0)]

result: SimplexState = nelder_mead_quadray(objective, initial)
```

### `quadmath/optimize/discrete_variational.py` - Discrete Descent

Greedy descent over the IVM lattice using 12 canonical neighbor moves:

```python
from quadmath.optimize.discrete_variational import discrete_ivm_descent, DiscretePath
from quadmath.core.quadray import Quadray

path: DiscretePath = discrete_ivm_descent(
    objective=my_function,
    start=Quadray(5, 3, 2, 0),
    max_iter=100
)
```

## Package Re-Exports

`src/quadmath/__init__.py` re-exports the historical top-level public API, so
`from quadmath import Quadray, nearest, ...` keeps working after the flat
`src/` modules moved into subpackages. Subpackage `__init__.py` files use
`from .<mod> import *` per module. Two cross-module name collisions prevent
flat re-export; import those from their concrete modules:

- `ball_sites`: `quadmath.lattice.ivm_field` vs `quadmath.lattice.ivm_dynamics`
  (distinct functions).
- `gallery` / `GALLERY_FILES`: `quadmath.viz.vis_lattice` vs
  `quadmath.viz.vis_stats` (distinct objects).

All internal imports are absolute (`from quadmath.<sub>.<mod> import ...`),
even within the same subpackage.

## Module Dependencies

```
quadmath.core.linalg_utils.py
      │
      ▼
quadmath.core.quadray.py ◄── quadmath.core.linalg_utils.py
    │
    ├──► quadmath/optimize/nelder_mead_quadray.py
    └──► quadmath/optimize/discrete_variational.py

quadmath/viz/visualize.py
    ├──► quadmath/core/quadray.py
    ├──► quadmath/optimize/nelder_mead_quadray.py
    ├──► quadmath/optimize/discrete_variational.py
    └──► quadmath/paths.py

quadmath/inference/information.py
    └──► quadmath/core/quadray.py (DEFAULT_EMBEDDING for the Quadray Fisher pullback)
quadmath/core/metrics.py (standalone, numpy only)
quadmath/core/cayley_menger.py (standalone, numpy only; imports quadray.to_xyz lazily)
quadmath/lattice/omni_numbering.py ──► quadmath/core/quadray.py
quadmath/lattice/ivm_field.py ──► quadmath/core/quadray.py
quadmath/lattice/ivm_dynamics.py ──► quadmath/core/quadray.py (+ quadmath/paths.py lazily)
quadmath/lattice/lattice_search.py ──► quadmath/lattice/omni_numbering.py
quadmath/stats/benchmarks.py ──► quadmath/lattice/{ivm_field,lattice_search,omni_numbering}.py
quadmath/learn/learning_eval.py ──► quadmath/lattice/{ivm_dynamics,ivm_field}.py
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

The test tree mirrors the package tree: `src/quadmath/core/quadray.py` is
tested by `tests/unit/core/test_quadray.py`, and so on for every subpackage
(`tests/unit/{core,lattice,optimize,inference,stats,learn,viz,tools}/`),
with `tests/unit/test_paths.py` and `tests/unit/test_pipeline.py` covering
the two top-level modules:

```bash
# Run tests for specific module
uv run pytest tests/unit/core/test_quadray.py -v

# Check coverage for specific module
uv run coverage run -m pytest tests/unit/core/test_quadray.py
uv run coverage report -m --include="src/quadmath/core/quadray.py"
```

## Cross-References

- [AGENTS.md](AGENTS.md) - Agent-specific guidance for source modifications
- [../tests/README.md](../tests/README.md) - Test suite documentation
- [../ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture overview