# QuadMath Test Suite

This directory contains the comprehensive test suite for QuadMath. All tests maintain **100% coverage** of the `quadmath` package (`src/quadmath/`) using real numerical examples (no mocks).

## Test Statistics

Counts drift; derive them live rather than trusting prose (verified
2026-09-11 after the package restructure: 35 test files on disk).

```bash
find tests/unit tests/tools -name "test_*.py" | wc -l              # test files
grep -c "def test_" $(find tests/unit tests/tools -name "test_*.py") | awk -F: '{s+=$NF} END {print s}'   # test functions (approx; parametrize may vary)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q --co | tail -1   # exact collected count
```

- **Coverage**: 100% of the `quadmath` package (`src/quadmath/`) required
  (statements and branches) — enforced by `.coveragerc` (`fail_under = 100`).
  Check: `uv run coverage report`.

## Running Tests

```bash
# Run all tests quickly
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q

# Run with coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q
uv run coverage report

# Run specific test file
uv run pytest tests/unit/core/test_quadray.py -v

# Run with detailed output
uv run pytest -v --tb=short
```

## Test File Mapping

Tests mirror the package layout: `tests/unit/<subpackage>/test_<module>.py`
tests `quadmath/<subpackage>/<module>.py`; package-root modules live in
`tests/unit/`; script modules under `quadmath/scripts/` are tested from
`tests/tools/`.

| Test File | Tested Module | Tests |
|-----------|---------------|-------|
| `tests/unit/core/test_quadray.py` | `quadmath.core.quadray` | Core Quadray operations |
| `tests/unit/core/test_quadray_cov.py` | `quadmath.core.quadray` | Additional coverage |
| `tests/unit/core/test_cayley_menger.py` | `quadmath.core.cayley_menger` | Determinant methods |
| `tests/unit/core/test_geometry.py` | `quadmath.core.geometry` | Geometric functions |
| `tests/unit/core/test_linalg_utils.py` | `quadmath.core.linalg_utils` | Linear algebra utilities |
| `tests/unit/core/test_metrics.py` | `quadmath.core.metrics` | Entropy, eigenspectrum |
| `tests/unit/core/test_metrics_cov.py` | `quadmath.core.metrics` | Additional coverage |
| `tests/unit/core/test_symbolic.py` | `quadmath.core.symbolic` | SymPy operations |
| `tests/unit/core/test_symbolic_cov.py` | `quadmath.core.symbolic` | Additional coverage |
| `tests/unit/core/test_examples.py` | `quadmath.core.examples` | Example configurations |
| `tests/unit/core/test_examples_cov.py` | `quadmath.core.examples` | Additional coverage |
| `tests/unit/lattice/test_conversions.py` | `quadmath.lattice.conversions` | Coordinate conversions |
| `tests/unit/lattice/test_ivm_field.py` | `quadmath.lattice.ivm_field` | Static IVM field learning |
| `tests/unit/lattice/test_ivm_dynamics.py` | `quadmath.lattice.ivm_dynamics` | Lattice dynamics |
| `tests/unit/lattice/test_lattice_search.py` | `quadmath.lattice.lattice_search` | Nearest-site queries |
| `tests/unit/lattice/test_omni_numbering.py` | `quadmath.lattice.omni_numbering` | Shell enumeration |
| `tests/unit/lattice/test_spec_examples.py` | `quadmath.lattice.*` | SPEC.md claim transcription |
| `tests/unit/optimize/test_discrete_variational.py` | `quadmath.optimize.discrete_variational` | IVM descent |
| `tests/unit/optimize/test_nelder_mead_visual.py` | `quadmath.optimize.nelder_mead_quadray` | Simplex optimization |
| `tests/unit/inference/test_information.py` | `quadmath.inference.information` | Fisher information, free energy |
| `tests/unit/inference/test_information_cov.py` | `quadmath.inference.information`, `quadmath.tools.glossary_gen` | Additional coverage |
| `tests/unit/inference/test_active_inference.py` | `quadmath.inference.information` | Active Inference functions |
| `tests/unit/stats/test_statistics.py` | `quadmath.stats.statistics` | Deterministic statistics toolkit |
| `tests/unit/stats/test_benchmarks.py` | `quadmath.stats.benchmarks` | Timing harness |
| `tests/unit/learn/test_learning_eval.py` | `quadmath.learn.learning_eval` | Train/test methodology |
| `tests/unit/viz/test_visualize.py` | `quadmath.viz.visualize` | Plotting functions |
| `tests/unit/viz/test_visualize_cov.py` | `quadmath.viz.visualize` | Additional coverage |
| `tests/unit/viz/test_vis_lattice.py` | `quadmath.viz.vis_lattice` | Lattice gallery |
| `tests/unit/viz/test_vis_stats.py` | `quadmath.viz.vis_stats` | Statistics gallery |
| `tests/unit/tools/test_glossary_gen.py` | `quadmath.tools.glossary_gen` | API documentation |
| `tests/unit/test_paths.py` | `quadmath.paths` | Path management |
| `tests/unit/test_paths_cov.py` | `quadmath.paths` | Additional coverage |
| `tests/unit/test_pipeline.py` | `quadmath.pipeline` | Typed pipeline composition |
| `tests/tools/test_sympy_formalisms.py` | `quadmath/scripts/sympy_formalisms.py` | Symbolic math (script module) |
| `tests/tools/test_validate_markdown.py` | `quadmath/scripts/validate_markdown.py` | Manuscript-validation contract |

## Test Configuration

### `conftest.py`

Forces the headless Matplotlib backend and puts `src/` (so the `quadmath`
package is importable) plus `quadmath/scripts/` for script-module tests on
`sys.path`. Test imports are absolute package imports
(`from quadmath.core.quadray import ...`); script modules stay bare
(`from validate_markdown import ...`).

```python
import os
import sys

# Force headless backend for matplotlib in tests
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

SCRIPTS = os.path.join(ROOT, "quadmath", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
```

### `.coveragerc`

Coverage configuration enforcing 100% (statements and branches):

```ini
[run]
branch = True
source = src

[report]
omit =
    **/tests/*
    **/site-packages/*
fail_under = 100
show_missing = True
precision = 0
```

### Slow collection on external drives

Full-suite wall time here is ~1.5 min with a healthy `.venv` (188 tests,
2026-09-05). Earlier reports of >5 min pathological collection coincided with
a corrupted venv (broken `numpy`/`matplotlib` installs make every module
import fail and retry, which looks like a collection hang). If collection
crawls again, first verify imports: `uv run python -c "import numpy,
matplotlib"`, and repair with `uv sync --reinstall-package numpy
--reinstall-package matplotlib`.

## Testing Patterns

### Real Numerical Examples

All tests use actual computed values, not mocks:

```python
def test_integer_tetra_volume():
    """Test volume of unit tetrahedron."""
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 1, 1, 0)
    p2 = Quadray(1, 2, 1, 0)
    p3 = Quadray(1, 1, 2, 0)
    
    vol = integer_tetra_volume(p0, p1, p2, p3)
    assert vol == 1  # Known mathematical result
```

### Deterministic Seeds

Random operations use fixed seeds:

```python
def test_fisher_information():
    """Test Fisher matrix computation."""
    np.random.seed(42)  # Fixed seed for reproducibility
    gradients = np.random.randn(100, 3)
    
    F = fisher_information_matrix(gradients)
    assert F.shape == (3, 3)
    assert np.allclose(F, F.T)  # Symmetry
```

### Edge Cases

Tests cover boundary conditions:

```python
def test_empty_path_gradients():
    """Test information_length with insufficient data."""
    result = information_length(np.array([[1.0]]))
    assert result == 0.0  # T < 2 case
```

## Coverage Files (`*_cov.py`)

Some modules have additional `test_<module>_cov.py` files that:

- Cover edge cases and error paths
- Test less common code branches
- Ensure 100% branch coverage

## Adding New Tests

When adding tests for new functionality:

1. **Match naming convention**: `test_<module>.py`
2. **Mirror the package layout**: put the file in `tests/unit/<subpackage>/`
   (or `tests/unit/` for package-root modules, `tests/tools/` for script
   modules) and use absolute package imports, e.g.
   `from quadmath.core.quadray import Quadray`
3. **Use real examples**: No mocking of internal functions
4. **Set random seeds**: Use `np.random.seed(42)` or similar
5. **Test edge cases**: Empty inputs, boundary values, error conditions

```python
# Template for new test file
"""Tests for quadmath/<subpackage>/new_module.py"""

import numpy as np
import pytest
from quadmath.<subpackage>.new_module import new_function


class TestNewFunction:
    """Tests for new_function."""
    
    def test_basic_operation(self):
        """Test basic functionality."""
        result = new_function(input_data)
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = new_function(edge_input)
        assert result == edge_expected
```

## Cross-References

- [AGENTS.md](AGENTS.md) - Agent-specific testing guidance
- [../src/README.md](../src/README.md) - Source module documentation
- [../.coveragerc](../.coveragerc) - Coverage configuration
