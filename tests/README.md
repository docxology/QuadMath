# QuadMath Test Suite

This directory contains the comprehensive test suite for QuadMath. All tests maintain **100% coverage** of the `src/` modules using real numerical examples (no mocks).

## Test Statistics

- **Total Tests**: 117
- **Coverage**: 100% (statements and branches)
- **Test Files**: 24

## Running Tests

```bash
# Run all tests quickly
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q

# Run with coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q
uv run coverage report

# Run specific test file
uv run pytest tests/test_quadray.py -v

# Run with detailed output
uv run pytest -v --tb=short
```

## Test File Mapping

| Test File | Source Module | Tests |
|-----------|---------------|-------|
| `test_quadray.py` | `quadray.py` | Core Quadray operations |
| `test_quadray_cov.py` | `quadray.py` | Additional coverage |
| `test_cayley_menger.py` | `cayley_menger.py` | Determinant methods |
| `test_information.py` | `information.py` | Fisher information, free energy |
| `test_information_cov.py` | `information.py` | Additional coverage |
| `test_metrics.py` | `metrics.py` | Entropy, eigenspectrum |
| `test_metrics_cov.py` | `metrics.py` | Additional coverage |
| `test_discrete_variational.py` | `discrete_variational.py` | IVM descent |
| `test_nelder_mead_visual.py` | `nelder_mead_quadray.py` | Simplex optimization |
| `test_visualize.py` | `visualize.py` | Plotting functions |
| `test_visualize_cov.py` | `visualize.py` | Additional coverage |
| `test_linalg_utils.py` | `linalg_utils.py` | Linear algebra utilities |
| `test_conversions.py` | `conversions.py` | Coordinate conversions |
| `test_geometry.py` | `geometry.py` | Geometric functions |
| `test_paths.py` | `paths.py` | Path management |
| `test_paths_cov.py` | `paths.py` | Additional coverage |
| `test_examples.py` | `examples.py` | Example configurations |
| `test_examples_cov.py` | `examples.py` | Additional coverage |
| `test_symbolic.py` | `symbolic.py` | SymPy operations |
| `test_symbolic_cov.py` | `symbolic.py` | Additional coverage |
| `test_glossary_gen.py` | `glossary_gen.py` | API documentation |
| `test_active_inference.py` | `information.py` | Active Inference functions |
| `test_sympy_formalisms.py` | `symbolic.py` | Symbolic math |

## Test Configuration

### `conftest.py`

Contains shared pytest fixtures:

```python
import pytest
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### `.coveragerc`

Coverage configuration enforcing 100%:

```ini
[run]
source = src
omit = 
    src/__pycache__/*
    tests/*

[report]
fail_under = 100
show_missing = true
```

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
2. **Import from src directly**: Tests assume `conftest.py` adds `src/` to path
3. **Use real examples**: No mocking of internal functions
4. **Set random seeds**: Use `np.random.seed(42)` or similar
5. **Test edge cases**: Empty inputs, boundary values, error conditions

```python
# Template for new test file
"""Tests for src/new_module.py"""

import numpy as np
import pytest
from new_module import new_function


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
