# AGENTS.md - Test Suite Directory

## Purpose

This directory contains all tests for the QuadMath `quadmath` package
(`src/quadmath/`). Tests mirror the package layout (`tests/unit/<subpackage>/`),
enforce **100% coverage**, and use **real numerical examples only** (no mocks).

## Agent Guidelines

### Critical Rules

1. **NO MOCKS** - Never mock internal functions or modules
2. **FIXED SEEDS** - Always set `np.random.seed(42)` before random operations
3. **100% COVERAGE** - Every new branch must be tested
4. **REAL VALUES** - Use known mathematical results for assertions

### Before Adding Tests

1. **Check existing coverage**:

   ```bash
   uv run coverage run -m pytest tests/unit/core/test_quadray.py
   uv run coverage report -m --include="src/quadmath/core/quadray.py"
   ```

2. **Identify uncovered branches** in the coverage report

3. **Understand the mathematical expectations** for the function being tested

### After Adding Tests

1. **Run full suite**: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q`
2. **Verify 100% coverage**: `uv run coverage report`
3. **Check that tests are deterministic**: Run twice, same results

## Test Naming Conventions

### File Names

- `tests/unit/<subpackage>/test_<module>.py` - Primary tests for
  `quadmath/<subpackage>/<module>.py` (package-root modules in `tests/unit/`,
  script-module tests in `tests/tools/`)
- `test_<module>_cov.py` - Additional coverage tests

### Function Names

```python
# Descriptive names that explain what's being tested
def test_quadray_normalize_positive_components():
    ...

def test_fisher_information_symmetric():
    ...

def test_discrete_descent_reaches_minimum():
    ...
```

### Class Names (Optional)

```python
class TestQuadrayOperations:
    """Group related Quadray tests."""
    
    def test_add(self):
        ...
    
    def test_sub(self):
        ...
```

## Common Testing Patterns

### Testing Quadray Operations

```python
from quadmath.core.quadray import Quadray, integer_tetra_volume

def test_integer_tetra_volume_unit():
    """Unit tetrahedron should have volume 1."""
    # Use known vertices
    p0 = Quadray(0, 0, 0, 0)
    p1 = Quadray(2, 1, 1, 0)
    p2 = Quadray(1, 2, 1, 0)
    p3 = Quadray(1, 1, 2, 0)
    
    vol = integer_tetra_volume(p0, p1, p2, p3)
    assert vol == 1
```

### Testing Information Geometry

```python
import numpy as np
from quadmath.inference.information import fisher_information_matrix

def test_fisher_symmetric():
    """Fisher information matrix must be symmetric."""
    np.random.seed(42)
    gradients = np.random.randn(100, 3)
    
    F = fisher_information_matrix(gradients)
    assert np.allclose(F, F.T)
```

### Testing Optimization

```python
from quadmath.optimize.nelder_mead_quadray import nelder_mead_quadray
from quadmath.core.quadray import Quadray

def test_nelder_mead_improves():
    """Optimizer should improve or maintain objective."""
    def objective(q: Quadray) -> float:
        return sum(abs(x) for x in q.as_tuple())
    
    initial = [
        Quadray(5, 4, 3, 2),
        Quadray(6, 4, 3, 2),
        Quadray(5, 5, 3, 2),
        Quadray(5, 4, 4, 2),
    ]
    
    result = nelder_mead_quadray(objective, initial, max_iter=50)
    
    # Best value should be <= initial best
    assert result.values[0] <= objective(initial[0])
```

### Testing Visualization (Headless)

```python
import os
os.environ["MPLBACKEND"] = "Agg"  # Set before importing matplotlib

from quadmath.viz.visualize import plot_ivm_neighbors

def test_plot_saves_file(tmp_path, monkeypatch):
    """Test that plotting saves a file."""
    monkeypatch.chdir(tmp_path)
    
    path = plot_ivm_neighbors(save=True)
    assert path != ""
    assert os.path.exists(path)
```

## What NOT to Do

### ❌ Mocking

```python
# WRONG - Never do this
from unittest.mock import patch

def test_with_mock():
    with patch('quadmath.core.quadray.bareiss_determinant_int', return_value=4):
        ...
```

### ❌ Non-deterministic Tests

```python
# WRONG - No fixed seed
def test_random_data():
    gradients = np.random.randn(100, 3)  # Different every run!
    ...
```

### ❌ Testing Implementation Details

```python
# WRONG - Testing internal state
def test_internal():
    assert obj._private_attr == 5  # Don't test private members
```

## Fixtures in `conftest.py`

Available fixtures:

```python
# conftest.py adds src/ (the quadmath package) and quadmath/scripts/ to sys.path
# Package modules import absolutely; script modules import bare

# Example: Import directly
from quadmath.core.quadray import Quadray
from quadmath.inference.information import fisher_information_matrix
```

## Coverage Requirements

The `.coveragerc` file enforces:

```ini
[report]
fail_under = 100
```

If coverage drops below 100%, CI will fail.

## Adding Coverage for Edge Cases

When adding `*_cov.py` files:

```python
"""Additional coverage tests for quadmath/<subpackage>/module.py

These tests cover:
- Error handling paths
- Edge cases (empty inputs, boundary values)
- Rarely executed branches
"""

def test_empty_input():
    """Test that empty input is handled correctly."""
    result = function([])
    assert result == expected_empty_result

def test_boundary_value():
    """Test at numerical boundaries."""
    result = function(np.finfo(float).tiny)
    assert np.isfinite(result)
```

## Quality Checklist

Before committing test changes:

- [ ] All tests pass: `pytest -q`
- [ ] Coverage is 100%: `coverage report`
- [ ] Tests are deterministic (run twice)
- [ ] No mocks used
- [ ] Fixed seeds for random operations
- [ ] Descriptive test names
- [ ] Edge cases covered
