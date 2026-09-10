# AGENTS.md - QuadMath Repository

## Overview

QuadMath is a comprehensive analytical review of Quadray coordinates (Fuller.4D), integer volume quantization, optimization on tetrahedral lattices, and information geometry. The repository implements a unified test-driven development workflow where source code, tests, and documentation remain in perfect coherence.

**Repository**: [https://github.com/docxology/QuadMath](https://github.com/docxology/QuadMath)  
**DOI**: [10.5281/zenodo.16887791](https://zenodo.org/records/16887791)  
**License**: Apache-2.0

## Repository Architecture

```
QuadMath/
├── src/                    # Source code modules (100% test coverage)
├── tests/                  # Test suite (no mocks, real numerical examples)
├── quadmath/
│   ├── markdown/           # Manuscript source files
│   ├── scripts/            # Figure generation and build scripts
│   └── output/             # Generated artifacts (disposable)
├── ARCHITECTURE.md         # Detailed system architecture
├── WORKFLOW.md             # Development workflow documentation
└── README.md               # Quick start and overview
```

## The render_pdf.sh Paradigm

This repository follows a unified test-driven development workflow orchestrated by `render_pdf.sh`:

1. **Source Code** (`src/`) → Implements mathematical functionality
2. **Tests** (`tests/`) → Validates all functionality with 100% coverage
3. **Scripts** (`quadmath/scripts/`) → Generate figures and data from source modules
4. **Documentation** (`quadmath/markdown/`) → References code and displays generated outputs
5. **`render_pdf.sh`** → Orchestrates the entire pipeline ensuring coherence

## Agent Guidelines

### Before Making Changes

1. **Run tests first**: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q`
2. **Check coverage**: `uv run coverage report` (must be 100%)
3. **Understand dependencies**: Source modules import from each other; check imports before refactoring

### Source Code Standards (`src/`)

- **100% test coverage required** - No exceptions
- **Type hints required** - All public functions must have full type annotations
- **Docstrings required** - Follow NumPy-style docstrings with Parameters/Returns sections
- **No external runtime dependencies** - Beyond numpy, matplotlib, sympy
- **Deterministic execution** - Use fixed RNG seeds where randomness is involved

### Test Standards (`tests/`)

- **No mocks allowed** - Use real numerical examples
- **Fixed RNG seeds** - Ensure reproducibility
- **Fast and hermetic** - Tests should not depend on external state
- **Coverage files**: `test_<module>.py` tests `src/<module>.py`

### Documentation Standards (`quadmath/markdown/`)

- **Reference source code** using inline code formatting (backticks)
- **Display generated figures** from `quadmath/output/figures/`
- **Use descriptive links** - No bare URLs
- **Equation labels must be unique** across all markdown files
- **All image references must exist** - Validated by `validate_markdown.py`

### Build Pipeline Standards (`quadmath/scripts/`)

- **Import from src/ only** - No code duplication
- **Use headless plotting** - `MPLBACKEND=Agg` for CI compatibility
- **Print output paths** to stdout for manifest collection
- **Generate deterministic outputs** with fixed seeds

## Key Commands

```bash
# Install dependencies
uv sync

# Run tests with coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q
uv run coverage report

# Generate all figures
uv run python quadmath/scripts/make_all_figures.py

# Validate markdown
uv run python quadmath/scripts/validate_markdown.py

# Build complete PDF pipeline
bash quadmath/scripts/render_pdf.sh

# Clean all generated outputs
bash quadmath/scripts/clean_output.sh
```

## Module Dependencies

```
quadray.py ─────────────────────┬──► nelder_mead_quadray.py
     │                          │
     ├──► linalg_utils.py       └──► discrete_variational.py
     │
     └──► visualize.py (also imports paths.py, discrete_variational.py)

cayley_menger.py ──► quadray.py (lazy import of to_xyz)
information.py ──► quadray.py (DEFAULT_EMBEDDING); metrics.py is standalone
conversions.py, examples.py ──► quadray.py; symbolic.py, glossary_gen.py: leaf modules
```

## Quality Gates

Before any PR or commit:

1. ✅ All tests pass: `pytest -q`
2. ✅ 100% coverage maintained: `coverage report`
3. ✅ Markdown validation passes: `python quadmath/scripts/validate_markdown.py`
4. ✅ Figure generation succeeds: `python quadmath/scripts/make_all_figures.py`
5. ✅ PDF builds successfully: `bash quadmath/scripts/render_pdf.sh`

## Cross-References

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [WORKFLOW.md](WORKFLOW.md) - Development workflow documentation
- [src/README.md](src/README.md) - Source module documentation
- [tests/README.md](tests/README.md) - Test suite documentation
- [quadmath/README.md](quadmath/README.md) - Build system documentation
