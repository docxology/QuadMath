# QuadMath Development Workflow: The render_pdf.sh Paradigm

This document explains the complete development workflow that ensures source code, tests, and documentation remain in perfect coherence.

## Overview

The QuadMath repository implements a **unified test-driven development paradigm** where:

- **Source code** implements mathematical functionality
- **Tests** validate all functionality with 100% coverage
- **Scripts** generate figures and data from source modules
- **Documentation** references code and displays generated outputs
- **`render_pdf.sh`** orchestrates the entire pipeline

## Complete Workflow Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source Code   │    │      Tests      │    │   Documentation │
│     (src/)      │◄──►│     (tests/)    │◄──►│   (markdown/)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Generation      │    │ Coverage        │    │ Validation      │
│ Scripts         │    │ Report          │    │ Scripts         │
│ (scripts/)      │    │ (100% required) │    │ (validate_md)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Output          │    │ Test Results    │    │ Markdown        │
│ Artifacts       │    │ (all pass)      │    │ Validation      │
│ (figures/data)  │    │                 │    │ (all pass)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    render_pdf.sh        │
                    │   (Orchestrator)        │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Final Outputs         │
                    │ • Individual PDFs       │
                    │ • Combined PDF          │
                    │ • LaTeX exports         │
                    │ • Validation reports    │
                    └─────────────────────────┘
```

## How render_pdf.sh Works with Markdown and Code

The `render_pdf.sh` script is the central orchestrator that ensures complete coherence between all components:

### 1. Code Validation Phase
- **Runs all generation scripts** - This validates that `src/` code works correctly
- **Scripts import from src/** - Ensures no code duplication and validates imports
- **Generates figures and data** - Creates outputs that markdown will reference

### 2. Markdown Validation Phase
- **Validates all image references** - Ensures figures referenced in markdown exist
- **Checks internal links** - Validates equation labels and section anchors
- **Validates equation formatting** - Ensures proper LaTeX equation environments

### 3. Documentation Generation Phase
- **Auto-generates glossary** - Creates API table from current `src/` code
- **Updates documentation** - Keeps code-doc sync automatically

### 4. Output Generation Phase
- **Builds individual PDFs** - Creates per-section PDFs from validated markdown
- **Builds combined PDF** - Creates unified document from all sections
- **Exports LaTeX** - Provides LaTeX source for further processing

## Test Suite and Code Connections

The test suite ensures 100% coverage of all `src/` modules and validates the entire pipeline:

### What Tests Validate
- **Mathematical correctness** - All functions produce expected results
- **Import compatibility** - Scripts can successfully import from `src/` modules
- **Output generation** - Figure and data generation works correctly
- **Deterministic execution** - All outputs are reproducible with fixed seeds
- **Path management** - Outputs go to correct directories

### Test-Driven Development Flow
1. **Write tests first** - Define expected behavior before implementation
2. **Implement functionality** - Write code to pass tests
3. **Validate integration** - Ensure scripts can use the code
4. **Update documentation** - Reflect changes in markdown
5. **Run complete pipeline** - Use `render_pdf.sh` to validate coherence

## Step-by-Step Workflow

### 1. Development Phase

```bash
# Always start with tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run -m pytest -q

# Check coverage (must be 100%)
coverage report

# Make code changes in src/
# Update corresponding tests
# Update documentation if needed
```

### 2. Validation Phase

```bash
# Run tests again to ensure changes work
uv run pytest

# Generate figures and data
uv run python quadmath/scripts/make_all_figures.py

# Validate markdown integrity
uv run python quadmath/scripts/validate_markdown.py
```

### 3. Integration Phase

```bash
# Run the complete pipeline
quadmath/scripts/render_pdf.sh
```

This script:
- Runs all generation scripts (validates src/ code works)
- Validates markdown references and images
- Generates auto-updated glossary from src/ API
- Builds individual and combined PDFs
- Exports LaTeX for further processing

## Key Components

### Source Code (`src/`)
- **quadray.py**: Core quadray coordinate system
- **cayley_menger.py**: Geometric algorithms
- **information.py**: Information geometry
- **discrete_variational.py**: Optimization methods
- **visualize.py**: Plotting utilities
- **paths.py**: Path management utilities

### Tests (`tests/`)
- **100% coverage required** for all src/ modules
- **Real numerical examples** (no mocks)
- **Deterministic RNG seeds** for reproducibility
- **Fast and hermetic** execution

### Generation Scripts (`quadmath/scripts/`)
- **Import from src/** modules (no code duplication)
- **Generate figures and data** deterministically
- **Print output paths** to stdout for manifest collection
- **Use headless plotting** (MPLBACKEND=Agg)

### Documentation (`quadmath/markdown/`)
- **References source code** using inline code formatting
- **Displays generated figures** from quadmath/output/
- **Passes validation** for images, references, and equations
- **Auto-updated glossary** from source API

### Output Structure (`quadmath/output/`)
```
quadmath/output/
├── figures/          # PNG/MP4/SVG files
├── data/             # CSV/NPZ files and manifests
├── pdf/              # Individual and combined PDFs
└── tex/              # Exported LaTeX files
```

## Validation Rules

### Markdown Validation
- All images must exist and be properly referenced
- Internal links must have valid anchors
- Equations must have unique labels
- No bare URLs (use informative link text)

### Code Validation
- All public APIs must have type hints
- No circular imports
- Consistent formatting and naming
- Error handling for edge cases

### Test Validation
- 100% statement and branch coverage
- All tests must pass
- No network or file-system writes outside output/
- Deterministic execution

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests with coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run -m pytest -q

# Generate all figures
uv run python quadmath/scripts/make_all_figures.py

# Validate markdown
uv run python quadmath/scripts/validate_markdown.py

# Build complete PDF pipeline
quadmath/scripts/render_pdf.sh

# Check specific coverage
coverage report -m
```

## Benefits of This Paradigm

1. **Coherence**: Source code, tests, and documentation stay synchronized
2. **Validation**: Automatic checking of all references and outputs
3. **Reproducibility**: Deterministic generation of all artifacts
4. **Maintainability**: Clear separation of concerns with unified workflow
5. **Quality**: 100% test coverage enforced automatically
6. **Documentation**: Auto-generated API references and validation

## Troubleshooting

### Common Issues

1. **Tests failing**: Check coverage and fix missing test cases
2. **Markdown validation errors**: Fix broken links, missing images, or duplicate labels
3. **Figure generation failures**: Ensure src/ modules work correctly
4. **PDF build errors**: Check pandoc and LaTeX installation

### Validation Commands

```bash
# Check what's failing
uv run python quadmath/scripts/validate_markdown.py --strict

# Regenerate specific figures
uv run python quadmath/scripts/information_demo.py

# Check test coverage gaps
coverage report -m
```

## Key Connections to Remember

1. **src/ modules → tests/ validation → scripts/ generation → markdown/ documentation**
2. **render_pdf.sh ensures all connections are valid before building outputs**
3. **Changes in any component must be reflected in all connected components**
4. **The test suite validates the entire pipeline, not just individual modules**
5. **Documentation is auto-generated where possible to maintain code-doc sync**

This workflow ensures that QuadMath maintains the highest standards of mathematical rigor, code quality, and documentation coherence.
