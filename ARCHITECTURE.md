# QuadMath Architecture: Complete System Overview

This document provides a comprehensive overview of how the QuadMath repository architecture works, explaining the connections between source code, tests, documentation, and the build pipeline.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QuadMath Repository                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Source    │    │    Tests    │    │  Scripts    │    │ Markdown    │  │
│  │   Code      │◄──►│   (tests/)  │◄──►│ (scripts/)  │◄──►│ (markdown/) │  │
│  │   (src/)    │    │             │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                   │                   │                   │      │
│         │                   │                   │                   │      │
│         ▼                   ▼                   ▼                   ▼      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Mathematical│    │ 100% Test   │    │ Figure &    │    │ Mathematical│  │
│  │ Algorithms  │    │ Coverage    │    │ Data Gen    │    │ Concepts    │  │
│  │ & Functions │    │ Validation  │    │ Outputs     │    │ & Code Ref  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    render_pdf.sh Orchestrator                       │    │
│  │                                                                     │    │
│  │  1. Run all scripts (validate src/ code)                           │    │
│  │  2. Validate markdown (images, refs, equations)                    │    │
│  │  3. Generate glossary (auto-update from src/ API)                  │    │
│  │  4. Build PDFs (individual + combined)                             │    │
│  │  5. Export LaTeX (for further processing)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Final Outputs                                │    │
│  │                                                                     │    │
│  │  • Individual PDFs (per markdown section)                          │    │
│  │  • Combined PDF (complete manuscript)                               │    │
│  │  • LaTeX exports (for academic publishing)                          │    │
│  │  • Validation reports (ensuring coherence)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Interactions

### 1. Source Code (`src/`)
**Purpose**: Implements mathematical functionality for quadray coordinates, optimization, and information geometry.

**Key Modules**:
- `quadray.py`: Core quadray coordinate system
- `cayley_menger.py`: Geometric algorithms (Bareiss algorithm)
- `information.py`: Information geometry and Fisher information
- `discrete_variational.py`: Optimization on tetrahedral lattices
- `visualize.py`: Plotting and visualization utilities
- `paths.py`: Path management and output directory utilities

**Responsibilities**:
- Provide clean, well-typed APIs for mathematical operations
- Ensure numerical stability and exact arithmetic where appropriate
- Maintain mathematical consistency across all modules

### 2. Test Suite (`tests/`)
**Purpose**: Validates all source code functionality with 100% coverage.

**Coverage Requirements**:
- **Statement coverage**: 100% of all code lines executed
- **Branch coverage**: 100% of all conditional branches taken
- **No mocks**: All tests use real numerical examples
- **Deterministic**: Fixed RNG seeds for reproducible results

**Validation Scope**:
- Mathematical correctness of all functions
- Import compatibility between modules
- Output generation and path management
- Integration with generation scripts

### 3. Generation Scripts (`quadmath/scripts/`)
**Purpose**: Generate figures and data from source modules for documentation.

**Key Scripts**:
- `information_demo.py`: Information geometry visualizations
- `simplex_animation.py`: Nelder-Mead optimization animations
- `volumes_demo.py`: Volume calculations and scaling
- `ivm_neighbors.py`: IVM lattice neighbor analysis
- `quadray_clouds.py`: Quadray coordinate visualizations

**Workflow**:
1. Import required functions from `src/` modules
2. Generate deterministic outputs with fixed seeds
3. Save figures to `quadmath/output/figures/`
4. Save data to `quadmath/output/data/`
5. Print output paths for manifest collection

### 4. Documentation (`quadmath/markdown/`)
**Purpose**: Document mathematical concepts with references to implemented code.

**Structure**:
- `01_introduction.md`: Introduction to 4D namespaces
- `02_4d_namespaces.md`: Coxeter.4D, Einstein.4D, Fuller.4D
- `03_quadray_methods.md`: Quadray analytical details
- `04_optimization_in_4d.md`: Optimization methods
- `05_extensions.md`: Extensions and applications
- `06_discussion.md`: Discussion and implications
- `07_resources.md`: References and resources
- `08_equations_appendix.md`: Mathematical equations
- `09_free_energy_active_inference.md`: Active inference
- `10_symbols_glossary.md`: Auto-generated API reference

**Content Requirements**:
- Reference source code using inline code formatting
- Display generated figures from `quadmath/output/`
- Use descriptive links (no bare URLs)
- Pass all validation checks

## The render_pdf.sh Pipeline

### Phase 1: Code Validation
```bash
# Run all generation scripts to validate src/ code works
uv run python quadmath/scripts/ivm_neighbors.py
uv run python quadmath/scripts/quadray_clouds.py
uv run python quadmath/scripts/volumes_demo.py
# ... and more
```

**Purpose**: Ensures that all source code modules can be imported and used successfully by generation scripts.

### Phase 2: Markdown Validation
```bash
# Validate all markdown references and images
uv run python quadmath/scripts/validate_markdown.py
```

**Checks**:
- All referenced images exist in output directories
- Internal links have valid anchors
- Equations have unique labels
- No bare URLs (use informative link text)

### Phase 3: Documentation Generation
```bash
# Auto-generate glossary from current src/ API
uv run python quadmath/scripts/generate_glossary.py
```

**Purpose**: Keeps documentation automatically synchronized with source code changes.

### Phase 4: Output Generation
```bash
# Build individual PDFs from validated markdown
pandoc [markdown_file] -o [output_pdf]

# Build combined PDF from all sections
pandoc [combined_markdown] -o quadmath_review.pdf

# Export LaTeX for further processing
pandoc [markdown_file] -o [output_tex]
```

## Data Flow and Dependencies

### Input Dependencies
1. **Source code** (`src/`) - Mathematical implementations
2. **Markdown files** (`quadmath/markdown/`) - Documentation content
3. **LaTeX preamble** (`quadmath/latex/preamble.tex`) - Formatting

### Processing Pipeline
1. **Scripts import from src/** → Validate code functionality
2. **Scripts generate outputs** → Create figures and data
3. **Markdown references outputs** → Link documentation to results
4. **Validation ensures coherence** → All references are valid
5. **PDF generation** → Create final documentation

### Output Structure
```
quadmath/output/
├── figures/          # PNG/MP4/SVG files from scripts
├── data/             # CSV/NPZ files and manifests
├── pdf/              # Individual and combined PDFs
└── tex/              # Exported LaTeX files
```

## Quality Assurance Mechanisms

### 1. Test Coverage Enforcement
- **100% coverage required** via `.coveragerc`
- **Automated validation** in CI/CD pipeline
- **Real numerical examples** ensure mathematical correctness

### 2. Markdown Validation
- **Image reference validation** - All figures must exist
- **Link validation** - Internal references must be valid
- **Equation validation** - Proper LaTeX formatting required

### 3. Pipeline Validation
- **Script execution** - All generation scripts must succeed
- **Output generation** - All expected files must be created
- **PDF compilation** - All markdown must generate valid PDFs

### 4. Reproducibility
- **Deterministic RNG** - Fixed seeds for all random operations
- **Headless plotting** - `MPLBACKEND=Agg` for CI compatibility
- **Path management** - Consistent output directory structure

## Development Workflow

### 1. Code Changes
```bash
# Write tests first (TDD)
# Implement functionality
# Ensure 100% test coverage
# Update documentation if needed
```

### 2. Validation
```bash
# Run complete test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run -m pytest -q

# Generate figures and validate
uv run python quadmath/scripts/make_all_figures.py
uv run python quadmath/scripts/validate_markdown.py
```

### 3. Integration
```bash
# Run complete pipeline
quadmath/scripts/render_pdf.sh

# Verify all outputs are generated
# Check that PDFs build successfully
```

## Benefits of This Architecture

1. **Coherence**: Source code, tests, and documentation stay synchronized
2. **Validation**: Automatic checking of all references and outputs
3. **Reproducibility**: Deterministic generation of all artifacts
4. **Maintainability**: Clear separation of concerns with unified workflow
5. **Quality**: 100% test coverage enforced automatically
6. **Documentation**: Auto-generated API references and validation

## Key Principles

1. **Single Source of Truth**: Source code is the authoritative implementation
2. **Test-Driven Development**: Tests validate functionality before implementation
3. **Automated Validation**: All components are automatically checked for coherence
4. **Reproducible Outputs**: All results are deterministic and verifiable
5. **Integrated Workflow**: One command (`render_pdf.sh`) validates the entire system

This architecture ensures that QuadMath maintains the highest standards of mathematical rigor, code quality, and documentation coherence while providing a clear, maintainable structure for development and collaboration.
