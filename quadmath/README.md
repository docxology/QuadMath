# QuadMath Build System

This directory contains the build system for generating figures, PDFs, and LaTeX exports from the manuscript source files.

## Directory Structure

```
quadmath/
├── markdown/          # Manuscript source files (11 sections)
├── scripts/           # Generation and build scripts (17 scripts)
├── output/            # Generated artifacts (disposable)
│   ├── figures/       # PNG, MP4, SVG files
│   ├── data/          # CSV, NPZ data files
│   ├── pdf/           # Generated PDFs
│   ├── tex/           # Exported LaTeX
│   └── latex_temp/    # Temporary build files
└── README.md          # This file
```

## The render_pdf.sh Pipeline

The central orchestrator `scripts/render_pdf.sh` ensures coherence between source code, figures, and documentation:

```bash
# Full pipeline (figures + validation + PDFs + LaTeX)
bash quadmath/scripts/render_pdf.sh

# Clean all generated outputs
bash quadmath/scripts/clean_output.sh
```

### Pipeline Phases

1. **Check Dependencies** - Verify pandoc, xelatex installed
2. **Setup Directories** - Create output subdirectories
3. **Run Generation Scripts** - Execute all figure/data generators
4. **Validate Markdown** - Check image references, links, equations
5. **Generate Glossary** - Auto-update API reference from `src/`
6. **Build Individual PDFs** - One PDF per markdown section
7. **Build Combined PDF** - Single `quadmath_review.pdf`
8. **Export LaTeX** - Generate `.tex` files for academic publishing

## Manuscript Sections

| File | Topic |
|------|-------|
| `00_preamble.md` | Title, abstract, metadata |
| `01_introduction.md` | Introduction to 4D namespaces |
| `02_4d_namespaces.md` | Coxeter.4D, Einstein.4D, Fuller.4D |
| `03_quadray_methods.md` | Quadray analytical details |
| `04_optimization_in_4d.md` | Optimization methods |
| `05_extensions.md` | Extensions and applications |
| `06_discussion.md` | Discussion and implications |
| `07_resources.md` | References and resources |
| `08_equations_appendix.md` | Mathematical equations |
| `09_free_energy_active_inference.md` | Active inference application |
| `10_symbols_glossary.md` | Auto-generated API glossary |

## Generation Scripts

Key scripts in `scripts/`:

| Script | Purpose | Output |
|--------|---------|--------|
| `render_pdf.sh` | Main orchestrator | PDFs, LaTeX |
| `make_all_figures.py` | Run all figure generators | figures/ |
| `validate_markdown.py` | Check references | Validation report |
| `generate_glossary.py` | Auto-generate API docs | `10_symbols_glossary.md` |
| `clean_output.sh` | Remove generated files | (cleans output/) |
| `information_demo.py` | Information geometry figures | PNG files |
| `simplex_animation.py` | Nelder-Mead animations | MP4 files |
| `ivm_neighbors.py` | IVM lattice visualization | PNG files |
| `quadray_clouds.py` | Quadray point clouds | PNG files |
| `volumes_demo.py` | Volume calculations | PNG, CSV |

## Output Files

All files under `output/` are regeneratable from source:

```
output/
├── figures/
│   ├── ivm_neighbors.png
│   ├── fisher_curvature.png
│   ├── simplex_animation.mp4
│   └── ...
├── data/
│   ├── volumes.csv
│   ├── trajectory.npz
│   └── output_manifest.txt
├── pdf/
│   ├── 01_introduction.pdf
│   ├── 02_4d_namespaces.pdf
│   └── quadmath_review.pdf (combined)
└── tex/
    ├── 01_introduction.tex
    └── ...
```

## Build Requirements

### System Dependencies

- **pandoc**: Markdown to PDF/LaTeX conversion
- **xelatex** (TeX Live): PDF generation with custom fonts
- **fonts-dejavu** or similar: Unicode font support

### Installation

```bash
# macOS
brew install pandoc
brew install --cask mactex-no-gui

# Ubuntu/Debian
sudo apt-get install pandoc texlive-xetex texlive-fonts-recommended fonts-dejavu
```

### Python Dependencies

Managed by `uv`:

```bash
uv sync
```

## Quick Commands

```bash
# Generate all figures only
uv run python quadmath/scripts/make_all_figures.py

# Validate markdown only
uv run python quadmath/scripts/validate_markdown.py

# Full build
bash quadmath/scripts/render_pdf.sh

# Clean everything
bash quadmath/scripts/clean_output.sh
```

## Cross-References

- [AGENTS.md](AGENTS.md) - Agent guidance for build system
- [scripts/README.md](scripts/README.md) - Detailed script documentation
- [markdown/README.md](markdown/README.md) - Manuscript formatting guide
- [../WORKFLOW.md](../WORKFLOW.md) - Development workflow
