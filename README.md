# QuadMath: Analytical Review of Quadray Coordinates (4D)

This repository organizes a comprehensive review of Quadray coordinates, integer volume quantization, optimization on tetrahedral lattices, and information geometry. It contains modular Markdown sources, LaTeX outputs, and build scripts to produce individual and combined PDFs.

## Project Structure

- `quadmath/markdown/`
  - `01_foundations.md`: Foundations of Quadray coordinates, synergetics, integer volumes
  - `02_framework.md`: Optimization framework, information geometry, security
  - `03_extension_and_implementation.md`: Extensions, implementations, applications
- `quadmath/latex/`
  - `preamble.tex`: LaTeX preamble loaded by Pandoc
  - Generated `.tex` files for each module and combined review
- `quadmath/scripts/`
  - `render_pdf.sh`: Build script that compiles per-module and combined PDFs
- `quadmath/output/`
  - Build artifacts (`*.pdf`, combined `quadmath_review.pdf`)
- `quadmath/resources/`
  - Images, diagrams, and supplementary assets

Source materials informing the modules are in `research_1.md`, `research_2.md`, `research_3.md` at the repo root.

## Build (PDF)

Dependencies: `pandoc`, `xelatex` (TeX Live).

- Ubuntu/Debian:
  - sudo apt-get update
  - sudo apt-get install -y pandoc texlive-xetex texlive-fonts-recommended fonts-dejavu
- macOS:
  - brew install pandoc
  - brew install --cask mactex-no-gui

Render all outputs:

```bash
bash quadmath/scripts/render_pdf.sh
```

Artifacts will be written to `quadmath/output/` and `.tex` exports to `quadmath/latex/`.