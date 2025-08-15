#!/bin/bash

# QuadMath PDF/LaTeX renderer
# - Builds individual PDFs from Markdown modules
# - Builds combined PDF
# - Exports corresponding .tex files into quadmath/latex/

set -euo pipefail
export LANG="${LANG:-C.UTF-8}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MARKDOWN_DIR="$REPO_ROOT/quadmath/markdown"
LATEX_DIR="$REPO_ROOT/quadmath/latex"
OUTPUT_DIR="$REPO_ROOT/quadmath/output"
PREAMBLE_TEX="$LATEX_DIR/preamble.tex"
COMBINED_MD="$OUTPUT_DIR/quadmath_review.md"

# Output subdirectories
PDF_DIR="$OUTPUT_DIR/pdf"
TEX_DIR="$OUTPUT_DIR/tex"
DATA_DIR="$OUTPUT_DIR/data"
FIGURE_DIR="$OUTPUT_DIR/figures"

# Author/metadata (used for all module PDFs and the combined review)
AUTHOR_NAME="Daniel Ari Friedman"
AUTHOR_ORCID="0000-0001-6232-9096"
AUTHOR_EMAIL="daniel@activeinference.institute"
# Render ORCID and email on separate lines in the LaTeX title block
AUTHOR_TEX="$AUTHOR_NAME\\\\ ORCID: $AUTHOR_ORCID\\\\ Email: $AUTHOR_EMAIL"

# Ensure directories exist
mkdir -p "$MARKDOWN_DIR" "$LATEX_DIR" "$OUTPUT_DIR" "$PDF_DIR" "$TEX_DIR" "$DATA_DIR" "$FIGURE_DIR"

# Dependency checks
if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc is not installed." >&2
  echo "Install: sudo apt-get install -y pandoc" >&2
  exit 1
fi
if ! command -v xelatex >/dev/null 2>&1; then
  echo "Error: xelatex not found. Install TeX Live: sudo apt-get install -y texlive-xetex texlive-fonts-recommended fonts-dejavu" >&2
  exit 1
fi

# Modules (ordered)
MODULES=(
  "00_front_matter.md"
  "01_introduction.md"
  "02_4d_namespaces.md"
  "03_quadray_methods.md"
  "04_optimization_in_4d.md"
  "05_extensions.md"
  "06_discussion.md"
  "07_resources.md"
  "08_equations_appendix.md"
  "09_free_energy_active_inference.md"
  "10_symbols_glossary.md"
)

DATE_STR="$(date '+%B %d, %Y')"
COMMON_ARGS=(
  --pdf-engine=xelatex
  --toc
  --toc-depth=3
  --number-sections
  -V secnumdepth=3
  -V mainfont="DejaVu Serif"
  -V monofont="DejaVu Sans Mono"
  -V fontsize=10pt
  -V linestretch=1.0
  -V geometry:margin=1cm
  -V geometry:top=1cm
  -V geometry:bottom=1cm
  -V geometry:left=1cm
  -V geometry:right=1cm
  -V geometry:includeheadfoot
  -V colorlinks=true
  -V linkcolor=red
  -V urlcolor=red
  -V citecolor=red
  -V toccolor=black
  -V filecolor=red
  -V menucolor=red
  -V linkbordercolor=red
  -V urlbordercolor=red
  -V citebordercolor=red
  --highlight-style=tango
  --listings

  --resource-path="$MARKDOWN_DIR:$OUTPUT_DIR:$LATEX_DIR:$REPO_ROOT"
)

if [ -f "$PREAMBLE_TEX" ]; then
  COMMON_ARGS+=( -H "$PREAMBLE_TEX" )
fi

run_generation_scripts() {
  echo "Running figure/data generation scripts..."
  local runner
  if command -v uv >/dev/null 2>&1; then
    runner="uv run python"
  else
    runner="python3"
  fi
  export MPLBACKEND=Agg
  echo "Using runner: $runner"
  set -x
  $runner "$REPO_ROOT/quadmath/scripts/ivm_neighbors.py" || exit 1
  $runner "$REPO_ROOT/quadmath/scripts/quadray_clouds.py" || exit 1
  $runner "$REPO_ROOT/quadmath/scripts/volumes_demo.py" || exit 1
  $runner "$REPO_ROOT/quadmath/scripts/simplex_animation.py" || exit 1
  # Graphical abstract for Quadray axes and example vector
  if [ -f "$REPO_ROOT/quadmath/scripts/graphical_abstract_quadray.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/graphical_abstract_quadray.py" || exit 1
  fi

  if [ -f "$REPO_ROOT/quadmath/scripts/polyhedra_quadray_constructions.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/polyhedra_quadray_constructions.py" || exit 1
  fi
  # Symbolic/SymPy demonstrations
  if [ -f "$REPO_ROOT/quadmath/scripts/sympy_formalisms.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/sympy_formalisms.py" || exit 1
  fi
  if [ -f "$REPO_ROOT/quadmath/scripts/information_demo.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/information_demo.py" || exit 1
  fi
  # Active Inference figures
if [ -f "$REPO_ROOT/quadmath/scripts/active_inference_figures.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/active_inference_figures.py" || exit 1
  fi
  # Auto-generate glossary API table
  if [ -f "$REPO_ROOT/quadmath/scripts/generate_glossary.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/generate_glossary.py" || exit 1
  fi
  # Validate markdown references and images
  if [ -f "$REPO_ROOT/quadmath/scripts/validate_markdown.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/validate_markdown.py" || exit 1
  fi
  set +x
  # Collate printed artifact paths into a manifest for quick inspection
  if [ -f "$REPO_ROOT/quadmath/scripts/make_all_figures.py" ]; then
    $runner "$REPO_ROOT/quadmath/scripts/make_all_figures.py" || true
  fi
}

build_one() {
  local in_md="$1"
  local title="$2"
  local base="${in_md%.md}"
  local pdf_out="$PDF_DIR/${base}.pdf"
  local tex_out="$TEX_DIR/${base}.tex"
  
  # Generate TeX file first
  pandoc "$MARKDOWN_DIR/$in_md" \
    -f markdown+implicit_figures+tex_math_dollars+tex_math_single_backslash+raw_tex+autolink_bare_uris \
    -s \
    -V title="$title" \
    -V author="$AUTHOR_TEX" \
    -V date="$DATE_STR" \
    "${COMMON_ARGS[@]}" \
    -o "$tex_out"
  
  echo "Generated TeX: $tex_out"

  # Compile TeX to PDF with XeLaTeX
  (
    cd "$OUTPUT_DIR"
    if command -v latexmk >/dev/null 2>&1; then
      latexmk -xelatex -interaction=nonstopmode -silent -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || latexmk -xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex"
      latexmk -c -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
    else
      # Fallback: run xelatex multiple times to resolve refs
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
    fi
  )
  
  if [ -f "$pdf_out" ]; then
    echo "✅ Built: $pdf_out"
  else
    echo "❌ Failed to build: $pdf_out"
    return 1
  fi
}

# Generate figures and data before building
run_generation_scripts

# Build each module
build_one "00_front_matter.md" "QuadMath: Front Matter and Abstract"
build_one "01_introduction.md" "Introduction"
build_one "02_4d_namespaces.md" "4D Namespaces: Coxeter.4D, Einstein.4D, Fuller.4D"
build_one "03_quadray_methods.md" "Quadray Analytical Details and Methods"
build_one "04_optimization_in_4d.md" "Optimization in 4D"
build_one "05_extensions.md" "Extensions of 4D and Quadrays"
build_one "06_discussion.md" "Discussion"
build_one "07_resources.md" "Resources"
build_one "08_equations_appendix.md" "Equations and Math Supplement"
build_one "09_free_energy_active_inference.md" "Appendix: Free Energy and Active Inference"
build_one "10_symbols_glossary.md" "Appendix: Symbols and Glossary"

# Build combined document with page breaks between sections
# Each major section (01_introduction, 02_4d_namespaces, etc.) will start on a new page
# This ensures clean separation and professional document layout
{
  : > "$COMBINED_MD"
  for i in "${!MODULES[@]}"; do
    # Add page break before each section (except the first)
    if [ $i -gt 0 ]; then
      printf '\n\\newpage\n\n' >> "$COMBINED_MD"
    fi
    cat "$MARKDOWN_DIR/${MODULES[$i]}" >> "$COMBINED_MD"
    # Add extra spacing after each section for better separation
    if [ $i -lt $((${#MODULES[@]} - 1)) ]; then
      printf '\n\n' >> "$COMBINED_MD"
    fi
  done
}

# Generate TeX file for combined document first
echo "Generating combined TeX file..."

pandoc "$COMBINED_MD" \
  -f markdown+implicit_figures+tex_math_dollars+tex_math_single_backslash+raw_tex+autolink_bare_uris \
  -s \
  -V title="QuadMath: An Analytical Review of 4D and Quadray Coordinates" \
  -V author="$AUTHOR_TEX" \
  -V date="$DATE_STR" \
  "${COMMON_ARGS[@]}" \
  -o "$TEX_DIR/quadmath_review.tex"

echo "Generated combined TeX: $TEX_DIR/quadmath_review.tex"

# Compile combined TeX to PDF with XeLaTeX
echo "Compiling combined PDF..."
(
  cd "$OUTPUT_DIR"
  if command -v latexmk >/dev/null 2>&1; then
    latexmk -xelatex -interaction=nonstopmode -silent -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || latexmk -xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex"
    latexmk -c -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
  else
    xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
    xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
    xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
  fi
)

if [ -f "$PDF_DIR/quadmath_review.pdf" ]; then
  echo "✅ Built combined PDF: $PDF_DIR/quadmath_review.pdf"
else
  echo "❌ Failed to build combined PDF"
fi

echo "All outputs in: $OUTPUT_DIR"
echo "  PDFs: $PDF_DIR"
echo "  LaTeX: $TEX_DIR"
echo "  Data: $DATA_DIR"
echo "  Figures: $FIGURE_DIR"
