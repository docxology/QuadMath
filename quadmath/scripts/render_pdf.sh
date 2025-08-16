#!/bin/bash

# QuadMath PDF/LaTeX renderer - Improved Modular Version
# - Builds individual PDFs from Markdown modules
# - Builds combined PDF
# - Exports corresponding .tex files
# - Generates preamble from markdown source
# - All output folders can be safely purged

set -euo pipefail
export LANG="${LANG:-C.UTF-8}"

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MARKDOWN_DIR="$REPO_ROOT/quadmath/markdown"
OUTPUT_DIR="$REPO_ROOT/quadmath/output"
PREAMBLE_MD="$MARKDOWN_DIR/00_preamble.md"

# Output subdirectories (all disposable)
PDF_DIR="$OUTPUT_DIR/pdf"
TEX_DIR="$OUTPUT_DIR/tex"
DATA_DIR="$OUTPUT_DIR/data"
FIGURE_DIR="$OUTPUT_DIR/output"
LATEX_TEMP_DIR="$OUTPUT_DIR/latex_temp"

# Author/metadata
AUTHOR_NAME="Daniel Ari Friedman"
AUTHOR_ORCID="0000-0001-6232-9096"
AUTHOR_EMAIL="daniel@activeinference.institute"
DOI="10.5281/zenodo.16887800"
AUTHOR_TEX="$AUTHOR_NAME\\\\ ORCID: $AUTHOR_ORCID\\\\ Email: $AUTHOR_EMAIL\\\\ DOI: $DOI"

# Modules (ordered)
MODULES=(
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

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

# Log levels
LOG_DEBUG=0
LOG_INFO=1
LOG_WARN=2
LOG_ERROR=3

# Current log level (can be set via LOG_LEVEL environment variable)
LOG_LEVEL="${LOG_LEVEL:-$LOG_INFO}"

log() {
  local level="$1"
  local message="$2"
  local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  
  if [ "$level" -ge "$LOG_LEVEL" ]; then
    case "$level" in
      $LOG_DEBUG) echo "[$timestamp] [DEBUG] $message" ;;
      $LOG_INFO)  echo "[$timestamp] [INFO]  $message" ;;
      $LOG_WARN)  echo "[$timestamp] [WARN]  $message" >&2 ;;
      $LOG_ERROR) echo "[$timestamp] [ERROR] $message" >&2 ;;
    esac
  fi
}

log_info() { log $LOG_INFO "$1"; }
log_warn() { log $LOG_WARN "$1"; }
log_error() { log $LOG_ERROR "$1"; }
log_debug() { log $LOG_DEBUG "$1"; }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

check_dependencies() {
  log_info "Checking dependencies..."
  
  if ! command -v pandoc >/dev/null 2>&1; then
    log_error "pandoc is not installed."
    echo "Install: sudo apt-get install -y pandoc" >&2
    exit 1
  fi
  
  if ! command -v xelatex >/dev/null 2>&1; then
    log_error "xelatex not found. Install TeX Live:"
    echo "sudo apt-get install -y texlive-xetex texlive-fonts-recommended fonts-dejavu" >&2
    exit 1
  fi
  
  log_info "All dependencies satisfied"
}

setup_directories() {
  log_info "Setting up output directories..."
  
  # Create all output directories (these can be safely purged)
  mkdir -p "$OUTPUT_DIR" "$PDF_DIR" "$TEX_DIR" "$DATA_DIR" "$FIGURE_DIR" "$LATEX_TEMP_DIR"
  
  # Clean up any existing content
  rm -rf "$LATEX_TEMP_DIR"/*
  
  log_info "Output directories ready"
}

# =============================================================================
# FIGURE GENERATION
# =============================================================================

run_generation_scripts() {
  log_info "Running figure/data generation scripts..."
  
  local runner
  if command -v uv >/dev/null 2>&1; then
    runner="uv run python"
  else
    runner="python3"
  fi
  
  export MPLBACKEND=Agg
  log_info "Using runner: $runner"
  
  # Array of scripts to run
  local scripts=(
    "ivm_neighbors.py"
    "quadray_clouds.py"
    "volumes_demo.py"
    "simplex_animation.py"
    "graphical_abstract_quadray.py"
    "polyhedra_quadray_constructions.py"
    "sympy_formalisms.py"
    "information_demo.py"
    "active_inference_figures.py"
    "generate_glossary.py"
    "validate_markdown.py"
    "make_all_figures.py"
  )
  
  local failed_scripts=()
  
  for script in "${scripts[@]}"; do
    local script_path="$REPO_ROOT/quadmath/scripts/$script"
    if [ -f "$script_path" ]; then
      log_info "Running: $script"
      if $runner "$script_path" >/dev/null 2>&1; then
        log_info "✅ Success: $script"
      else
        log_warn "⚠️  Failed: $script (continuing)"
        failed_scripts+=("$script")
      fi
    else
      log_debug "Skipping: $script (not found)"
    fi
  done
  
  if [ ${#failed_scripts[@]} -gt 0 ]; then
    log_warn "Some scripts failed: ${failed_scripts[*]}"
  fi
  
  log_info "Figure generation complete"
}

# =============================================================================
# PDF BUILDING
# =============================================================================

build_one() {
  local in_md="$1"
  local title="$2"
  local preamble_tex="$3"
  local base="${in_md%.md}"
  local pdf_out="$PDF_DIR/${base}.pdf"
  local tex_out="$TEX_DIR/${base}.tex"
  
  log_info "Building: $in_md -> $base.pdf"
  
  # Generate TeX file first
  local pandoc_args=(
    -f markdown+implicit_figures+tex_math_dollars+tex_math_single_backslash+raw_tex+autolink_bare_uris
    -s
    -V title="$title"
    -V author="$AUTHOR_TEX"
    -V date="$(date '+%B %d, %Y')"
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
    --resource-path="$MARKDOWN_DIR:$OUTPUT_DIR:$LATEX_TEMP_DIR:$REPO_ROOT"
    -H "$preamble_tex"
    -o "$tex_out"
  )
  
  if pandoc "$MARKDOWN_DIR/$in_md" "${pandoc_args[@]}"; then
    log_info "Generated TeX: $tex_out"
  else
    log_error "Failed to generate TeX for $in_md"
    return 1
  fi

  # Compile TeX to PDF with Xelatex
  log_info "Compiling PDF: $base.pdf"
  (
    cd "$OUTPUT_DIR"
    
    # Use optimized xelatex compilation
    log_info "Using optimized xelatex compilation"
    
    # First run - generate initial PDF
    if xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1; then
      log_info "First xelatex run completed"
    else
      log_warn "First xelatex run had warnings (continuing)"
    fi
    
    # Check if we need additional runs by looking for unresolved references
    local aux_file="$PDF_DIR/${base}.aux"
    if [ -f "$aux_file" ]; then
      # Look for unresolved references in .aux file
      if grep -q "\\\@ref" "$aux_file" 2>/dev/null || grep -q "\\\@cite" "$aux_file" 2>/dev/null; then
        log_info "Unresolved references detected, running second xelatex pass"
        xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
      fi
      
      # Final run to ensure all references are resolved
      log_info "Running final xelatex pass"
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
    else
      # If no .aux file, run twice to be safe
      log_info "No .aux file, running two xelatex passes"
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/$base.tex" >/dev/null 2>&1 || true
    fi
    
    # Clean up auxiliary files
    rm -f "$PDF_DIR/${base}.aux" "$PDF_DIR/${base}.log" "$PDF_DIR/${base}.toc" 2>/dev/null || true
  )
  
  if [ -f "$pdf_out" ]; then
    log_info "✅ Built: $pdf_out"
    return 0
  else
    log_error "❌ Failed to build: $pdf_out"
    return 1
  fi
}

build_combined() {
  local preamble_tex="$1"
  local combined_md="$OUTPUT_DIR/quadmath_review.md"
  
  log_info "Building combined document..."
  
  # Build combined markdown with page breaks
  {
    : > "$combined_md"
    for i in "${!MODULES[@]}"; do
      # Add page break before each section (except the first)
      if [ $i -gt 0 ]; then
        printf '\n\\newpage\n\n' >> "$combined_md"
      fi
      cat "$MARKDOWN_DIR/${MODULES[$i]}" >> "$combined_md"
      # Add extra spacing after each section for better separation
      if [ $i -lt $((${#MODULES[@]} - 1)) ]; then
        printf '\n\n' >> "$combined_md"
      fi
    done
  }
  
  log_info "Generated combined markdown: $combined_md"
  
  # Generate TeX file for combined document
  log_info "Generating combined TeX file..."
  
  local pandoc_args=(
    -f markdown+implicit_figures+tex_math_dollars+tex_math_single_backslash+raw_tex+autolink_bare_uris
    -s
    -V title="QuadMath: An Analytical Review of 4D and Quadray Coordinates"
    -V author="$AUTHOR_TEX"
    -V date="$(date '+%B %d, %Y')"
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
    -V geometry:left=1.5cm
    -V geometry:right=1.5cm
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
    --resource-path="$MARKDOWN_DIR:$OUTPUT_DIR:$LATEX_TEMP_DIR:$REPO_ROOT"
    -H "$preamble_tex"
    -o "$TEX_DIR/quadmath_review.tex"
  )
  
  if pandoc "$combined_md" "${pandoc_args[@]}"; then
    log_info "Generated combined TeX: $TEX_DIR/quadmath_review.tex"
  else
    log_error "Failed to generate combined TeX"
    return 1
  fi

  # Compile combined TeX to PDF
  log_info "Compiling combined PDF..."
  (
    cd "$OUTPUT_DIR"
    
    # Use optimized xelatex compilation for combined document
    log_info "Using optimized xelatex compilation for combined document"
    
    # First run - generate initial PDF
    if xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1; then
      log_info "First xelatex run completed for combined document"
    else
      log_warn "First xelatex run had warnings (continuing)"
    fi
    
    # Check if we need additional runs by looking for unresolved references
    local aux_file="$PDF_DIR/quadmath_review.aux"
    if [ -f "$aux_file" ]; then
      # Look for unresolved references in .aux file
      if grep -q "\\\@ref" "$aux_file" 2>/dev/null || grep -q "\\\@cite" "$aux_file" 2>/dev/null; then
        log_info "Unresolved references detected, running second xelatex pass"
        xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
      fi
      
      # Final run to ensure all references are resolved
      log_info "Running final xelatex pass for combined document"
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
    else
      # If no .aux file, run twice to be safe
      log_info "No .aux file, running two xelatex passes for combined document"
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$TEX_DIR/quadmath_review.tex" >/dev/null 2>&1 || true
    fi
    
    # Clean up auxiliary files
    rm -f "$PDF_DIR/quadmath_review.aux" "$PDF_DIR/quadmath_review.log" "$PDF_DIR/quadmath_review.toc" 2>/dev/null || true
  )
  
  if [ -f "$PDF_DIR/quadmath_review.pdf" ]; then
    log_info "✅ Built combined PDF: $PDF_DIR/quadmath_review.pdf"
    return 0
  else
    log_error "❌ Failed to build combined PDF"
    return 1
  fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
  local start_time=$(date +%s)
  
  log_info "Starting QuadMath PDF generation..."
  log_info "Repository root: $REPO_ROOT"
  log_info "Markdown source: $MARKDOWN_DIR"
  log_info "Output directory: $OUTPUT_DIR"
  
  # Setup and validation
  check_dependencies
  setup_directories
  
  # Generate preamble from markdown (ONCE)
  log_info "Generating LaTeX preamble from markdown..."
  local preamble_tex
  if [ ! -f "$PREAMBLE_MD" ]; then
    log_error "Preamble markdown file not found: $PREAMBLE_MD"
    exit 1
  fi
  
  # Extract LaTeX content from the markdown file
  preamble_tex="$LATEX_TEMP_DIR/preamble.tex"
  
  # Extract content between ```latex and ``` blocks
  sed -n '/^```latex$/,/^```$/p' "$PREAMBLE_MD" | sed '1d;$d' > "$preamble_tex"
  
  if [ ! -s "$preamble_tex" ]; then
    log_error "Failed to extract LaTeX preamble from $PREAMBLE_MD"
    exit 1
  fi
  
  log_info "Generated preamble: $preamble_tex"
  
  # Run figure generation
  run_generation_scripts
  
  # Build individual modules
  log_info "Building individual module PDFs..."
  local failed_modules=()
  
  for module in "${MODULES[@]}"; do
    local title
    case "$module" in
      "01_introduction.md") title="Introduction" ;;
      "02_4d_namespaces.md") title="4D Namespaces: Coxeter.4D, Einstein.4D, Fuller.4D" ;;
      "03_quadray_methods.md") title="Quadray Analytical Details and Methods" ;;
      "04_optimization_in_4d.md") title="Optimization in 4D" ;;
      "05_extensions.md") title="Extensions of 4D and Quadrays" ;;
      "06_discussion.md") title="Discussion" ;;
      "07_resources.md") title="Resources" ;;
      "08_equations_appendix.md") title="Equations and Math Supplement" ;;
      "09_free_energy_active_inference.md") title="Appendix: Free Energy and Active Inference" ;;
      "10_symbols_glossary.md") title="Appendix: Symbols and Glossary" ;;
      *) title="${module%.md}" ;;
    esac
    
    if build_one "$module" "$title" "$preamble_tex"; then
      log_info "✅ Module built successfully: $module"
    else
      log_error "❌ Module failed: $module"
      failed_modules+=("$module")
    fi
  done
  
  # Build combined document
  if build_combined "$preamble_tex"; then
    log_info "✅ Combined document built successfully"
  else
    log_error "❌ Combined document failed"
  fi
  
  # Summary
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  
  log_info "Build complete in ${duration}s"
  log_info "All outputs in: $OUTPUT_DIR"
  log_info "  PDFs: $PDF_DIR"
  log_info "  LaTeX: $TEX_DIR"
  log_info "  Data: $DATA_DIR"
  log_info "  Figures: $FIGURE_DIR"
  
  if [ ${#failed_modules[@]} -gt 0 ]; then
    log_warn "Failed modules: ${failed_modules[*]}"
    exit 1
  else
    log_info "All modules built successfully!"
  fi
}

# Run main function
main "$@"
