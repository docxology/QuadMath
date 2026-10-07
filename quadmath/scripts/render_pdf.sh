#!/bin/bash

# QuadMath PDF/LaTeX renderer - Improved Modular Version
# - Builds individual PDFs from Markdown modules (sections 01-18)
# - Builds combined PDF
# - Exports corresponding .tex files
# - Generates preamble from markdown source
# - All output folders can be safely purged
#
# Page margins: every PDF (per-module and combined) uses 1cm on all four sides.
# The combined build used 1.5cm left/right before this was unified; that
# difference was accidental.

set -euo pipefail
export LANG="${LANG:-C.UTF-8}"
# Optional flags (parsed from argv)
SKIP_FIGURES=false
for arg in "$@"; do
  case "$arg" in
    --skip-figures) SKIP_FIGURES=true ;;
    -h|--help)
      echo "Usage: render_pdf.sh [--skip-figures]"
      echo "  --skip-figures  Skip figure/data regeneration; still run glossary + validation + PDF builds"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

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
FIGURE_DIR="$OUTPUT_DIR/figures"
LATEX_TEMP_DIR="$OUTPUT_DIR/latex_temp"

# Author/metadata
AUTHOR_NAME="Daniel Ari Friedman"
AUTHOR_ORCID="0000-0001-6232-9096"
AUTHOR_EMAIL="daniel@activeinference.institute"
DOI="10.5281/zenodo.16887791"
AUTHOR_TEX="$AUTHOR_NAME\\\\ ORCID: $AUTHOR_ORCID\\\\ Email: $AUTHOR_EMAIL\\\\ DOI: $DOI"

# Sections built as individual PDFs and concatenated (in this order) into the
# combined PDF. Entries are "filename|title". Titles are inserted into LaTeX
# unescaped, so they must not contain & % $ # _ { } ~ ^ (check_module_coverage
# does not enforce this; tests/tools/test_render_pdf.py does).
MODULES=(
  "01_introduction.md|Introduction"
  "02_4d_namespaces.md|4D Namespaces: Coxeter.4D, Einstein.4D, Fuller.4D"
  "03_quadray_methods.md|Quadray Analytical Details and Methods"
  "04_optimization_in_4d.md|Optimization in 4D"
  "05_extensions.md|Extensions of 4D and Quadrays"
  "06_discussion.md|Discussion"
  "07_resources.md|Resources"
  "08_equations_appendix.md|Equations and Math Supplement"
  "09_free_energy_active_inference.md|Appendix: Free Energy and Active Inference"
  "10_symbols_glossary.md|Appendix: Symbols and Glossary"
  "11_ivm_field_learning.md|Static IVM Field Learning"
  "12_ivm_dynamics.md|Dynamics and Learning on the IVM Lattice"
  "13_lattice_tooling.md|Lattice Tooling: Omnidirectional Numbering and Nearest-Site Search"
  "14_conversions_spec.md|Conversions and Specification"
  "15_learning_evaluation.md|Learning and Evaluation on the IVM Lattice"
  "16_lattice_gallery.md|Lattice Visualization Gallery"
  "17_benchmarks_statistics.md|Benchmarks and Statistics"
  "18_stats_gallery.md|Statistics and Scaling Gallery"
)

# Markdown files in MARKDOWN_DIR that are deliberately not built as sections.
# Entries are "filename|reason"; the reason is mandatory.
EXCLUDED_MODULES=(
  "00_preamble.md|LaTeX preamble source; extracted into the -H header, not a section"
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

  if ! command -v fc-list >/dev/null 2>&1; then
    log_error "fc-list not found (fontconfig is required to verify PDF fonts)."
    exit 1
  fi
  local families
  families=$'\n'"$(fc-list : family | tr ',' '\n')"$'\n'
  local font
  for font in "DejaVu Serif" "DejaVu Sans Mono"; do
    case "$families" in
      *$'\n'"$font"$'\n'*) ;;
      *)
        log_error "Font '$font' is not installed; pandoc uses it as mainfont/monofont."
        echo "Install: brew install --cask font-dejavu  |  sudo apt-get install -y fonts-dejavu" >&2
        exit 1
        ;;
    esac
  done

  log_info "All dependencies satisfied"
}

# Every NN_*.md section must be listed in MODULES or EXCLUDED_MODULES, every
# MODULES file must exist, and every exclusion must give a reason.
check_module_coverage() {
  local entry file reason path i
  local listed=""
  for ((i = 0; i < ${#MODULES[@]}; i++)); do
    file="${MODULES[$i]%%|*}"
    if [ ! -f "$MARKDOWN_DIR/$file" ]; then
      log_error "MODULES lists a missing section file: $file"
      return 1
    fi
    listed="$listed|$file|"
  done
  for ((i = 0; i < ${#EXCLUDED_MODULES[@]}; i++)); do
    entry="${EXCLUDED_MODULES[$i]}"
    file="${entry%%|*}"
    reason="${entry#*|}"
    if [ -z "$reason" ] || [ "$reason" = "$entry" ]; then
      log_error "EXCLUDED_MODULES entry needs a reason: $file"
      return 1
    fi
    listed="$listed|$file|"
  done
  for path in "$MARKDOWN_DIR"/[0-9][0-9]_*.md; do
    [ -e "$path" ] || continue
    file="$(basename "$path")"
    case "$listed" in
      *"|$file|"*) ;;
      *)
        log_error "Section not listed in MODULES or EXCLUDED_MODULES: $file"
        return 1
        ;;
    esac
  done
  log_info "Section coverage OK"
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

  # Single generation pass: make_all_figures.py runs every figure/data
  # generator, fails hard if any generator fails, and writes the manifest.
  local scripts=(
    "make_all_figures.py"
    "generate_glossary.py"
    "validate_markdown.py"
  )


  for script in "${scripts[@]}"; do
    local script_path="$REPO_ROOT/quadmath/scripts/$script"
    if [ ! -f "$script_path" ]; then
      log_error "Missing script: $script_path"
      exit 1
    fi
    if [ "$script" = "make_all_figures.py" ] && [ "$SKIP_FIGURES" = "true" ]; then
      log_info "Skipping figure/data regeneration (--skip-figures)"
      continue
    fi
    local extra_args=()
    if [ "$script" = "validate_markdown.py" ]; then
      # Strict: validation issues fail the build instead of printing warnings
      extra_args+=("--strict")
    fi
    log_info "Running: $script"
    if $runner "$script_path" ${extra_args[@]+"${extra_args[@]}"} >/dev/null 2>&1; then
      log_info "✅ Success: $script"
    else
      log_error "❌ Failed: $script"
      exit 1
    fi
  done

  log_info "Figure generation complete"
}

# =============================================================================
# PDF BUILDING
# =============================================================================

# Markdown -> TeX with the shared QuadMath pandoc options (one set for both the
# per-section and combined builds). Arguments: input markdown, title, preamble
# TeX, output TeX.
pandoc_to_tex() {
  local in_md="$1"
  local title="$2"
  local preamble_tex="$3"
  local out_tex="$4"

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
    -o "$out_tex"
  )

  pandoc "$in_md" "${pandoc_args[@]}"
}

# Three-pass xelatex build of $TEX_DIR/<base>.tex into $PDF_DIR/<base>.pdf.
# A second pass runs only when the first aux file shows unresolved references;
# otherwise the build runs two passes when there is no aux file. A non-zero exit
# from any pass fails the build. Returns 0 only if the PDF exists.
compile_tex_to_pdf() {
  local base="$1"
  local tex_file="$TEX_DIR/${base}.tex"
  local pdf_out="$PDF_DIR/${base}.pdf"
  local aux_file="$PDF_DIR/${base}.aux"
  local xelatex_log="$LATEX_TEMP_DIR/${base}.xelatex.log"
  rm -f "$pdf_out" "$xelatex_log"

  local compile_status=0
  (
    cd "$OUTPUT_DIR"

    run_xelatex() {
      xelatex -interaction=nonstopmode -output-directory="$PDF_DIR" "$tex_file" >>"$xelatex_log" 2>&1
    }

    # First run - generate initial PDF
    xelatex_status=0
    if run_xelatex; then
      log_info "First xelatex run completed: $base"
    else
      xelatex_status=$?
      log_warn "First xelatex run had warnings (continuing): $base"
    fi

    # Check if we need additional runs by looking for unresolved references
    if [ -f "$aux_file" ]; then
      if grep -q "\\\@ref" "$aux_file" 2>/dev/null || grep -q "\\\@cite" "$aux_file" 2>/dev/null; then
        log_info "Unresolved references detected, running second xelatex pass: $base"
        run_xelatex || xelatex_status=$?
      fi

      # Final run to ensure all references are resolved
      log_info "Running final xelatex pass: $base"
      run_xelatex || xelatex_status=$?
    else
      # If no .aux file, run twice to be safe
      log_info "No .aux file, running two xelatex passes: $base"
      run_xelatex || xelatex_status=$?
      run_xelatex || xelatex_status=$?
    fi

    # Clean up auxiliary files
    rm -f "$PDF_DIR/${base}.aux" "$PDF_DIR/${base}.log" "$PDF_DIR/${base}.toc" 2>/dev/null || true
    exit "$xelatex_status"
  ) || compile_status=$?

  if [ "$compile_status" -eq 0 ] && [ -f "$pdf_out" ]; then
    rm -f "$xelatex_log"
    log_info "✅ Built: $pdf_out"
    return 0
  fi
  log_error "❌ Failed to build: $pdf_out (xelatex exit $compile_status)"
  if [ -f "$xelatex_log" ]; then
    log_error "Last 20 lines of $xelatex_log:"
    tail -n 20 "$xelatex_log" >&2
  fi
  return 1
}

build_one() {
  local in_md="$1"
  local title="$2"
  local preamble_tex="$3"
  local base="${in_md%.md}"
  local tex_out="$TEX_DIR/${base}.tex"

  log_info "Building: $in_md -> $base.pdf"
  if pandoc_to_tex "$MARKDOWN_DIR/$in_md" "$title" "$preamble_tex" "$tex_out"; then
    log_info "Generated TeX: $tex_out"
  else
    log_error "Failed to generate TeX for $in_md"
    return 1
  fi

  log_info "Compiling PDF: $base.pdf"
  compile_tex_to_pdf "$base"
}

build_combined() {
  local preamble_tex="$1"
  local combined_md="$OUTPUT_DIR/quadmath_review.md"
  local combined_tex="$TEX_DIR/quadmath_review.tex"

  log_info "Building combined document..."

  # Build combined markdown with page breaks
  {
    : > "$combined_md"
    for i in "${!MODULES[@]}"; do
      # Add page break before each section (except the first)
      if [ $i -gt 0 ]; then
        printf '\n\\newpage\n\n' >> "$combined_md"
      fi
      cat "$MARKDOWN_DIR/${MODULES[$i]%%|*}" >> "$combined_md"
      # Add extra spacing after each section for better separation
      if [ $i -lt $((${#MODULES[@]} - 1)) ]; then
        printf '\n\n' >> "$combined_md"
      fi
    done
  }

  log_info "Generated combined markdown: $combined_md"

  log_info "Generating combined TeX file..."
  if pandoc_to_tex "$combined_md" "QuadMath: An Analytical Review of 4D and Quadray Coordinates" "$preamble_tex" "$combined_tex"; then
    log_info "Generated combined TeX: $combined_tex"
  else
    log_error "Failed to generate combined TeX"
    return 1
  fi

  log_info "Compiling combined PDF..."
  compile_tex_to_pdf "quadmath_review"
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
  check_module_coverage || exit 1
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

  for entry in "${MODULES[@]}"; do
    local module="${entry%%|*}"
    local title="${entry#*|}"
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
    failed_modules+=("quadmath_review.pdf")
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

# Run main unless this file is sourced (tests source it to reach MODULES and
# the build helpers).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
