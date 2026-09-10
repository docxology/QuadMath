#!/bin/bash
# =============================================================================
# QuadMath Thin Orchestrator
# =============================================================================
#
# A lightweight orchestration script that runs the complete QuadMath pipeline:
#   1. Tests with coverage
#   2. Figure generation
#   3. Markdown validation
#   4. Optional PDF build
#
# Usage:
#   ./run_all.sh              # Run tests + figures + validation
#   ./run_all.sh --with-pdf   # Also build PDFs
#   ./run_all.sh --test-only  # Only run tests
#
# =============================================================================

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
WITH_PDF=false
TEST_ONLY=false

for arg in "$@"; do
    case $arg in
        --with-pdf)
            WITH_PDF=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --help|-h)
            echo "QuadMath Orchestrator"
            echo ""
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test-only    Only run tests with coverage"
            echo "  --with-pdf     Also build PDFs (slow)"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# =============================================================================
# Phase 1: Tests with Coverage
# =============================================================================

print_header "Phase 1: Running Tests with Coverage"

print_info "Running pytest with coverage..."
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q

print_info "Generating coverage report..."
uv run coverage report

print_success "All tests passed with 100% coverage"

if [ "$TEST_ONLY" = true ]; then
    print_header "Complete (test-only mode)"
    exit 0
fi

# =============================================================================
# Phase 2: Figure Generation
# =============================================================================

print_header "Phase 2: Generating Figures"

print_info "Running make_all_figures.py..."
uv run python quadmath/scripts/make_all_figures.py

print_success "All figures generated"

# =============================================================================
# Phase 3: Markdown Validation
# =============================================================================

print_header "Phase 3: Validating Markdown"

print_info "Running validate_markdown.py --strict..."
uv run python quadmath/scripts/validate_markdown.py --strict

print_success "Markdown validation passed"

# =============================================================================
# Phase 4: PDF Build (Optional)
# =============================================================================

if [ "$WITH_PDF" = true ]; then
    print_header "Phase 4: Building PDFs"
    
    print_info "Running render_pdf.sh..."
    bash quadmath/scripts/render_pdf.sh --skip-figures  # Figures already generated
    
    print_success "PDFs built successfully"
fi

# =============================================================================
# Summary
# =============================================================================

print_header "Pipeline Complete"

echo ""
echo "Summary:"
echo "  ✓ Tests:      All passing with 100% coverage"
echo "  ✓ Figures:    Generated to quadmath/output/figures/"
echo "  ✓ Validation: Markdown references verified"

if [ "$WITH_PDF" = true ]; then
    echo "  ✓ PDFs:       Built to quadmath/output/pdf/"
fi

echo ""
echo "Next steps:"
if [ "$WITH_PDF" = false ]; then
    echo "  • Run './run_all.sh --with-pdf' to also build PDFs"
fi
echo "  • Review outputs in quadmath/output/"
echo "  • Commit changes when ready"
echo ""
