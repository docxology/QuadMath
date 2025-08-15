#!/bin/bash

# QuadMath Output Cleanup Script
# Safely removes all generated output since everything is regenerated from markdown

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/quadmath/output"
LATEX_DIR="$REPO_ROOT/quadmath/latex"

echo "🧹 Cleaning QuadMath output directories..."
echo "Repository root: $REPO_ROOT"

# Clean output directory (all disposable)
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
    echo "✅ Output directory cleaned"
else
    echo "ℹ️  Output directory not found: $OUTPUT_DIR"
fi

# Clean latex directory (all disposable)
if [ -d "$LATEX_DIR" ]; then
    echo "Removing latex directory: $LATEX_DIR"
    rm -rf "$LATEX_DIR"
    echo "✅ Latex directory cleaned"
else
    echo "ℹ️  Latex directory not found: $LATEX_DIR"
fi

echo ""
echo "🎯 All output directories cleaned!"
echo "💡 Run 'quadmath/scripts/render_pdf.sh' to regenerate everything from markdown sources"
echo ""
echo "📁 Markdown sources remain intact in: $REPO_ROOT/quadmath/markdown/"
echo "🔧 Scripts remain intact in: $REPO_ROOT/quadmath/scripts/"
echo "📚 Source code remains intact in: $REPO_ROOT/src/"
