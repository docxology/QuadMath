#!/bin/bash

# QuadMath Output Cleanup Script
# Safely removes all generated output since everything is regenerated from markdown

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/quadmath/output"

echo "🧹 Cleaning QuadMath output directories..."
echo "Repository root: $REPO_ROOT"

# Clean output directory (regeneratable; note: currently git-tracked, so
# cleaning produces a large diff — commit or stash deliberately)
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing output directory: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
    echo "✅ Output directory cleaned"
else
    echo "ℹ️  Output directory not found: $OUTPUT_DIR"
fi

echo "💡 Run 'quadmath/scripts/render_pdf.sh' to regenerate everything from markdown sources"

echo ""
echo "🎯 All output directories cleaned!"
echo "💡 Run 'quadmath/scripts/render_pdf.sh' to regenerate everything from markdown sources"
echo ""
echo "📁 Markdown sources remain intact in: $REPO_ROOT/quadmath/markdown/"
echo "🔧 Scripts remain intact in: $REPO_ROOT/quadmath/scripts/"
echo "📚 Source code remains intact in: $REPO_ROOT/src/"
