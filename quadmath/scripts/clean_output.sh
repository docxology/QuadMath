#!/bin/bash

# QuadMath Output Cleanup Script
# Safely removes all generated output since everything is regenerated from markdown

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/quadmath/output"

echo "🧹 Cleaning QuadMath output directories..."
echo "Repository root: $REPO_ROOT"

# Clean generated output (regeneratable; note: currently git-tracked, so
# cleaning produces a large diff — commit or stash deliberately). The
# per-folder AGENTS.md and README.md files are kept.
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing generated files under: $OUTPUT_DIR (keeping AGENTS.md and README.md)"
    find "$OUTPUT_DIR" -type f ! \( -name AGENTS.md -o -name README.md \) -delete
    find "$OUTPUT_DIR" -mindepth 1 -depth -type d -empty -delete
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
