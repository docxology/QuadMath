# QuadMath Script Improvements

## Overview

The `render_pdf.sh` script has been significantly improved for better modularity, logging, and maintainability. The workflow is now completely markdown-centric, with all output folders being safely disposable.

## Key Improvements

### 1. **Markdown-Centric Workflow**
- **Before**: Required separate `quadmath/latex/preamble.tex` file
- **After**: Preamble is generated from `quadmath/markdown/00_preamble.md` at runtime
- **Benefit**: All writing and editing happens in the `markdown/` folder

### 2. **Improved Logging System**
- **Structured logging** with timestamps and log levels (DEBUG, INFO, WARN, ERROR)
- **Configurable verbosity** via `LOG_LEVEL` environment variable
- **Better error reporting** with clear success/failure indicators
- **Progress tracking** for long-running operations

### 3. **Modular Architecture**
- **Separated concerns** into distinct functions:
  - `check_dependencies()` - Validate system requirements
  - `setup_directories()` - Create and clean output directories
  - `generate_preamble()` - Extract LaTeX from markdown
  - `run_generation_scripts()` - Execute figure generation
  - `build_one()` - Build individual module PDFs
  - `build_combined()` - Build combined document
  - `main()` - Orchestrate the entire process

### 4. **Disposable Output Folders**
- **All output directories** can be safely purged and regenerated
- **No persistent files** outside of source markdown
- **Clean separation** between source (markdown) and generated content

### 5. **Better Error Handling**
- **Graceful degradation** when scripts fail
- **Detailed error messages** with context
- **Failure tracking** and reporting
- **Build continuation** even when some components fail

## Usage

### Basic Usage
```bash
# Generate all PDFs from markdown sources
./quadmath/scripts/render_pdf.sh
```

### Logging Control
```bash
# Set log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
LOG_LEVEL=0 ./quadmath/scripts/render_pdf.sh  # Verbose debug output
LOG_LEVEL=2 ./quadmath/scripts/render_pdf.sh  # Only warnings and errors
```

### Clean Output
```bash
# Remove all generated content
./quadmath/scripts/clean_output.sh

# Regenerate everything
./quadmath/scripts/render_pdf.sh
```

## File Structure

```
quadmath/
├── markdown/                    # 📝 Source content (never deleted)
│   ├── 00_preamble.md          # LaTeX preamble in markdown
│   ├── 00_front_matter.md      # Front matter
│   ├── 01_introduction.md      # Introduction
│   └── ...                     # Other content modules
├── scripts/                     # 🔧 Build scripts
│   ├── render_pdf.sh           # Main build script (improved)
│   ├── clean_output.sh         # Cleanup script (new)
│   └── ...                     # Other generation scripts
├── output/                      # 📤 Generated content (disposable)
│   ├── pdf/                    # Generated PDFs
│   ├── tex/                    # Generated LaTeX
│   ├── figures/                # Generated figures
│   └── data/                   # Generated data
└── latex/                      # 📄 LaTeX files (disposable)
```

## Benefits

### For Developers
- **Single source of truth**: All content in markdown
- **Version control friendly**: Only source files tracked
- **Easy cleanup**: Regenerate everything with one command
- **Better debugging**: Structured logging and error reporting

### For Maintainers
- **Modular code**: Easy to modify individual components
- **Consistent interface**: Standardized logging and error handling
- **Self-documenting**: Clear function names and structure
- **Robust**: Better error handling and recovery

### For Users
- **Simple workflow**: Edit markdown, run script, get PDFs
- **Clear feedback**: Know exactly what's happening and what failed
- **Reproducible**: Same output every time from same sources
- **Clean**: No leftover files or artifacts

## Migration Notes

### From Old Script
- **No changes needed** to existing markdown files
- **Preamble content** moved from `latex/preamble.tex` to `markdown/00_preamble.md`
- **Output structure** remains the same
- **Script interface** unchanged

### New Features
- **Better logging**: More informative output
- **Cleaner output**: All generated content in one place
- **Faster debugging**: Clear error messages and progress
- **Easier maintenance**: Modular, well-structured code

## Troubleshooting

### Common Issues
1. **Missing dependencies**: Script checks and reports missing tools
2. **Failed scripts**: Individual script failures don't stop the build
3. **LaTeX errors**: Better error reporting and fallback compilation
4. **Permission issues**: Scripts are executable by default

### Debug Mode
```bash
# Enable debug logging for troubleshooting
LOG_LEVEL=0 ./quadmath/scripts/render_pdf.sh
```

### Clean Rebuild
```bash
# Start fresh if something goes wrong
./quadmath/scripts/clean_output.sh
./quadmath/scripts/render_pdf.sh
```

## Future Enhancements

- **Parallel builds**: Build multiple modules simultaneously
- **Incremental builds**: Only rebuild changed modules
- **Dependency tracking**: Track figure dependencies for smart rebuilds
- **Configuration file**: Externalize build parameters
- **CI/CD integration**: Better integration with automated builds
