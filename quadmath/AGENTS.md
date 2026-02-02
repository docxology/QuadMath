# AGENTS.md - Build System Directory

## Purpose

This directory contains the complete build system for generating figures, PDFs, and LaTeX exports from QuadMath manuscript sources.

## Agent Guidelines

### Critical Rules

1. **Scripts import from `src/` only** - No code duplication
2. **Headless plotting required** - Use `MPLBACKEND=Agg`
3. **Fixed RNG seeds** - Deterministic output generation
4. **Print output paths** - Scripts must print generated file paths

### Directory Roles

| Directory | Role | Disposable? |
|-----------|------|-------------|
| `markdown/` | Manuscript source | ❌ No |
| `scripts/` | Build scripts | ❌ No |
| `output/` | Generated artifacts | ✅ Yes |

### Before Modifying Scripts

1. **Run tests**: `uv run pytest -q`
2. **Check that script runs**: `uv run python quadmath/scripts/<script>.py`
3. **Verify outputs generated**: Check `quadmath/output/`

### Before Modifying Markdown

1. **Validate existing state**: `uv run python quadmath/scripts/validate_markdown.py`
2. **Check image references exist**: All `![...]()` paths must be valid
3. **Verify equation labels unique**: No duplicate `\label{eq:...}`

## Script Development Standards

### Figure Generation Scripts

```python
#!/usr/bin/env python
"""Generate [description] figures.

Outputs:
    quadmath/output/figures/figure_name.png
    quadmath/output/data/data_name.csv
"""
from __future__ import annotations

import os
os.environ["MPLBACKEND"] = "Agg"  # Headless mode - BEFORE importing matplotlib

import matplotlib.pyplot as plt
import numpy as np

# Import from src/ only
from quadray import Quadray
from paths import get_figure_dir, get_data_dir

def main():
    np.random.seed(42)  # Deterministic
    
    # Generate figure
    fig, ax = plt.subplots()
    # ... plotting code ...
    
    # Save with explicit path
    out_path = get_figure_dir() / "figure_name.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Print output path (for manifest)
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    main()
```

### Key Requirements

1. **Set MPLBACKEND before import**:

   ```python
   import os
   os.environ["MPLBACKEND"] = "Agg"
   import matplotlib.pyplot as plt
   ```

2. **Use path utilities**:

   ```python
   from paths import get_figure_dir, get_data_dir, get_output_dir
   ```

3. **Print outputs**:

   ```python
   print(f"Generated: {output_path}")
   ```

4. **Close figures**:

   ```python
   plt.close(fig)  # Prevent memory leaks
   ```

## Markdown Standards

### Image References

```markdown
<!-- Correct: relative path from markdown/ directory -->
![Caption](../output/figures/figure_name.png)

<!-- Wrong: absolute path -->
![Caption](/Users/.../figure.png)
```

### Equation Labels

```markdown
<!-- Must be unique across all markdown files -->
\begin{equation}
\label{eq:unique_name}
E = mc^2
\end{equation}
```

### Cross-References

```markdown
<!-- Reference equations -->
See Equation \eqref{eq:unique_name}.

<!-- Reference figures -->
See Figure \ref{fig:figure_name}.
```

### Internal Links

```markdown
<!-- Link to other sections -->
See [Section Title](02_4d_namespaces.md#section-anchor)
```

## Build Pipeline Modifications

### Adding a New Generation Script

1. Create script in `scripts/`:

   ```python
   # quadmath/scripts/new_figure.py
   ```

2. Add to `make_all_figures.py`:

   ```python
   scripts = [
       # ...existing...
       "new_figure.py",
   ]
   ```

3. Add to `render_pdf.sh` (if needed for PDF build):

   ```bash
   log_info "Running new_figure.py..."
   uv run python "$SCRIPT_DIR/new_figure.py"
   ```

### Adding a New Markdown Section

1. Create file with proper numbering:

   ```
   quadmath/markdown/XX_section_name.md
   ```

2. Add to `MODULES` array in `render_pdf.sh`:

   ```bash
   MODULES=(
       # ...existing...
       "XX_section_name.md"
   )
   ```

3. Update `00_preamble.md` table of contents if needed

## Validation Commands

```bash
# Validate markdown references
uv run python quadmath/scripts/validate_markdown.py

# Check for broken image links
uv run python quadmath/scripts/validate_markdown.py --strict

# Regenerate glossary from src/
uv run python quadmath/scripts/generate_glossary.py
```

## Common Issues

### Missing Images

```
ERROR: Image not found: ../output/figures/missing.png
```

**Solution**: Run the generation script that creates this figure:

```bash
uv run python quadmath/scripts/<relevant_script>.py
```

### Duplicate Equation Labels

```
ERROR: Duplicate label: eq:example
```

**Solution**: Rename one of the duplicate labels to be unique

### LaTeX Build Failures

```
! LaTeX Error: File 'something.sty' not found.
```

**Solution**: Install missing LaTeX package:

```bash
# macOS
tlmgr install package-name

# Ubuntu
sudo apt-get install texlive-latex-extra
```

## Quality Checklist

Before committing build system changes:

- [ ] All scripts run without errors
- [ ] `validate_markdown.py` passes
- [ ] `render_pdf.sh` completes successfully
- [ ] Generated PDFs render correctly
- [ ] No hardcoded paths
- [ ] Outputs are deterministic (same input → same output)
