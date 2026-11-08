# QuadMath Development Workflow: Isolated Work and Integrated Evidence

Runtime tests, documentation checks, figure rendering and PDF compilation are
separate gates. No single command establishes complete source/text coherence.
See [architecture](ARCHITECTURE.md) and [documentation navigation](docs/README.md).

## Development Phase

1. Read root and scoped `AGENTS.md`, inspect the implementing source, and find
   callers before changing public contracts.
2. Change reusable numerical/rendering code in `src/quadmath/`; keep scripts
   thin. Add regressions for observable numerical, geometry, resource, error
   or transition behavior rather than pinning wording or saved test counts.
3. Update canonical mathematics in `quadmath/markdown/` or operational guidance
   in `docs/` as appropriate. Preserve equation/code anchors and fix inbound
   links when headings change.
4. Preserve user-generated outputs and immutable release artifacts. During
   delegated implementation, skip builds/tests/lint/formatters; report focused
   consumer probes separately and let the integration owner verify once.

## Python environment

Commands below run from the repository root and explicitly select scientific
Python 3.13. `.python-version` remains unchanged; bare `uv run` can choose its
pinned interpreter instead. The declared runtime floor is Python 3.11, not a
claim that every scientific dependency has wheels for every later interpreter.

```bash
uv sync --python 3.13
uv run --python 3.13 python -c "import quadmath; from quadmath.core.quadray import Quadray; print(Quadray(2, 1, 1, 0).normalize())"
```

Use `python -I` for out-of-checkout numerical scripts so unrelated working-directory
modules do not shadow dependencies. An isolated interpreter does not include the
checkout's `src` automatically: use an installed package or explicitly insert
the known checkout `src` path in the consumer script.

## Validation Phase

These commands are for the integration owner when verification is assigned,
not a requirement for every concurrently editing worker.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 coverage run -m pytest -q
uv run --python 3.13 coverage report
uv run --python 3.13 python quadmath/scripts/generate_glossary.py --check
uv run --python 3.13 python quadmath/scripts/validate_markdown.py --strict
uv run --python 3.13 python docs/development/check_links.py
```

- Coverage uses branches and the configured aggregate floor. Report the actual
  result, never infer correctness or a current percentage from configuration.
- `generate_glossary.py --check` checks freshness; without `--check` it updates
  the marked canonical API block.
- The manuscript validator checks its implemented source-tree syntax/reference
  rules, not every mathematical claim or every renderer-dependent cross-reference.
- The docs checker accepts `check_links.py [ROOT ...]`; no roots defaults to
  QuadMath `docs/`. Explicit roots are documentation directories and missing
  or empty roots fail. It checks same-file and cross-file fragments, including
  duplicate heading slugs, URL-decoded targets and anchors in referenced
  Markdown outside a scan root. Fenced/inline code and remote URLs are not
  content-validated. It is not a mathematical or rendered-layout validator.
  See the multi-root recipe in [development checks](docs/development/README.md).

See [test guide](tests/README.md) and [development checks](docs/development/README.md).

## Figure generation without tracked-output writes

```bash
export QUADMATH_OUTPUT_DIR="$(mktemp -d)/quadmath-results"
MPLBACKEND=Agg uv run --python 3.13 python quadmath/scripts/stats_gallery.py
# An integration owner may generate the full configured gallery set:
MPLBACKEND=Agg uv run --python 3.13 python quadmath/scripts/make_all_figures.py
```

Python helpers resolve the override, create `figures/` and `data/` as requested,
and permit wheel consumers to render without a repository-root lookup.
`make_all_figures.py` runs its explicit list and records recognized emitted
paths relative to that output root in `data/figure_manifest.txt`. A manifest
or successful save is not visual inspection. Examine mathematical units,
actual panel data, clipping and legibility before accepting a figure.

## Integration Phase

The shell wrapper selects Python through its internal `uv run` calls, so supply
`UV_PYTHON=3.13` explicitly:

```bash
# Suite/coverage only
UV_PYTHON=3.13 bash run_all.sh --test-only
# Suite/coverage, figures, glossary freshness and strict manuscript validation
UV_PYTHON=3.13 bash run_all.sh
# Same workflow, then PDF/TeX rendering
UV_PYTHON=3.13 bash run_all.sh --with-pdf
```

The default wrapper generates figures but does not render PDFs. The PDF branch
calls `render_pdf.sh --skip-figures`; the renderer still regenerates the API
glossary, validates manuscript sources and compiles PDFs. Lean and the docs
checker remain separate gates.

### PDF rendering and isolation boundary

Install Pandoc, XeLaTeX, fontconfig and DejaVu Serif/DejaVu Sans Mono first.
The detailed [build guide](quadmath/README.md) describes dependencies and outputs.

```bash
# Run only when checkout-generated changes are intentional.
unset QUADMATH_OUTPUT_DIR
UV_PYTHON=3.13 bash quadmath/scripts/render_pdf.sh
# Reuse existing checkout figures; does not skip glossary/validation/PDF work.
UV_PYTHON=3.13 bash quadmath/scripts/render_pdf.sh --skip-figures
```

`render_pdf.sh` fixes PDF/TeX/temp paths under `quadmath/output/` and writes the
canonical glossary in the source tree. `QUADMATH_OUTPUT_DIR` does not isolate
this pipeline: it can send Python figures elsewhere while Markdown still
references checkout figures. Use a separate checkout for isolated PDF builds.
`SOURCE_DATE_EPOCH` controls the render date where supported; this alone does
not guarantee byte-identical PDFs across toolchains.

### Lean and manuscript projection

Run `lake build` from `lean/` as a separate assigned integration gate. Review
[proof boundaries](docs/lean/README.md) before reporting machine verification.
Port canonical manuscript updates through `docs/manuscript/port_from_source.py`
according to its [guide](docs/manuscript/README.md); do not hand-edit generated
chapter/figure copies. The shared template renderer is another toolchain, not
an implicit phase of `run_all.sh`.

## Output Management

`quadmath/output/` contains generated but partly tracked artifacts, and may
contain user work not reproducible from the current revision. Never treat it
as automatically safe to delete. `clean_output.sh` removes that entire fixed
checkout tree; it does not honor the Python output override. Use it only when
explicitly assigned and after preserving required artifacts. Prefer a fresh
isolated output root for numerical/figure recipes.

## Troubleshooting

| Symptom | Inspect before changing anything |
|---|---|
| Scientific import failure | Actual interpreter and resolved NumPy/Matplotlib/Pillow/SymPy environment; use explicit Python 3.13 |
| File written to unexpected location | `QUADMATH_OUTPUT_DIR`, helper defaults and the PDF isolation boundary above |
| Stale glossary | Run the freshness check; regenerate only when canonical-source writes are intended |
| Missing Markdown figure | Correct generator, selected output root and canonical relative path |
| PDF tool/font failure | Pandoc, XeLaTeX, `fc-list`, DejaVu families and the first actual TeX error |
| Learning result too optimistic | Spatial dependence, selection-versus-test distinction and chronological holdout |

A failed numerical assertion is not repaired by increasing coverage or suppressing
an exception. Fix its source contract and report only the gates actually exercised.
