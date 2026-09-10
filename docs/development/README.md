# development/ — test and coverage workflow

The repository enforces a strict test-driven contract on `src/` (see root
`AGENTS.md` and `../overview.md`).

## The workflow

```bash
# from the repository root
uv sync

# run the full suite with coverage (plugin autoload disabled on purpose)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q

# check the coverage floor
uv run coverage report
```

## Rules (enforced by review and by the suite)

- **100% test coverage for every module under `src/`.** The test file for
  `src/<module>.py` is `tests/test_<module>.py`.
- **No mocks.** Tests use real numerical examples — actual lattice points,
  actual volumes, actual embeddings — never stubs.
- **Fixed RNG seeds** wherever randomness appears, so every run is
  reproducible.
- **Fast and hermetic**: no network, no external state, deterministic outputs.
- **Dependencies**: `numpy`, `matplotlib`, `sympy` only (`pyproject.toml`).
- **Type hints and NumPy-style docstrings** on all public functions.

## Source ↔ test correspondence

| Source module | Test file |
|---|---|
| `src/quadray.py` | `tests/test_quadray.py` |
| `src/linalg_utils.py` | `tests/test_linalg_utils.py` |
| `src/cayley_menger.py` | `tests/test_cayley_menger.py` |
| `src/conversions.py` | `tests/test_conversions.py` |
| `src/nelder_mead_quadray.py` | `tests/test_nelder_mead_quadray.py` |
| `src/discrete_variational.py` | `tests/test_discrete_variational.py` |
| `src/information.py` | `tests/test_information.py` |
| `src/ivm_dynamics.py` | `tests/test_ivm_dynamics.py` |
| `src/<module>.py` | `tests/test_<module>.py` (pattern) |

## Manuscript validation

```bash
uv run python quadmath/scripts/validate_markdown.py --strict
```

Checks the manuscript source tree (`quadmath/markdown/`): image references
resolve under `quadmath/output/`, equation labels are unique, internal anchors
exist, and no bare URLs. The template-layout projection under
`docs/manuscript/` has its own checker:

```bash
uv run python docs/development/check_links.py
```

## Docs link checker

`check_links.py` resolves every relative markdown link/image under `docs/`
against the filesystem and validates `#fragment` targets against explicit
`{#id}` anchors / heading slugs. It matches targets by `](target)` position
so captions whose alt text contains brackets are still checked, and it
excludes fenced code blocks and inline code spans. Run after any `docs/` edit
(exit 0 = all resolve).

## Full build

`bash run_all.sh` chains the suite, figure generation, and PDF rendering;
`bash quadmath/scripts/clean_output.sh` purges the generated
`quadmath/output/` tree (safe to re-run; outputs are regeneratable).
