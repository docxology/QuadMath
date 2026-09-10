# development/ — Agent Notes

Documentation of the test/coverage workflow, plus the docs-tree link checker.

## Contents

- `README.md` — the test/coverage contract (mirrors root `AGENTS.md`).
- `check_links.py` — resolves every relative markdown link/image under
  `docs/` against the filesystem; run after any `docs/` edit:
  `uv run python docs/development/check_links.py` (exit 0 = all resolve).

## Rules

- Keep `README.md` in lock-step with the root `AGENTS.md` quality gates; this
  folder restates for navigation, it does not override.
- `check_links.py` skips fenced code blocks and non-`http(s)`/relative
  targets; treat any new failure as a broken doc link, not a checker bug,
  unless you can reproduce the miss on a resolvable file.
- The manuscript markdown validator (`quadmath/scripts/validate_markdown.py`)
  owns `quadmath/markdown/`; this checker owns `docs/`. Do not merge their
  responsibilities.
