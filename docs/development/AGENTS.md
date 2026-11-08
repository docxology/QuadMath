# Development Documentation Agent Guide: Separate Verification Gates

`README.md` explains integration checks; `check_links.py` implements a Markdown
link checker. Follow [root rules](../../AGENTS.md), [docs ownership](../AGENTS.md)
and [the workflow](../../WORKFLOW.md).

## Documentation responsibilities

Keep commands source-accurate, explicitly selecting scientific Python 3.13 and
preserving `.python-version`. Distinguish configured coverage floors from measured
results, Markdown syntax checks from semantic review, and external-tool fixtures
from actual PDF builds. Do not promise a single wrapper proves complete coherence.

Explain that Python rendering honors `QUADMATH_OUTPUT_DIR`, but the shell PDF
renderer/cleanup script uses fixed checkout paths and can update canonical
sources. Preserve generated user work and immutable release artifacts.

## Checker and integration ownership

The integration owner owns checker source changes and final combined link checks.
Read its actual CLI/scope when documenting invocation; do not assume every Markdown
construct or remote URL is validated. The canonical manuscript validator remains
a separate tool. When a link fails, inspect the intended source/anchor before
changing text or suppressing the checker.

During delegated edits skip builds/tests/lint/formatters. Report actual isolated
consumer probes separately, with expanded paths and unresolved cross-owner links.
