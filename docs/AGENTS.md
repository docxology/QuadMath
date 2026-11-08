# Documentation Agent Guide: Canonical Sources and Operational Ownership

Read [root guidance](../AGENTS.md) and [the documentation index](README.md).

## Layout and responsibility

- `overview.md`: repository tour, not an alternate mathematical specification.
- `development/`: separate test/coverage, link-check and render gates.
- `learning/`: runtime field/dynamics/evaluation operations with canonical links.
- `analysis/`: timing/statistical methods and figure data provenance.
- `lean/`: declaration/trust scope, not an undated green-build receipt.
- `manuscript/`: scripted template-layout projection of `quadmath/markdown/`.
  Canonical chapters and copied figures are not independently edited here.
- `MANUSCRIPT_STATUS.md`: projection/render evidence maintained by integration.

## Source-informed maintenance

Audit actual signatures/implementations before documenting recipes. Link
mathematical details to canonical chapters. Distinguish package runtime from
checkout scripts, soft fitting from hard constraints, normalized heat from
conservative transport, and validation selection from untouched test estimates.
Do not claim universal Lean counts from finite checks, source elaboration from
unrun builds, or complete semantic coherence from a validator.

Use informative topic-specific headings and relative navigation. Preserve explicit
equation/code anchors and update inbound links when headings change. Avoid saved
current test/file counts and cross-platform determinism promises. Historical
receipts must identify their date/revision and not imply current status.

## Recipes and output safety

Use explicit Python 3.13 commands, retaining `.python-version`. Isolated Python
rendering sets a fresh `QUADMATH_OUTPUT_DIR`; the shell PDF/cleanup scripts do
not share that override. Never purge or overwrite tracked/user-generated outputs.
A projection port or PDF render can write beyond an apparent output directory;
follow its real implementation and [workflow](../WORKFLOW.md).

## Integration handoff

During delegated editing skip builds/tests/lint/formatters. The integration owner
runs source/docs checks, actual rendering/visual inspection and any Lean gate
once edits settle. Report expanded changed paths, direct recipe evidence and
cross-owner heading-link changes. `development/check_links.py` is checker source,
not a prose-maintenance workaround; source changes require its owner's scope.
