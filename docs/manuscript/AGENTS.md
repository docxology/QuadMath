# Manuscript projection: source ownership, assets and verification

## Preserve canonical content and local work

- `quadmath/markdown/` is authoritative. Never hand-edit projected numbered
  chapters, glossary, preamble or figure bytes; change their producer and port.
- `config.yaml`, `references.bib` and directory guides are maintained here.
  Keep bibliography keys case-insensitively unique.
- The canonical preamble is already closed and is copied unchanged. Fix source
  defects at source rather than adding repair shims or divergent chapter text.
- Preserve unrelated local projection/figure changes before porting.

## Regenerate with an explicit figure source and one link gate

- Run `uv run --python 3.13 --group dev python docs/manuscript/port_from_source.py
  --figures-dir /path/to/run/figures` from the repository root.
- The required asset inventory includes the graphical abstract's PNG and SVG.
  Missing assets fail; the porter invokes the shared documentation checker
  for relative links and Markdown fragments.
- Source content invariants are checked separately by
  `quadmath/scripts/validate_markdown.py --strict`.
- Read [README](README.md) and the [manuscript workflow](../MANUSCRIPT_STATUS.md)
  for regeneration, publication paths and render limitations.

## Keep verification claims specific

- Template-compatible layout is not evidence that the external template
  renderer ran. Local production rendering is `quadmath/scripts/render_pdf.sh`.
- Linux CI provisions its fonts/media/TeX tools. Do not re-label source checks
  or alternate-font diagnostics as a successful production PDF build.
- Keep status outside this directory so renderer discovery does not ingest it
  as an additional manuscript chapter.
