# manuscript/ — Agent Notes

Template-layout projection of `../../quadmath/markdown/` for the shared
docxology/template render pipeline.

## Invariants

- `quadmath/markdown/` is **authoritative for content edits**; this tree is a
  scripted port (copy + the path rewrites documented in `README.md`). Never
  hand-edit ported section content here — change the source, then re-port.
- Re-port source edits with `uv run python docs/manuscript/port_from_source.py`
  (copies sections/figures, applies the rewrites, verifies links).
- `preamble.md` must keep its fenced `latex` block **closed** — upstream
  `00_preamble.md` is missing the closing fence; `extract_preamble` needs a
  complete block and silently drops an unclosed one. `port_from_source.py`
  appends the fence on every re-port; do not remove it.
- The three code-anchor links in `09_free_energy_active_inference.md` are
  deliberately repointed to `03_quadray_methods.md#code:*` (upstream points
  them at `08_equations_appendix.md`, which has no `{#code:*}` anchors).
  `port_from_source.py` re-applies this on every port; do not "fix" them back.
- `references.bib` keys must stay unique case-insensitively (template
  `_validate_unique_citation_keys` fails closed on duplicates).
- Every image/link (including `{#id}` fragments) must resolve on disk: run
  `uv run python docs/development/check_links.py` after edits here.
- Do not regenerate figures into this tree by hand; figures are copies of
  `quadmath/output/figures/` (which `clean_output.sh` wipes). Regenerate via
  the quadmath scripts, re-port.

## Render parity

- Target pipeline: template `stage_03_render.py` → `infrastructure.rendering`
  (discovers top-level `NN_*.md`, extracts `preamble.md`, consumes `config.yaml`
  and `*.bib`).
- Local pipeline: `quadmath/scripts/render_pdf.sh` (unchanged).
- Resolved caveat: `discover_manuscript_files` sweeps non-excluded top-level
  `.md` files into the "other" bucket, and `MANUSCRIPT_STATUS.md` was not in
  the template `EXCLUDE_NAMES` set — the status file now lives at
  `docs/MANUSCRIPT_STATUS.md`, outside the render tree.
- Validate markdown invariants of the source tree with
  `uv run python quadmath/scripts/validate_markdown.py --strict` (it checks
  `quadmath/markdown/`, not this projection).
