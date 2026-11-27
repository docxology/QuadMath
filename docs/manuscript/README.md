# Generated manuscript projection and published figure assets

Edit [canonical chapters](../chapters/01_introduction.md), then
regenerate this template-layout tree. The [manuscript workflow](../MANUSCRIPT_STATUS.md)
owns setup, isolation and production-font rendering requirements.

## Authoritative content versus maintained template metadata

| Files | Role | Editing rule |
|---|---|---|
| Numbered chapters, `preamble.md` | Projection of `docs/chapters/` | Edit source, regenerate glossary if needed, then re-port |
| `98_symbols_glossary.md` | Projection of source chapter `10` | Regenerate through `generate_glossary.py`, not here |
| `99_resources.md` | Projection of source chapter `07` | Edit canonical bibliography prose and re-port |
| `docs/figures/` | Required generated raster assets and graphical-abstract SVG companion | Regenerate through the owning figure scripts, then re-port |
| Configuration, bibliography, [claim ledger](claim-ledger.yaml), and directory guides | Maintained template metadata, evidence and guidance | Keep numeric claims tied to their declared formulas, recorded inputs or exact receipts |

## Port only the current source; do not retain repair shims

```bash
uv run --python 3.13 --group dev python docs/manuscript/port_from_source.py --figures-dir /path/to/isolated-run/figures
```

The porter rebases images to the shared figure directory and uses public
repository URLs for chapter and source links in the projected edition.
The projection uses dollar-delimited inline math for the shared renderer.
Canonical chapters retain checkout-relative links. The porter also maps
the resource and glossary filenames. Other content, including the already-closed preamble, comes
from the canonical source unchanged. Correct code anchors in the source;
there is no separate repair-only version of chapter 09.

Missing required assets fail. Port verification uses the shared relative-link
and Markdown-fragment checker; it is not a semantic proof or an external URL
crawler. Porting writes tracked projection files, so preserve unrelated local
changes before regeneration.

## Inspect the vector abstract and numerical figure boundaries

The [graphical abstract SVG](../figures/graphical_abstract_quadray.svg) preserves
the actual scale-one basis rays and full twelve-around-one FCC contact cluster.
Its PNG counterpart is embedded in the manuscript for the PDF toolchain.
The regression, categorical and simplex figures label their actual models,
units, feasible domains and recorded rather than presumed convergence.

## Distinguish template compatibility from a successful PDF build

This layout supplies numbered chapters, fenced preamble, bibliography and
configuration for the external template renderer. The local production
pipeline is [`render_pdf.sh`](../../quadmath/scripts/render_pdf.sh); its real
font/TeX/Pandoc/video build has a separate Linux CI gate. Old deposited PDFs
and release assets remain evidence for their archived versions, not for an
unrendered changed checkout.
