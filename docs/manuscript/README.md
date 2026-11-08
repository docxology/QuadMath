# Generated manuscript projection and published figure assets

Edit [canonical chapters](../../quadmath/markdown/01_introduction.md), then
regenerate this template-layout tree. The [manuscript workflow](../MANUSCRIPT_STATUS.md)
owns setup, isolation and production-font rendering requirements.

## Authoritative content versus maintained template metadata

| Files | Role | Editing rule |
|---|---|---|
| Numbered chapters, `preamble.md` | Projection of `quadmath/markdown/` | Edit source, regenerate glossary if needed, then re-port |
| `98_symbols_glossary.md` | Projection of source chapter `10` | Regenerate through `generate_glossary.py`, not here |
| `99_resources.md` | Projection of source chapter `07` | Edit canonical bibliography prose and re-port |
| `figures/` | Required generated raster assets and graphical-abstract SVG companion | Regenerate through the owning figure scripts, then re-port |
| `config.yaml`, `references.bib`, directory guides | Maintained template metadata and operational guidance | Keep aligned with source and actual pipeline contracts |

## Port only the current source; do not retain repair shims

```bash
uv run --python 3.13 --group dev python docs/manuscript/port_from_source.py --figures-dir /path/to/isolated-run/figures
```

The porter rewrites image paths, root-license paths and the resource/glossary
bucket filenames. Other content, including the already-closed preamble, comes
from the canonical source unchanged. Correct code anchors in the source;
there is no separate repair-only version of chapter 09.

Missing required assets fail. Port verification uses the shared relative-link
and Markdown-fragment checker; it is not a semantic proof or an external URL
crawler. Porting writes tracked projection files, so preserve unrelated local
changes before regeneration.

## Inspect the vector abstract and numerical figure boundaries

The [graphical abstract SVG](figures/graphical_abstract_quadray.svg) preserves
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
