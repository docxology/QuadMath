# Canonical QuadMath Manuscript Sources

Edit these Markdown sources, not generated copies under `docs/manuscript/` or `quadmath/output/`. The build scripts assemble individual chapters and the combined manuscript. The mathematical authority for quotient coordinates, conversion, and lattice contracts is [SPEC.md](../../SPEC.md).

## Reading paths through the mathematical modules {#manuscript-sections}

| Source | Topic and boundary |
| --- | --- |
| [01_introduction.md](01_introduction.md) | Rank-three physical coordinates, manuscript scope, native-basis/FCC graphical abstract |
| [02_4d_namespaces.md](02_4d_namespaces.md) | Euclidean four-space, Minkowski spacetime, and tetrahedral three-space are distinct |
| [03_quadray_methods.md](03_quadray_methods.md) | Exact metric/affine primitives, rational volumes, source-linked code anchors |
| [04_optimization_in_4d.md](04_optimization_in_4d.md) | Discrete search, finite local stopping, and metric increments rather than geodesic integration |
| [05_extensions.md](05_extensions.md) | Explicitly unimplemented research directions and cultural context |
| [06_discussion.md](06_discussion.md) | Evidence boundaries, units, statistical interpretation, and limitations |
| [07_resources.md](07_resources.md) | External bibliography, not an executed validation manifest |
| [08_equations_appendix.md](08_equations_appendix.md) | Labeled volume, probability, and metric equations with scale conventions |
| [09_free_energy_active_inference.md](09_free_energy_active_inference.md) | Toy probability objectives, state increments, and didactic partition diagrams |
| [10_symbols_glossary.md](10_symbols_glossary.md) | Authored notation plus a generated API-index block |
| [11_ivm_field_learning.md](11_ivm_field_learning.md) | Soft observations, hard-boundary harmonic extension, and general linear geometry fitting |
| [12_ivm_dynamics.md](12_ivm_dynamics.md) | Symmetric-normalized heat versus conservative transport versus rounded averaging |
| [13_lattice_tooling.md](13_lattice_tooling.md) | FCC graph-shell enumeration and bounded float64 spatial queries |
| [14_conversions_spec.md](14_conversions_spec.md) | Embedding/gauge, exact inverse membership, and distinct floating snapping |
| [15_learning_evaluation.md](15_learning_evaluation.md) | Random-site interpolation holdouts, temporal forecasting, and tabular fits |
| [16_lattice_gallery.md](16_lattice_gallery.md) | Composable lattice panels, owned figures, and distinct animation diffusion |
| [17_benchmarks_statistics.md](17_benchmarks_statistics.md) | Variable wall-clock timing and assumption-bound statistical inference |
| [18_stats_gallery.md](18_stats_gallery.md) | Computed intervals, stable log fits and empirical mass; labelled synthetic data, not measured benchmarks |

For coordinate geometry, read 02 → 03 → 14 and consult SPEC. For lattice models, read 11 → 12 → 13 → 15, then the gallery in 16. For measurement, read 17 before interpreting 18.

## Equations, listings, and source references {#writing-guidelines}

- Preserve explicit equation and code identifiers when changing titles; update heading backlinks when an automatic identifier changes.
- Use LaTeX for mathematical prose. Do not put mathematical Unicode or prose equations inside inline code/listings; reserve code formatting for actual identifiers, paths, and executable expressions.
- State the embedding scale. Default scale-one XYZ uses native volume factor 3/8; S3 applies only to sphere-radius-one lengths.
- Use relative image paths such as `../output/figures/graphical_abstract_quadray.png`, with captions tied to the script's actual model, axes, units, and display normalization.
- Reference implementations under `src/quadmath/` and distinguish implemented APIs from proposals.
- A saved seeded example is not a current test/coverage receipt. Record actual execution evidence separately.

## Preamble and generated API block {#special-files}

`00_preamble.md` contains LaTeX commands and listing styles, not YAML author/title front matter. Its code fence is consumed by the manuscript tooling.

Only the block between `BEGIN: AUTO-API-GLOSSARY` and `END: AUTO-API-GLOSSARY` in `10_symbols_glossary.md` is generated. The preceding symbol tables are authored and should be edited normally. Regenerate the API block with:

```bash
uv run --python 3.13 python quadmath/scripts/generate_glossary.py
```

## Validation and build entry points {#validation}

Run from the repository root in the supported Python 3.13 environment. These are instructions, not a claim that the commands were run for this edit:

```bash
uv run --python 3.13 python quadmath/scripts/validate_markdown.py --strict
bash quadmath/scripts/render_pdf.sh
```

The builder owns combined/individual PDF and LaTeX output. A direct single-file Pandoc invocation does not reproduce its preamble, staging, and figure handling. See [the build-system overview](../README.md), [script documentation](../scripts/README.md), and [operational workflow](../../WORKFLOW.md). When retaining existing generated evidence, select an isolated `QUADMATH_OUTPUT_DIR`; do not overwrite user-generated outputs just to inspect an example.
