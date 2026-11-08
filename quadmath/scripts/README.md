# Manuscript figures, data manifests and PDF orchestration

Checkout entrypoints compose seeded examples and call package numerical and
rendering primitives under `src/`. Reusable algorithms and general-purpose
renderers belong in the package. Established manuscript demos retain some
model setup and plot composition here; they are not all single-call wrappers.

## Script Inventory

| Script | Purpose | Delegates to | Run command |
|--------|---------|--------------|-------------|
| `render_pdf.sh` | Full build: figures, glossary, validation, per-section (18) + combined PDFs at 1cm margins, LaTeX export | `make_all_figures.py`, `generate_glossary.py`, `validate_markdown.py` (its `scripts` array; figures skipped with `--skip-figures`) + `pandoc`/`xelatex` | `bash quadmath/scripts/render_pdf.sh` |
| `clean_output.sh` | Remove generated files under `quadmath/output/`, keeping the per-folder `AGENTS.md` and `README.md` | filesystem only | `bash quadmath/scripts/clean_output.sh` |
| `make_all_figures.py` | Run the 19 generators; atomically write deduplicated paths relative to the configured output root, including isolated `QUADMATH_OUTPUT_DIR` builds | the 19 scripts listed under Pipeline composition (subprocess); `quadmath.tools.atomic_write`; `src/quadmath/paths.py` | `uv run python quadmath/scripts/make_all_figures.py` |
| `validate_markdown.py` | Check image refs, internal links/anchors, equation-label uniqueness | none (stdlib only) | `uv run python quadmath/scripts/validate_markdown.py` |
| `generate_glossary.py` | Regenerate `quadmath/markdown/10_symbols_glossary.md` from the `src/` API; `--check` exits 1 when the file is stale and never writes | `src/quadmath/tools/glossary_gen.py` | `uv run python quadmath/scripts/generate_glossary.py [--check]` |
| `information_demo.py` | Actual regression predictions and gradient-Gram spectra; separate constant-metric descent, categorical free energy and discrete path | `src/quadmath/inference/information.py`, `src/quadmath/core/metrics.py`, `src/quadmath/optimize/discrete_variational.py`, `src/quadmath/viz/visualize.py`, `src/quadmath/viz/_common.py` | `uv run --python 3.13 python quadmath/scripts/information_demo.py` |
| `active_inference_figures.py` | State-aligned four-coefficient regression diagnostics and a valid categorical-simplex face; not a biological agent | `src/quadmath/inference/information.py`, `src/quadmath/viz/_common.py`, `src/quadmath/paths.py` | `uv run --python 3.13 python quadmath/scripts/active_inference_figures.py` |
| `simplex_animation.py` | Integer Nelder–Mead history, bounded snapshots and the actual final recorded best; slot traces are not persistent vertex identities | `src/quadmath/optimize/nelder_mead_quadray.py`, `src/quadmath/viz/visualize.py`, `src/quadmath/viz/_common.py`, `src/quadmath/core/quadray.py` | `uv run --python 3.13 python quadmath/scripts/simplex_animation.py` |
| `discrete_variational_demo.py` | Discrete IVM-lattice descent demo (figure + data) | `src/quadmath/optimize/discrete_variational.py`, `src/quadmath/viz/visualize.py`, `src/quadmath/core/quadray.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/discrete_variational_demo.py` |
| `volumes_demo.py` | Tetrahedron volume scaling: integer vs Cayley-Menger (figure + CSV) | `src/quadmath/core/cayley_menger.py`, `src/quadmath/core/quadray.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/volumes_demo.py` |
| `ivm_neighbors.py` | 12-neighbor IVM lattice structure (figure + NPZ data) | `src/quadmath/viz/visualize.py`, `src/quadmath/core/quadray.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/ivm_neighbors.py` |
| `quadray_clouds.py` | 3D scatter of deterministic quadray point clouds | `src/quadmath/lattice/conversions.py`, `src/quadmath/core/quadray.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/quadray_clouds.py` |
| `polyhedra_quadray_constructions.py` | Closed polyhedra on shared XYZ axes, with native volumes measured from their faces | `src/quadmath/viz/vis_polyhedra.py` | `uv run python quadmath/scripts/polyhedra_quadray_constructions.py` |
| `graphical_abstract_quadray.py` | PNG and SVG of the rank-three Quadray quotient, native basis rays and complete twelve-around-one FCC contact cluster | `src/quadmath/viz/abstracts.py` (`render_quadray_abstract`) | `uv run --python 3.13 python quadmath/scripts/graphical_abstract_quadray.py` |
| `sympy_formalisms.py` | Symbolic Cayley-Menger / IVM volume formalisms (SymPy figure + CSV) | `src/quadmath/core/symbolic.py`, `src/quadmath/core/quadray.py` | `uv run python quadmath/scripts/sympy_formalisms.py` |
| `gpu_acceleration_demo.py` | Integer tetra-volume timing benchmark (manual; prints timings, writes no artifact, not part of the pipeline) | `src/quadmath/core/quadray.py` | `uv run python quadmath/scripts/gpu_acceleration_demo.py` |
| `ivm_field_demo.py` | Static IVM field learning demo: three-panel lattice / noisy observations / learned-field figure | `src/quadmath/lattice/ivm_field.py`, `src/quadmath/core/quadray.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/ivm_field_demo.py` |
| `ivm_dynamics_demo.py` | IVM lattice dynamics demo figure (multi-snapshot) | `src/quadmath/lattice/ivm_dynamics.py` | `uv run python quadmath/scripts/ivm_dynamics_demo.py` |
| `lattice_gallery.py` | Three-figure lattice visualization gallery | `src/quadmath/viz/vis_lattice.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/lattice_gallery.py` |
| `stats_gallery.py` | Four-figure statistics gallery (latency, scaling, CI, ECDF) | `src/quadmath/viz/vis_stats.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/stats_gallery.py` |
| `animation_gallery.py` | Three deterministic IVM lattice animation GIFs | `src/quadmath/viz/animations.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/animation_gallery.py` |
| `learning_gallery.py` | Learning-evaluation loss-history figure | `src/quadmath/learn/learning_eval.py`, `src/quadmath/viz/plots.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/learning_gallery.py` |
| `quaternion_gallery.py` | Quaternion slerp geodesic path figure | `src/quadmath/viz/plots.py` | `uv run python quadmath/scripts/quaternion_gallery.py` |
| `animation_stills.py` | Single-row stills strip sampled from the three animation sequences | `src/quadmath/viz/animations.py`, `src/quadmath/core/quadray.py` | `uv run python quadmath/scripts/animation_stills.py` |
| `stats_diagnostics_gallery.py` | Bootstrap-vs-jackknife confidence-interval comparison figure | `src/quadmath/stats/statistics.py`, `src/quadmath/viz/vis_stats.py`, `src/quadmath/paths.py` | `uv run python quadmath/scripts/stats_diagnostics_gallery.py` |

All commands run from the repository root.

### Pipeline composition

- `render_pdf.sh` runs 3 scripts, in order: `make_all_figures.py` (skipped with
  `--skip-figures`), `generate_glossary.py`, `validate_markdown.py --strict`.
  It then builds one PDF per section and a combined PDF with `pandoc` and
  `xelatex`. All PDFs share one margin (1cm on every side).
- `render_pdf.sh` section list: `MODULES` holds the 18 sections `01`-`18`
  (`NN_*.md|title`). `00_preamble.md` is listed in `EXCLUDED_MODULES` with a
  reason; it is the LaTeX header, not a section. `check_module_coverage` fails
  the build if any `NN_*.md` file is unlisted, a listed file is missing, or an
  exclusion has no reason. Titles must avoid LaTeX special characters because
  pandoc copies them into `\title{}` unescaped.
- `make_all_figures.py` runs 19 scripts, in its `FIGURE_SCRIPTS` order:
  `information_demo.py`, `active_inference_figures.py`, `volumes_demo.py`,
  `ivm_neighbors.py`, `quadray_clouds.py`, `simplex_animation.py`,
  `graphical_abstract_quadray.py`, `polyhedra_quadray_constructions.py`,
  `discrete_variational_demo.py`, `sympy_formalisms.py`,
  `ivm_field_demo.py`, `ivm_dynamics_demo.py`, `lattice_gallery.py`,
  `stats_gallery.py`, `animation_gallery.py`, `learning_gallery.py`,
  `quaternion_gallery.py`, `animation_stills.py`,
  `stats_diagnostics_gallery.py`. Every entry must print at least one path
  that exists afterwards; there are no exemptions.
- `gpu_acceleration_demo.py` is not in `FIGURE_SCRIPTS`. It is a manual timing
  benchmark: its output varies by machine and it writes no artifact, so it
  cannot satisfy the manifest contract.
- Atomic writes: `quadmath.tools.atomic_write` (`atomic_open`,
  `atomic_write_text`) writes to a temp file in the target directory and
  `os.replace`s it on success, so a failed run leaves the previous file intact
  and no temp file behind. Used by `make_all_figures.py` (manifest),
  `volumes_demo.py`, `ivm_neighbors.py`, `sympy_formalisms.py` (CSV and
  symbolics text), and `information_demo.py` (CSV and `np.savetxt` output).
  `.npz` outputs use `atomic_savez`, which also pins ZIP timestamps so a
  regenerated archive is byte-identical. The glossary markdown uses
  `atomic_write_text`.

## Script Development

### Importable graphical-abstract entrypoint

From a current checkout environment, the renderer is directly usable:

```python
from quadmath.viz.abstracts import render_quadray_abstract

png_path, svg_path = render_quadray_abstract()
print(png_path)
print(svg_path)
```

The default PNG filename is unchanged. The sibling SVG preserves vector
geometry and belongs in the figure manifest. An explicit destination must be a
PNG path; the sibling uses its stem. Each save is atomic, but the pair is not
a transaction: a successfully saved PNG remains if the SVG save fails.
Renderer-owned figures close on success and failure without closing caller
figures. This entrypoint is a main/Unreleased addition, not an API of the
immutable 0.6.0 wheel.

### Requirements

1. **Reusable behavior in `src/`**: keep shared numerical kernels and renderers
   importable and tested; established demos may compose fixed models and plots.
2. **Path bootstrap**: `_ensure_src_on_path()` inserts `<repo>/src` so manuscript
   builds exercise that checkout rather than an unrelated installed version.
   The same `quadmath` modules are also shipped in the release wheel.
3. **Headless mode**: `MPLBACKEND=Agg` before importing matplotlib
4. **Local seeded generators**: use `np.random.default_rng(seed)` without
   resetting another consumer's global random state.
5. **Print outputs**: absolute paths ending in
   `.png/.svg/.mp4/.pdf/.csv/.npz/.gif/.txt` feed the manifest.
6. **Failure-safe figures and main guards**: use `figure_scope`/`save_figure`
   from `quadmath.viz._common`; keep entry logic behind `if __name__ == "__main__"`.
   (`tests/tools/test_sympy_formalisms.py` imports `sympy_formalisms.py` directly)

## Output Locations

Python output helpers honor `QUADMATH_OUTPUT_DIR`; the comments show defaults.
The PDF shell script still publishes PDF/TeX under the checkout's default output
tree. Use a separate checkout for a fully isolated PDF build.

```python
from quadmath.paths import get_figure_dir, get_data_dir, get_output_dir  # after bootstrap

get_figure_dir()  # -> quadmath/output/figures/
get_data_dir()    # -> quadmath/output/data/
get_output_dir()  # -> quadmath/output/
```

## Adding New Scripts

1. Follow the import/bootstrap, numerical-delegation and figure-lifetime rules above
2. If it is a figure generator, add it to the `FIGURE_SCRIPTS` tuple in
   `make_all_figures.py` (`render_pdf.sh` reaches it through that script)
3. Test individually: `uv run python quadmath/scripts/new_script.py`
4. Run the full pipeline: `bash quadmath/scripts/render_pdf.sh`

## Cross-References

- [Repository overview and quick start](../../README.md) — repo-root `README.md`
- [Repository-wide agent guidelines](../../AGENTS.md) — repo-root `AGENTS.md`
- [Build-system overview](../README.md) — `quadmath/README.md`
- [Build-system directory agent notes](../AGENTS.md) — `quadmath/AGENTS.md`
- [Path utilities](../../src/quadmath/paths.py) — `src/quadmath/paths.py`
