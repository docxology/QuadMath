# TODO.md — QuadMath backlog

Canonical to-do list for this repo (created 2026-08-31 by the agent-ergonomics
pass). One line per entry + file path(s). Completed items get marked, not
deleted.

## Minor

- [x] README.md referenced nonexistent `quadmath/latex/` and `quadmath/resources/` directories — removed. (README.md)
- [x] README.md manuscript section list omitted `00_preamble.md` and the `10_symbols_glossary.md` auto-generated file — fixed. (README.md)
- [x] `quadmath/markdown/AGENTS.md` claimed sections run `00..07` — actual: `00..10`. (quadmath/markdown/AGENTS.md)
- [x] `docs/README.md` map row said sections end at `09_free_energy_active_inference.md` — fixed to `10_symbols_glossary.md`. (docs/README.md)
- [x] tests/README.md stated "Total Tests: 117 / Test Files: 24" with no verification path; disk has 23 test files (118 collected uncertain — full pytest collection exceeds 8 min on the external drive). Replaced with live-derivation commands + dated note. (tests/README.md)

## Medium

- [x] Entry doc had no status/next-actions section (orientation ladder) — added Status & Next Actions pointing at MANUSCRIPT_STATUS.md and TODO.md. (README.md)
- [x] Render pipeline advertised DOI `10.5281/zenodo.16887800` in `quadmath/scripts/render_pdf.sh` while README/docs cite `10.5281/zenodo.16887791` — unified to `10.5281/zenodo.16887791` (cited in 6 doc locations vs 1; canonical per README/AGENTS/docs). (quadmath/scripts/render_pdf.sh, README.md)
- [x] Full pytest collection was pathologically slow (>5 min) on external-drive checkouts — cause identified: corrupted `.venv` (broken numpy/matplotlib installs make each module import fail and retry). Healthy-venv full suite: ~1.5 min. Diagnosis + repair commands documented in tests/README.md. (tests/README.md)

## Major

- [ ] None open.

## Flagged 2026-09-08 review (verified findings awaiting owner decision)

- [x] `src/information.py` `expected_free_energy`: entropy term entered as `+H(q)` and `p` was normalized but unused; preference term added `+log_p_o` as documented. Fixed 2026-09-10: aligned with the canonical expected free energy (Parr/Pezzulo/Friston) — epistemic KL(q‖p), posterior entropy with the variational-bound sign (−H), ambiguity (negative expected log-likelihood), and pragmatic preference entering negatively; pinned-value tests added. (src/information.py, tests/test_information.py)
- [x] `src/nelder_mead_quadray.py`: a degenerate (collinear/zero-volume) initial simplex stayed confined to its line (affine combinations). Fixed 2026-09-10: CVP-style restart implemented — a zero-volume simplex with spread above tolerance is re-seeded at the best vertex plus quadray unit directions scaled by 2 (guaranteed non-zero volume); each restart consumes one iteration. Degenerate line/plane/collapse cases pinned in tests. (src/nelder_mead_quadray.py, tests/test_nelder_mead_visual.py)
- [x] `quadray_from_xyz` off-lattice contract: component-wise rounding is the documented contract (`14_conversions_spec.md`, `quadray.py` docstring, pinned by tests). Closest-vector correction is not implemented.
- [ ] `quadmath/output/**` build artifacts are git-tracked (~25 MB incl. 11 PDFs) while docs call them regeneratable. Deliberate as of 8d36f7c ("artifacts rebuilt"); if untracked later, note that standalone `validate_markdown.py` on a fresh clone requires running `make_all_figures.py` first. (.gitignore, quadmath/output/)

## Review 2026-10-07 (released as 0.2.0)

- [x] `expected_free_energy` subtracted posterior entropy twice (KL already contains −H). Now `G = KL[Q‖P] − E_q[log P(o|s)] − log P(o)`; pinned values updated. Behavior change: values differ from 0.1.0. Manuscript formula corrected in `quadmath/markdown/08` and `09`. (src/quadmath/inference/information.py)
- [x] `mutual_information` and `information_gain` silently treated negative weights as zero; now raise ValueError. (src/quadmath/inference/information.py)
- [x] `dot`, `distance`, `angle` consumed one-shot iterable embeddings twice. (src/quadmath/core/quadray.py)
- [x] `centroid` used banker's rounding and was not shift-invariant; now rounds half up. (src/quadmath/core/quadray.py)
- [x] `qrotate` rejected a quaternion and its negation (same rotation). (src/quadmath/core/quadray.py)
- [x] Nelder–Mead result depended on the representative chosen for initial vertices; vertices now normalized before ordering. (src/quadmath/optimize/nelder_mead_quadray.py)
- [x] `get_repo_root` returned `/` when no marker was found; now raises RuntimeError. (src/quadmath/paths.py)
- [x] Pillow was imported but undeclared; added `pillow>=11.0`. `Image.fromarray(mode=...)` deprecation removed. (pyproject.toml, src/quadmath/viz/animations.py)
- [x] `ivm_dynamics.ball_sites` admitted tetrahedral and octahedral void sites (coordinate sum not divisible by 4). Radius-3 ball is now 13 sites (was 27); radius 1 and 2 are origin-only. Figures `ivm_dynamics_demo.png` and `vis_gallery_dynamics.png` regenerated. (src/quadmath/lattice/ivm_dynamics.py)
- [x] NaN passed the `alpha`, `lam`, and `kernel_width` range checks. (src/quadmath/lattice/ivm_dynamics.py, src/quadmath/lattice/ivm_field.py)
- [x] `ivm_field.learn` silently double-counted duplicate observation sites; now rejected. (src/quadmath/lattice/ivm_field.py)
- [x] `scaling_fit` raised ZeroDivisionError on constant times; r2 is now NaN. Slope docstring corrected. (src/quadmath/stats/statistics.py)
- [x] Benchmark timed cache hits for `sites_through_shell` and `generate_shell`; timing now clears the cache per call. `sites_through_shell` returns a copy. (src/quadmath/stats/benchmarks.py, src/quadmath/lattice/omni_numbering.py)
- [x] Unused imports and locals removed in `src/` and `quadmath/scripts/`; `ruff check --select F401,F841,F811,F821,E9` clean. (src, quadmath/scripts)
- [x] `cayley_menger.py` referenced an undefined `Quadray` annotation. (src/quadmath/core/cayley_menger.py)
- [x] 42 tracked `__pycache__/*.pyc` files untracked. (src/__pycache__)
- [x] `quadray_from_xyz` nearest-point change conflicts with the documented contract (quadmath/markdown/14_conversions_spec.md, `test_spec_examples.py::test_quadray_from_xyz_not_always_xyz_nearest`). Decision: component-wise rounding is the contract. Documented in `14_conversions_spec.md`, the `quadray_from_xyz` docstring, and pinned by `tests/unit/core/test_quadray.py`.
- [x] `docs/manuscript/` and `quadmath/markdown/` have diverged (only `quadmath/markdown/` was updated in this release; `docs/manuscript/` still reports 27 sites and the old `-H` formula). Decision: `quadmath/markdown/` is canonical; `docs/manuscript/` is generated by `docs/manuscript/port_from_source.py` and synced.
- [x] Figure scripts (`visualize.py`, `vis_lattice.py`, `vis_stats.py`, `plots.py`, `animations.py`) close figures in `try/finally`; `plots.py` with `save=False` no longer leaks an open figure. (src/quadmath/viz/)
- [x] `GradientDescentTrainer` checks step size against the augmented Hessian `(2/n)[Z 1]^T[Z 1]` and raises when `lr * lambda_max >= 2`; a non-finite loss raises. (src/quadmath/learn/learning_eval.py)
- [x] `statistics.summarize` returns NaN std for n=1 explicitly, with no RuntimeWarning. (src/quadmath/stats/statistics.py)
- [x] Viz outputs and figure-script CSV/manifest writes go through temp file and `os.replace` (`viz/_common.py`, `tools/atomic_write.py`). Still open: `.npz` and glossary markdown (see below).
- [x] `vis_lattice.field_slice` transposed the plane for square planes and raised TypeError for non-square planes; axes now follow the plane. `vis_gallery_field.png` regenerated. (src/quadmath/viz/vis_lattice.py)
- [x] `visualize._set_axes_equal` set box aspect without equal data scales and swallowed exceptions; now sets aspect from data spans. (src/quadmath/viz/_common.py)
- [x] Public viz functions consumed generator `embedding` inputs twice; now converted once. `visualize.py` gets an explicit `__all__`. (src/quadmath/viz/visualize.py)
- [x] `animations.py` duplicated core `slerp`/`qrotate`; now uses core. Docstring of the quaternion rotation formula corrected. (src/quadmath/viz/animations.py)
- [x] `permutation_test` accepted NaN and returned p=1/51 for `[1,nan,3,4]` vs `[2,5,6]`. (src/quadmath/stats/statistics.py)
- [x] `bootstrap_ci` accepted alpha outside (0, 1). `p_adjust_bonferroni` accepted p outside [0, 1]. `cohens_d` accepted n<2 per sample. (src/quadmath/stats/statistics.py)
- [x] `scaling_fit` accepted a single point or equal sizes. (src/quadmath/stats/statistics.py)
- [x] NaN passed silently through `summarize`, `welch_t_test`, `cohens_d`, ridge, and GD. (src/quadmath/stats/statistics.py, src/quadmath/learn/learning_eval.py)
- [x] `welch_t_test` underflowed for samples near 1e-85 scale; rescaled before squaring. (src/quadmath/stats/statistics.py)
- [x] `run_all.sh` ignored unknown flags and exited 0; unknown options now exit 2. (run_all.sh)
- [x] `render_pdf.sh` combined-PDF failure exited 0; stale PDFs counted as success; xelatex output discarded. Now logged and failure-checked. (quadmath/scripts/render_pdf.sh)
- [x] `clean_output.sh` deleted git-tracked AGENTS.md and README.md under `quadmath/output/`. (quadmath/scripts/clean_output.sh)
- [x] `make_all_figures.py` manifest dropped `.gif`/`.txt`, kept prose prefixes, and duplicated entries; now extracts repo-relative paths and asserts each path exists. (quadmath/scripts/make_all_figures.py)
- [x] `generate_glossary.py` silently rewrote a tracked file; `--check` mode added. `glossary_gen` skipped syntax errors silently and indexed private constants. (quadmath/scripts/generate_glossary.py, src/quadmath/tools/glossary_gen.py)
- [x] `learning_gallery.py` saved the figure via `plt.gcf()` after `save=False`; now passes `out_path` to `plot_loss_history`. (quadmath/scripts/learning_gallery.py, src/quadmath/viz/plots.py)
- [x] Doc drift corrected: radius-3 lattice is 13 sites and free-energy formula in `docs/manuscript/08, 09, 12, 16`; test and file counts in `tests/README.md`, `tests/AGENTS.md`; flat `src/*.py` paths in SPEC, ARCHITECTURE, README, AGENTS, docs; broken links in `quadmath/markdown/`; scripts README counts and paths.
- [x] `validate.check_slerp_midpoint` (validate.py ~L270-286) fails correct slerp for any pair with dot < 0, so q and -q are reported as failures. Design decision: compares against the shorter-arc representative (`validate/validate.py`).
- [x] Stat-callable convention: `bootstrap_ci` calls `stat(x, axis=1)`; `jackknife_ci` calls `stat(1-D)`. Decision: the stat callable is 1-D in, scalar out (`stats/statistics.py`); `bootstrap_ci` loops per resample.
- [x] `bootstrap_ci` and `scaling_fit` reject NaN; `permutation_test` compares with tolerance 1e-12 * max|pool| for float ties.
- [x] `quadmath/output/data/figure_manifest.txt` regenerated with repo-relative paths (no `/Volumes/...`). (34 lines, `/Volumes/...` paths) and fails the new existence assertion. Regenerate with the full pipeline.
- [x] `quadmath/markdown/10_symbols_glossary.md` not regenerated; regenerated; `generate_glossary.py --check` passes.
- [x] `render_pdf.sh` MODULES covers sections 01–18 (`check_module_coverage`); pandoc and xelatex helpers are shared (`pandoc_to_tex`, `compile_tex_to_pdf`). Decision: `gpu_acceleration_demo.py` stays a manual timing benchmark, documented as writing no artifact and outside `FIGURE_SCRIPTS`.
- [x] `render_pdf.sh` dependency check verifies the DejaVu Serif and DejaVu Sans Mono families that the preamble requires (install: `brew install --cask font-dejavu`).
- [x] Viz path policy: `resolve_output_path` for all figure builders. MP4 output is bit-exact (`mp4_writer` uses `+bitexact` flags; `tests/unit/viz/test_mp4_reproducibility.py`). `vis_lattice.shell_scatter` and `dynamics_strip` materialize their inputs once.
- [x] `volumes_demo` SVG is byte-reproducible (fixed `svg.hashsalt`, no `dc:date`).
- [x] `src/AGENTS.md` and `src/README.md` no longer name `_set_axes_equal` in `visualize.py`; the helper lives in `viz/_common.py`.
- [x] Lint: `uvx ruff check --isolated --select F,E9 src tests quadmath/scripts` is clean (star re-exports carry `# noqa: F403`; module re-exports in `lattice/__init__.py` use redundant aliases).
- [x] Previously reported: 8 `F`/`E9` errors in `tests/` (including F841 at `tests/unit/viz/test_vis_stats.py:135`), and 23 in `src/` and `quadmath/scripts/` (including F403/F401 in `src/quadmath/viz/__init__.py`). Resolved.
- [x] Test counts: `tests/README.md` and `tests/AGENTS.md` use a live count command, no hard totals. Previously: a 2026-10-07 snapshot (40 files, 709 tests); the tree now has 41 files and 745 tests. 
- [x] `ivm_field.ball_sites` renamed to `shell_ball_sites` (cumulative shell union). `ivm_dynamics.ball_sites` keeps its name (void-filtered ball).
- [x] `.npz` and glossary markdown writes are atomic: all `.npz` writes go through `tools/atomic_write.atomic_savez` (`discrete_path.npz` was already atomic via `atomic_target` and now uses `atomic_savez`; `quadray_clouds_data.npz` and `figure_13_data.npz` in `quadmath/scripts/`), and the glossary uses `atomic_write_text` (`quadmath/scripts/generate_glossary.py`).
- [x] `figure_13_data.npz` is byte-reproducible: `atomic_savez` writes fixed ZIP timestamps (1980-01-01), `ZIP_STORED`, and `allow_pickle=False`. Two consecutive `make_all_figures.py` runs produced identical SHA-256 for every `.npz`, PNG, and SVG under `quadmath/output/`.
- [x] Neighbor-move sets have one definition: `core/quadray.IVM_NEIGHBOR_MOVES`. `omni_numbering.NEIGHBOR_MOVES`, `ivm_field.IVM_NEIGHBOR_STEPS`, `ivm_dynamics.neighbor_shifts`, and `core/examples._ivm_neighbor_permutations` all derive from it.
- [x] Glossary empty signature cells rendered as a literal double backtick, which pandoc paired across cells and broke the PDF build (`glossary_gen.generate_markdown_table`). Empty cells are now blank; `render_pdf.sh` exits 0 with 18 modules and the combined PDF built.
