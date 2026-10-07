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
- [ ] `src/quadray.py` `quadray_from_xyz`: exact for lattice points (fixed 2026-09-08), but for off-lattice inputs component-wise rounding is not the true closest lattice point in XYZ distance; a proper closest-vector correction would fulfill the original "nearest" wording. (src/quadray.py)
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
- [ ] `quadray_from_xyz` nearest-point change conflicts with the documented contract (quadmath/markdown/14_conversions_spec.md, `test_spec_examples.py::test_quadray_from_xyz_not_always_xyz_nearest`). Needs an owner decision: keep component rounding and document it, or change the contract and test.
- [ ] `docs/manuscript/` and `quadmath/markdown/` have diverged (only `quadmath/markdown/` was updated in this release; `docs/manuscript/` still reports 27 sites and the old `-H` formula). Pick one canonical copy and delete or sync the other.
- [ ] Figure scripts (`animations.py`, `visualize.py`, `vis_lattice.py`) close figures with `plt.close` outside `try/finally`; an exception during save leaks the figure.
- [ ] `learning_eval` gradient descent has no step-size stability check for `lr` (learning_eval.py, `lr` loop).
- [ ] `statistics.summarize` returns NaN std for n=1 with a numpy RuntimeWarning rather than an explicit value.
- [ ] Figure and data writes are not atomic.
- [ ] `ivm_field.ball_sites` and `ivm_dynamics.ball_sites` share a name with different semantics (see `src/quadmath/lattice/__init__.py`); consider renaming one.
