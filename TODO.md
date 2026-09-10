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

- [ ] `src/information.py` `expected_free_energy`: entropy term enters as `+H(q)` and `p` is normalized but unused; preference term adds `+log_p_o` as documented. This is self-consistent with its docstring but does not match canonical expected free energy (Parr/Pezzulo/Friston: epistemic KL + ambiguity − pragmatic), where entropy is penalized and preferences lower G. Changing semantics affects the appendix narrative — needs an owner decision, then pinned-value tests. (src/information.py, tests/test_information.py)
- [ ] `src/nelder_mead_quadray.py`: a degenerate (collinear/zero-volume) initial simplex stays confined to its line (affine combinations); termination relies on lattice collapse, and `int()` truncation plus floor centroid bias directions. Documented in the docstring; a CVP-style restart/perturbation would be the robust fix. (src/nelder_mead_quadray.py)
- [ ] `src/quadray.py` `quadray_from_xyz`: exact for lattice points (fixed 2026-09-08), but for off-lattice inputs component-wise rounding is not the true closest lattice point in XYZ distance; a proper closest-vector correction would fulfill the original "nearest" wording. (src/quadray.py)
- [ ] `quadmath/output/**` build artifacts are git-tracked (~25 MB incl. 11 PDFs) while docs call them regeneratable. Deliberate as of 8d36f7c ("artifacts rebuilt"); if untracked later, note that standalone `validate_markdown.py` on a fresh clone requires running `make_all_figures.py` first. (.gitignore, quadmath/output/)

