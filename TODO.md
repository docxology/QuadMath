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
- [ ] Render pipeline advertises DOI `10.5281/zenodo.16887800` in `quadmath/scripts/render_pdf.sh:33` while README/docs cite `10.5281/zenodo.16887791` — confirm which DOI is canonical for the rendered PDF stamp and unify (source change, needs owner decision). (quadmath/scripts/render_pdf.sh, README.md)
- [ ] Full pytest collection is pathologically slow (>5 min; blocked two audit attempts) on external-drive checkouts — consider `--co -q` guidance or a `.pytest_cache`/`-p no:cacheprovider` note in tests/README if a cause is identified. (tests/README.md)

## Major

- [ ] None open.
