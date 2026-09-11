# analysis/ — Agent Notes

Documentation for the landed benchmarks/statistics surface:
`src/benchmarks.py` (perf_counter timing harness), `src/statistics.py`
(bootstrap, permutation tests, effect sizes, scaling fits), and
`src/vis_stats.py` (input-agnostic gallery rendering primitives).
Manuscript treatments: `quadmath/markdown/17_benchmarks_statistics.md` and
`quadmath/markdown/18_stats_gallery.md`.

## Rules

- Keep this folder's API map in lock-step with the actual module docstrings;
  the modules are owned by their implementing agent — propose changes there,
  don't patch `src/` from here.
- Every markdown link must resolve (checker:
  [development/check_links.py](../development/check_links.py) validates
  fragments too); reference section files and sources with inline code
  paths rather than links when the target may not exist yet.
- When the manuscript section content changes in `quadmath/markdown/`, the
  ported copy under `docs/manuscript/` is refreshed by
  `docs/manuscript/port_from_source.py` — do not hand-edit the port.
- Do not duplicate the repository conventions here; link to
  [development](../development/README.md) for the test/coverage workflow.