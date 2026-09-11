# learning/ — Agent Notes

Documentation for `src/ivm_field.py` and `src/ivm_dynamics.py` (landed; see
`quadmath/markdown/12_ivm_dynamics.md` for the manuscript treatment).

## Rules

- Keep this folder's API map in lock-step with the actual module docstrings;
  the modules are owned by their implementing agent — propose changes there,
  don't patch `src/` from here.
- Every markdown link must resolve (checker:
  [development/check_links.py](../development/check_links.py) validates
  fragments too).
- When the manuscript section content changes in `quadmath/markdown/`, the
  ported copy under `docs/manuscript/` is refreshed by
  `docs/manuscript/port_from_source.py` — do not hand-edit the port.
- Do not duplicate the repository conventions here; link to
  [development](../development/README.md) for the test/coverage workflow.
