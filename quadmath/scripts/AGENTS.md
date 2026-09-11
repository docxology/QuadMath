# `quadmath/scripts/` — Thin-Orchestrator Contract

## Contract

Scripts in this directory are thin orchestrators ONLY: argument handling, path
bootstrap, logging/environment setup, and a single delegated call into
importable entrypoints under `src/`.

- Business/data/plot/analysis logic lives in `src/` (importable, tested under
  `tests/`, 100% coverage gate). New behavior goes in `src/`, never in scripts.
- There is NO installed `quadray` package: every script bootstraps `<repo>/src`
  onto `sys.path` (`_repo_root()` / `_ensure_src_on_path()`) and imports
  top-level src modules — `from quadray import ...`, `from paths import ...`.
- All entry logic sits behind `if __name__ == "__main__":` — keep scripts
  import-safe (`tests/test_sympy_formalisms.py` imports
  `quadmath/scripts/sympy_formalisms.py` directly).
- Known inline-logic exceptions (documented, do not grow):
  `validate_markdown.py` (stdlib-only checks, no src delegation),
  `polyhedra_quadray_constructions.py` (figure construction inline; src
  delegation = `paths.py` only), `render_pdf.sh` / `clean_output.sh`
  (bash orchestration).

## Inventory

| Script | Delegates to |
|--------|--------------|
| `render_pdf.sh` | all scripts below (its `scripts` array) + `pandoc`/`xelatex` |
| `clean_output.sh` | filesystem only |
| `make_all_figures.py` | 15 figure scripts (subprocess); `src/paths.py` |
| `validate_markdown.py` | none (stdlib only) |
| `generate_glossary.py` | `src/glossary_gen.py` |
| `information_demo.py` | `src/information.py`, `src/discrete_variational.py`, `src/metrics.py`, `src/visualize.py`, `src/quadray.py`, `src/paths.py` |
| `active_inference_figures.py` | `src/information.py`, `src/paths.py` |
| `simplex_animation.py` | `src/nelder_mead_quadray.py`, `src/visualize.py`, `src/quadray.py`, `src/paths.py` |
| `discrete_variational_demo.py` | `src/discrete_variational.py`, `src/visualize.py`, `src/quadray.py`, `src/paths.py` |
| `volumes_demo.py` | `src/cayley_menger.py`, `src/quadray.py`, `src/paths.py` |
| `ivm_neighbors.py` | `src/visualize.py`, `src/quadray.py`, `src/paths.py` |
| `quadray_clouds.py` | `src/conversions.py`, `src/quadray.py`, `src/paths.py` |
| `polyhedra_quadray_constructions.py` | `src/paths.py` (figure construction inline) |
| `graphical_abstract_quadray.py` | `src/quadray.py`, `src/paths.py` |
| `sympy_formalisms.py` | `src/symbolic.py`, `src/quadray.py` |
| `gpu_acceleration_demo.py` | `src/quadray.py` |
| `ivm_field_demo.py` | `src/ivm_field.py`, `src/quadray.py`, `src/paths.py` |
| `ivm_dynamics_demo.py` | `src/ivm_dynamics.py` |
| `lattice_gallery.py` | `src/vis_lattice.py` (`gallery` composer), `src/paths.py` |
| `stats_gallery.py` | `src/vis_stats.py` (`gallery` composer), `src/paths.py` |

## Gotchas

- Run from the repo root: `uv run python quadmath/scripts/<script>.py`
  (`render_pdf.sh` picks `uv run python`, falling back to `python3`).
- `MPLBACKEND=Agg` must be set before matplotlib import. `render_pdf.sh`
  exports it and `make_all_figures.py` passes it to subprocesses via
  `env.setdefault`; standalone runs rely on each script setting it.
- Deferred imports: import src modules only after `_ensure_src_on_path()`;
  most scripts import inside `main()`.
- `make_all_figures.py` writes `figure_manifest.txt` from stdout lines ending
  in `.png/.mp4/.pdf/.csv/.npz` — generators MUST print their output paths.
- `validate_markdown.py --strict` exits 1 on any warning (plain `sys.argv`
  check; there is no argparse).
- `render_pdf.sh` requires `pandoc` + `xelatex`; `LOG_LEVEL=0..3` controls
  verbosity (0 = debug, default 1 = info).
- Who runs what: `render_pdf.sh` directly runs 12 scripts (including
  `graphical_abstract_quadray.py`, `polyhedra_quadray_constructions.py`,
  `generate_glossary.py`, `validate_markdown.py`, `make_all_figures.py`);
  `make_all_figures.py` runs 15 figure scripts. `discrete_variational_demo.py`
  and `gpu_acceleration_demo.py` are reached only via `make_all_figures.py`;
  `graphical_abstract_quadray.py` and `polyhedra_quadray_constructions.py`
  only directly in `render_pdf.sh`.
- Never commit `__pycache__/` or `.DS_Store`; `.gitignore` covers both
  (force-added `.pyc` files were removed in the 2026-09 scripts audit).
- Adding a script: follow the README template, then register it in the
  `make_all_figures.py` list and/or the `render_pdf.sh` array as appropriate.
