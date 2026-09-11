# `quadmath/scripts/` — Thin-Orchestrator Contract

## Contract

Scripts in this directory are thin orchestrators ONLY: argument handling, path
bootstrap, logging/environment setup, and a single delegated call into
importable entrypoints under `src/`.

- Business/data/plot/analysis logic lives in `src/` (importable, tested under
  `tests/`, 100% coverage gate). New behavior goes in `src/`, never in scripts.
- There is NO installed package: every script bootstraps `<repo>/src` onto
  `sys.path` (`_repo_root()` / `_ensure_src_on_path()`) and imports the
  factored package — `from quadmath.core.quadray import ...`,
  `from quadmath.paths import ...`.
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
| `make_all_figures.py` | 20 figure scripts (subprocess); `quadmath.paths` |
| `validate_markdown.py` | none (stdlib only) |
| `generate_glossary.py` | `quadmath.tools.glossary_gen` |
| `information_demo.py` | `quadmath.inference.information`, `quadmath.optimize.discrete_variational`, `quadmath.core.metrics`, `quadmath.viz.visualize`, `quadmath.core.quadray`, `quadmath.paths` |
| `active_inference_figures.py` | `quadmath.inference.information`, `quadmath.paths` |
| `simplex_animation.py` | `quadmath.optimize.nelder_mead_quadray`, `quadmath.viz.visualize`, `quadmath.core.quadray`, `quadmath.paths` |
| `discrete_variational_demo.py` | `quadmath.optimize.discrete_variational`, `quadmath.viz.visualize`, `quadmath.core.quadray`, `quadmath.paths` |
| `volumes_demo.py` | `quadmath.core.cayley_menger`, `quadmath.core.quadray`, `quadmath.paths` |
| `ivm_neighbors.py` | `quadmath.viz.visualize`, `quadmath.core.quadray`, `quadmath.paths` |
| `quadray_clouds.py` | `quadmath.lattice.conversions`, `quadmath.core.quadray`, `quadmath.paths` |
| `polyhedra_quadray_constructions.py` | `quadmath.paths` (figure construction inline) |
| `graphical_abstract_quadray.py` | `quadmath.core.quadray`, `quadmath.paths` |
| `sympy_formalisms.py` | `quadmath.core.symbolic`, `quadmath.core.quadray` |
| `gpu_acceleration_demo.py` | `quadmath.core.quadray` |
| `ivm_field_demo.py` | `quadmath.lattice.ivm_field`, `quadmath.core.quadray`, `quadmath.paths` |
| `ivm_dynamics_demo.py` | `quadmath.lattice.ivm_dynamics` |
| `lattice_gallery.py` | `quadmath.viz.vis_lattice` (`gallery` composer), `quadmath.paths` |
| `stats_gallery.py` | `quadmath.viz.vis_stats` (`gallery` composer), `quadmath.paths` |
| `animation_gallery.py` | `quadmath.viz.animations` (`simplex_frames`/`lattice_frames`/`diffusion_frames`/`frames_to_gif`), `quadmath.paths` |
| `learning_gallery.py` | `quadmath.learn.learning_eval` (`GradientDescentTrainer`), `quadmath.viz.plots` (`plot_loss_history`), `quadmath.paths` |
| `quaternion_gallery.py` | `quadmath.viz.plots` (`plot_slerp_path`), `quadmath.core.quadray`, `quadmath.paths` |
| `animation_stills.py` | `quadmath.viz.animations` (`frames_strip` + frame builders), `quadmath.paths` |
| `stats_diagnostics_gallery.py` | `quadmath.viz.vis_stats` (`plot_ci_bars`), `quadmath.stats.statistics` (CI families), `quadmath.paths` |

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
- Who runs what: `render_pdf.sh` directly runs 3 scripts
  (`make_all_figures.py`, `generate_glossary.py`,
  `validate_markdown.py`); all 20 figure/data/GIF generators are reached
  via `make_all_figures.py`. When adding a script, update that list —
  see "Adding a script" below.
- Never commit `__pycache__/` or `.DS_Store`; `.gitignore` covers both
  (force-added `.pyc` files were removed in the 2026-09 scripts audit).
- Adding a script: follow the README template, then register it in the
  `make_all_figures.py` list and/or the `render_pdf.sh` array as appropriate.
