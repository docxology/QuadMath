# scripts/ — Agent Notes

Manuscript generation pipeline (verified 2026-08-29; categorized table in the
existing README.md). Key entry points: `render_pdf.sh` (main build orchestrator —
runs generators, validates, builds PDFs), `clean_output.sh` (safe clean —
everything in `../output/` is regenerable), `make_all_figures.py`,
`validate_markdown.py`. Individual generators: figure demos
(`discrete_variational_demo.py`, `gpu_acceleration_demo.py`, `information_demo.py`,
`volumes_demo.py`, `quadray_clouds.py`, `simplex_animation.py`,
`ivm_neighbors.py`, `polyhedra_quadray_constructions.py`),
`active_inference_figures.py`, `graphical_abstract_quadray.py`,
`sympy_formalisms.py`, `generate_glossary.py`. `__pycache__/` present — ignore.
Run via `uv run`/repo root per the root README quick start.
