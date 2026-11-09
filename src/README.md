# QuadMath Source Modules: Runtime Responsibilities and Exactness

`src/quadmath/` is the importable package shipped in wheels. It does not include
checkout manuscript scripts, PDFs or Lean sources. Installation and explicit
Python 3.13 setup are in the [root guide](../README.md#quick-start).

## Module Map

| Package/module | Responsibility | Canonical explanation |
|---|---|---|
| `core.quadray` | Four-component labels, canonical gauge, embedding, exact metric/affine/oriented operations, native tetra-volumes and quaternions | [Methods](../quadmath/markdown/03_quadray_methods.md) |
| `core.linalg_utils` | Exact Bareiss determinant/rank and integer adjugate | [Equations](../quadmath/markdown/08_equations_appendix.md) |
| `core.cayley_menger` | Length-based volume and tetrahedron radius calculations | [Methods](../quadmath/markdown/03_quadray_methods.md) |
| `core.geometry`, `core.metrics`, `core.symbolic`, `core.examples` | Lorentz helpers, information diagnostics, symbolic expressions and example configurations | [Namespaces](../quadmath/markdown/02_4d_namespaces.md), [glossary](../quadmath/markdown/10_symbols_glossary.md) |
| `lattice.omni_numbering`, `lattice.lattice_search` | BFS shell/site numbering and precomputed finite-ball queries | [Lattice tooling](../quadmath/markdown/13_lattice_tooling.md) |
| `lattice.conversions` | Shape-validated embedding and strict represented-value rational inverse, distinct from floating snapping | [Conversions](../quadmath/markdown/14_conversions_spec.md) |
| `lattice.ivm_field` | Centered-norm shell enumeration, soft field fitting and hard-boundary harmonic extension | [Field learning](../quadmath/markdown/11_ivm_field_learning.md) |
| `lattice.ivm_dynamics` | Euclidean-radius graphs, distinct update laws, trajectories and coupling identification | [Dynamics](../quadmath/markdown/12_ivm_dynamics.md) |
| `learn.learning_eval` | Site-fold selection, temporal holdout, learning curves and linear baselines | [Evaluation](../quadmath/markdown/15_learning_evaluation.md) |
| `optimize` | Quadray Nelder–Mead and discrete IVM descent | [Optimization](../quadmath/markdown/04_optimization_in_4d.md) |
| `inference.information` | Fisher pullbacks, free energy and Active Inference helpers | [Inference](../quadmath/markdown/09_free_energy_active_inference.md) |
| `stats.statistics`, `stats.benchmarks` | Range-safe `sample_mean`, inference, anchored `scaling_fit`/`scaling_predict`, versus machine-dependent timing | [Analysis operations](../docs/analysis/README.md) |
| `viz` | Reusable plots, animations, lattice/statistics galleries and graphical abstract | [Figure guide](../quadmath/README.md) |
| `validate.validate` | Numerical quaternion property reports | [API glossary](../quadmath/markdown/10_symbols_glossary.md) |
| `pipeline` | Typed `Step`/`Pipeline` composition over runtime operations | [API glossary](../quadmath/markdown/10_symbols_glossary.md) |
| `paths`, `tools.atomic_write`, `tools.glossary_gen` | Output roots, atomic writes and API-document generation | [Architecture](../ARCHITECTURE.md) |

## Exact geometry consumer example

After setup, run with `uv run --python 3.13 python`:

```python
from fractions import Fraction
from quadmath.core.quadray import (
    DEFAULT_EMBEDDING, Quadray, integer_tetra_volume, squared_distance_exact, to_xyz,
)

origin = Quadray(0, 0, 0, 0)
a, b, c = Quadray(2, 1, 1, 0), Quadray(1, 2, 1, 0), Quadray(1, 1, 2, 0)
print(Quadray(3, 2, 2, 1).normalize())  # same physical class as a
print(to_xyz(a, DEFAULT_EMBEDDING))
print(squared_distance_exact(origin, a))  # 8 at scale one
print(integer_tetra_volume(origin, a, b, c))  # Fraction(1, 1)
print(integer_tetra_volume(origin, Quadray(1, 0, 0, 0),
                           Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0)))  # 1/4
```

The name `integer_tetra_volume` describes the exact integer determinant
construction, not an always-integer result. Its native volume is `abs(det)/4`
and scale-one XYZ volume is `8/3` times native. Do not replace this with the
unit-edge or QuadCraft-rescaled volume convention. Use
[SPEC.md](../SPEC.md) for all conventions, not an alternate local derivation.

## Package Re-Exports

`quadmath.__init__` exposes a selected top-level API, not every symbol in every
module. Prefer concrete imports when names are ambiguous: both field and
dynamics modules expose ball constructors with different radius meanings, and
both gallery modules expose `gallery`. Inspect actual `__all__`/initializer
exports before adding names; do not assume star re-exports make collisions safe.
Internal imports use the absolute `quadmath.<subpackage>.<module>` form rather
than obsolete flat `quadray` or `paths` modules.

## Rendering and output ownership

Use `quadmath.paths.get_output_dir`, `get_figure_dir` and `get_data_dir` when a
renderer follows package output policy. They return strings, create directories
and honor `QUADMATH_OUTPUT_DIR`. Without an override they search for checkout
markers; wheel consumers should supply an explicit output root.

Reusable plotting lives in `viz`, including `abstracts.py`; scripts only choose
inputs and print paths. Follow `_common.figure_scope`/`save_figure` lifecycle
patterns so figures close even if saving raises. Caller-owned axes primitives
and complete saved-figure renderers have different ownership: document which
one an API provides. Headless consumers select `MPLBACKEND=Agg` before imports.
The shell PDF renderer is separate and does not honor this output override.

## Development Standards

Keep public annotations/docstrings accurate about exact versus floating results,
units, gauge, shapes, failure modes and resources. Numerical regressions should
exercise behavior rather than source text or coverage totals. The configured
coverage floor is an integration gate, not a measured status claim. For test
commands see [tests/README.md](../tests/README.md); for delegated-work and build
rules see [AGENTS.md](AGENTS.md) and [WORKFLOW.md](../WORKFLOW.md).
