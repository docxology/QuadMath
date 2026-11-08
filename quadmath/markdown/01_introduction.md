# Quadray Geometry, Lattice Optimization, and Information Metrics {#introduction}

## Abstract

We review tetrahedral coordinates for ordinary three-dimensional space, exact lattice volumes, discrete optimization, and information metrics. A physical Quadray is a common-offset equivalence class of four components, with three independent spatial degrees of freedom; it is not Euclidean four-space. Integer Quadray vertices have exact native tetravolumes in quarter-unit increments, not necessarily whole units. The Python implementation supplies determinant formulas, lattice-adapted Nelder–Mead and twelve-neighbor descent, score-Gram estimators, and damped natural-gradient increments. These illustrate mathematical relationships rather than establish universal convergence, physical energy quantization, or application-level robustness. Natural-gradient flows are metric-aware steepest descent, not generally geodesics. The [namespace reference](02_4d_namespaces.md), [coordinate methods](03_quadray_methods.md), and [conversion specification](14_conversions_spec.md) separate these contracts.

**Keywords**: Quadray coordinates, rank-three quotient, tetrahedral lattice, rational tetravolume, information geometry, discrete optimization.

## Coordinates, packing, and optimization scope

Quadrays use four symmetric tetrahedral directions to describe Euclidean three-space. Distinct origin-to-vertex basis directions make angles $\arccos(-1/3)$; the 60-degree angles belong to regular tetrahedron faces and suitable FCC neighbor directions, not pairs of these basis rays. The regular unit tetrahedron has native volume 1 and edge $2\sqrt2$ in `DEFAULT_EMBEDDING`. Close-packed sphere centers form one residue class of the ambient integer Quadray quotient.

This paper unifies three threads:

- **Foundations**: Quadray coordinates and their relation to 4D modeling more generally, with explicit namespace usage (Coxeter.4D, Einstein.4D, Fuller.4D) to maintain clarity.
- **Optimization framework**: Tracks exact rational simplex volumes and discrete moves without a general convergence or efficiency guarantee.
- **Information geometry**: Separates score-based Fisher metrics, illustrative loss-gradient Gram matrices, and free-energy toy models.

## 4D Namespace Framework

In this synthetic review, we distinguish three internal meanings of "4D," following a dot-notation that avoids cross-domain confusion. For comprehensive details, see [Section 2: 4D Namespaces](02_4d_namespaces.md).

- **Coxeter.4D** — four-dimensional Euclidean space (E⁴) with mutually orthogonal axes and a positive-definite metric: the setting of classical polytope theory, and explicitly not spacetime (Coxeter, Regular Polytopes, Dover ed., p. 119). Lattice packings in four dimensions align with the treatment in Conway & Sloane's [Sphere Packings, Lattices and Groups](https://link.springer.com/book/10.1007/978-1-4757-6568-7).
- **Einstein.4D** — Minkowski spacetime: three space dimensions plus time, with an indefinite metric of signature $(-,+,+,+)$; the arena of relativistic physics, distinct from Euclidean E⁴.
- **Fuller.4D** — synergetics' tetrahedral accounting of space: Quadray coordinates (four non-negative coordinates with at least one zero after normalization) on the Isotropic Vector Matrix (IVM) = Cubic Close Packing (CCP) = Face-Centered Cubic (FCC) lattice, with the regular tetrahedron as the natural unit container; it emphasizes angle/shape relations independent of time and energy.

## Contributions

The paper makes the following key contributions:

- **Namespaces mapping**: Coxeter.4D (Euclidean E⁴), Einstein.4D (Minkowski spacetime), and Fuller.4D (Quadrays/IVM) → analytical tools and examples.
- **Quadray-adapted Nelder–Mead**: Integer-lattice normalization and volume-level tracking.
- **Equations and methods**: Comprehensive supplement with guidance for high-precision computation using `libquadmath`.
- **Discrete optimizer**: Integer-valued variational descent over the IVM (`discrete_ivm_descent`) with animation tooling, connecting lattice geometry to information-theoretic objectives.

## Manuscript Structure

- **Introduction**: motivates Quadrays, clarifies 4D namespaces, and summarizes contributions.
- **Methods**: details coordinate conventions, exact tetravolumes, conversions, and lattice-aware optimization methods (Nelder–Mead and discrete IVM descent).
- **Results**: empirical comparisons and demonstrations are shown inline and saved under `quadmath/output/` (PNG/CSV/NPZ/MP4) for reproducibility.
- **Discussion**: interprets results, limitations, and implications; outlines future work.
- **Appendices**: equations, free-energy background, and a consolidated symbols/glossary with an auto-generated API index.

## Companion Code and Tests

The companion Python implementation is under `src/quadmath/`, with numerical regressions under `tests/`. This chapter identifies source surfaces; it does not certify a fresh test or coverage run. Key artifacts:

- **Quadray APIs**: `src/quadmath/core/quadray.py` (`Quadray`, `integer_tetra_volume`, `ace_tetravolume_5x5`).
- **Determinant utilities**: `src/quadmath/core/linalg_utils.py` (`bareiss_determinant_int`).
- **Length-based volume**: `src/quadmath/core/cayley_menger.py` (`tetra_volume_cayley_menger`, `ivm_tetra_volume_cayley_menger`).
- **XYZ conversion**: `src/quadmath/lattice/conversions.py` (`urner_embedding`, `quadray_to_xyz`).
- **Examples**: `src/quadmath/core/examples.py` (`example_ivm_neighbors`, `example_volume`, `example_optimize`).

For comprehensive background resources, computational implementations, and related work, see the [Resources](07_resources.md) section.

## Reproducibility and Data Availability

- The manuscript Markdown and code to generate the PDF are available on the project repository (`QuadMath` on GitHub, `@docxology` username). See the repository home page for source, figures, and scripts: [QuadMath repository](https://github.com/docxology/QuadMath). The repository is also archived on [Zenodo record 16887791](https://zenodo.org/records/16887791) with DOI 10.5281/zenodo.16887791.
- The manuscript is licensed under the Apache License 2.0. See the [LICENSE](../../LICENSE) file for details.
- Numerical regressions and external reference implementations provide comparison routes, not an assertion that all external projects have been cross-validated against this checkout.
- Figure scripts under `quadmath/scripts/` normally write to `quadmath/output/`; `QUADMATH_OUTPUT_DIR` selects an isolated output root. CSV/NPZ companions are script-specific.
- Coverage requirements are project policy; current results must come from a dated run.
- Symbols and notation are standardized across sections; see [Appendix: Symbols and Glossary](10_symbols_glossary.md) for a consolidated table of variables and constants used throughout. Equation labels (e.g., Eq. \eqref{eq:lattice_det} and Eq. \eqref{eq:fim}) and figure labels are automatically numbered by LaTeX for consistent cross-referencing.
- The manuscript is a work in progress and will be updated as the project progresses. There may be errors and missing references, check all methods and equations for consistency.

## Native basis, common-offset gauge, and FCC contacts {#graphical-abstract}

Panel A displays the actual columns of `DEFAULT_EMBEDDING`, not unit-normalized substitutes. Their sum is zero, making the common-offset gauge visible. Panel B displays the complete thirteen-site cluster: a center and twelve FCC neighbors, not merely four tangent spheres.

![**Native Quadray basis and complete twelve-around-one FCC neighborhood.** Left: the four basis endpoints $A=(1,1,1)$, $B=(-1,1,-1)$, $C=(-1,-1,1)$, $D=(1,-1,-1)$ in scale-one XYZ. Their sum is zero. Right: the origin plus all twelve permutations of the Quadray move $(2,1,1,0)$, embedded as permutations of $(\pm2,\pm2,0)$. Center-to-neighbor distance is $2\sqrt2$ and the tangent-sphere radius is $\sqrt2$; the outer centers form a cuboctahedron. The cluster demonstrates local FCC coordination, not a finite proof of global packing optimality or integer volumes for all Quadrays. Generated by `quadmath/scripts/graphical_abstract_quadray.py`; the reusable renderer also emits an SVG companion.](../output/figures/graphical_abstract_quadray.png)

The reusable API is `quadmath.viz.abstracts.render_quadray_abstract`. It owns
and closes its figure and returns the written PNG/SVG paths; the existing
script is the command-line entry. See the [operational workflow](../../WORKFLOW.md)
for isolated output generation.
