# Introduction

## Abstract

We review a unified analytical framework for four-dimensional (4D) modeling with Quadray coordinates, synthesizing geometric foundations, optimization on tetrahedral lattices, and information geometry. Building on R. Buckminster Fuller's [Synergetics](https://en.wikipedia.org/wiki/Synergetics_(Fuller)) and Kirby Urner's computational implementations in the [4dsolutions ecosystem](https://github.com/4dsolutions) (Python, Rust, Clojure, and POV-Ray), we show that integer lattice constraints force the volume of any tetrahedron with lattice vertices to be an integer multiple of the unit tetrahedron's volume — a quantization into discrete "energy levels" that regularizes optimization on the tetrahedral lattice. We adapt the [Nelder–Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) to the Quadray lattice, define [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) in Quadray parameter space, and analyze optimization as geodesic motion on an information manifold via the [natural gradient](https://en.wikipedia.org/wiki/Natural_gradient). We distinguish three 4D namespaces — Coxeter.4D (Euclidean E⁴), Einstein.4D (Minkowski spacetime), and Fuller.4D (synergetics/Quadrays) — and map each to the analytical tools it supports. The result is a cohesive, interpretable approach for robust, geometry-grounded computation in 4D. All source code for the manuscript is available at [QuadMath](https://github.com/docxology/quadmath).

**Keywords**: Quadray coordinates, 4D geometry, tetrahedral lattice, integer volume quantization, information geometry, optimization, synergetics, active inference.

## Overview

Quadray coordinates provide a tetrahedral basis for modeling space and computation, standing in contrast to Cartesian cubic frameworks. Originating in Buckminster Fuller's Synergetics, Quadray coordinates replace right-angle orthonormal assumptions with 60-degree coordination and a unit tetrahedron of volume 1. This reframing yields striking integer relationships among common polyhedra and provides a natural account of space via close-packed spheres and the isotropic vector matrix (IVM).

This paper unifies three threads:

- **Foundations**: Quadray coordinates and their relation to 4D modeling more generally, with explicit namespace usage (Coxeter.4D, Einstein.4D, Fuller.4D) to maintain clarity.
- **Optimization framework**: Leverages integer volume quantization on tetrahedral lattices to achieve robust, discrete convergence.
- **Information geometry**: Tools (e.g., Fisher Information, free-energy minimization) for interpreting optimization as geodesic motion on statistical manifolds.

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

The manuscript is accompanied by a fully-tested Python codebase under `src/` with unit tests under `tests/`. Key artifacts used throughout the paper:

- **Quadray APIs**: `src/quadmath/core/quadray.py` (`Quadray`, `integer_tetra_volume`, `ace_tetravolume_5x5`).
- **Determinant utilities**: `src/quadmath/core/linalg_utils.py` (`bareiss_determinant_int`).
- **Length-based volume**: `src/quadmath/core/cayley_menger.py` (`tetra_volume_cayley_menger`, `ivm_tetra_volume_cayley_menger`).
- **XYZ conversion**: `src/quadmath/lattice/conversions.py` (`urner_embedding`, `quadray_to_xyz`).
- **Examples**: `src/quadmath/core/examples.py` (`example_ivm_neighbors`, `example_volume`, `example_optimize`).

For comprehensive background resources, computational implementations, and related work, see the [Resources](99_resources.md) section.

## Reproducibility and Data Availability

- The manuscript Markdown and code to generate the PDF are available on the project repository (`QuadMath` on GitHub, `@docxology` username). See the repository home page for source, figures, and scripts: [QuadMath repository](https://github.com/docxology/QuadMath). The repository is also archived on [Zenodo record 16887791](https://zenodo.org/records/16887791) with DOI 10.5281/zenodo.16887791.
- The manuscript is licensed under the Apache License 2.0. See the [LICENSE](../../LICENSE) file for details.
- The manuscript is accompanied by a fully-tested Python codebase under `src/` with unit tests under `tests/`, complemented by extensive cross-validation against Kirby Urner's reference implementations in the [4dsolutions ecosystem](https://github.com/4dsolutions). See the [Resources](99_resources.md) section for comprehensive details on computational implementations and validation.
- All figures referenced in the manuscript are generated by scripts under `quadmath/scripts/` and saved to `quadmath/output/` with lightweight CSV/NPZ alongside images.
- Tests accompany all methods under `src/` and enforce 100% coverage for `src/`.
- Symbols and notation are standardized across sections; see [Appendix: Symbols and Glossary](98_symbols_glossary.md) for a consolidated table of variables and constants used throughout. Equation labels (e.g., Eq. \eqref{eq:lattice_det} and Eq. \eqref{eq:fim}) and figure labels are automatically numbered by LaTeX for consistent cross-referencing.
- The manuscript is a work in progress and will be updated as the project progresses. There may be errors and missing references, check all methods and equations for consistency.

## Graphical Abstract

**Panel A** shows the four Quadray axes (A, B, C, D) under the default symmetric embedding with a tetrahedron wireframe. **Panel B** shows the same vertices as close-packed spheres (IVM = CCP = FCC) sized to kiss along the tetrahedron edges.

![**Quadray coordinate system overview (graphical abstract)**. **Panel A**: The four Quadray axes — rays A, B, C, D from the origin to the vertices of a reference regular tetrahedron under the default symmetric embedding — drawn as colored arrows (A=blue, B=orange, C=green, D=red) with axis labels at the vertex endpoints and a light gray wireframe joining the four vertices. Notice that the axes are four canonical directions (spokes) separated by 60° angles, not orthogonal Cartesian dimensions — the defining "directions, not dimensions" character of Fuller.4D. **Panel B**: The same four vertices shown as close-packed spheres, each colored to match its Panel A axis and with radius equal to half the minimum edge length, so neighboring spheres kiss exactly along the tetrahedron edges; a light black wireframe marks those edges. This is the local building block of the Isotropic Vector Matrix (IVM) = Cubic Close Packing (CCP) = Face-Centered Cubic (FCC) correspondence; extending the kissing arrangement across the lattice yields the "twelve around one" coordination central to synergetics. What to notice: one and the same tetrahedral skeleton underlies both the direction-based Quadray coordinate system and the densest sphere packing in 3D, which is why integer Quadray coordinates and integer sphere counts (and hence integer volumes) live on a single lattice. Generated by `quadmath/scripts/graphical_abstract_quadray.py` (output at `quadmath/output/figures/graphical_abstract_quadray.png`).](figures/graphical_abstract_quadray.png)
