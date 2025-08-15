# Introduction

Quadray coordinates provide a tetrahedral basis for modeling space and computation, standing in contrast to Cartesian cubic frameworks. Originating in Buckminster Fuller's Synergetics, quadray coordinates enable the replacement of right-angle orthonormal assumptions, with 60-degree coordination and a unit tetrahedron of volume 1. This reframing yields striking integer relationships among common polyhedra and provides a natural account of space via close-packed spheres and the isotropic vector matrix (IVM).

In this synthetic review, we distinguish three internal meanings of “4D,” following a dot-notation that avoids cross-domain confusion:

- **Coxeter.4D** — four-dimensional Euclidean space (E⁴), as in classical polytope theory. Coxeter emphasizes that Euclidean 4D is not spacetime; see the Dover edition of Regular Polytopes (p. 119) for a clear statement to this effect; background on lattice packings in four dimensions aligns with the treatment in Conway & Sloane’s [Sphere Packings, Lattices and Groups](https://link.springer.com/book/10.1007/978-1-4757-6568-7).
- **Einstein.4D** — Minkowski spacetime (3D + time) with an indefinite metric; appropriate for relativistic physics but distinct from Euclidean E⁴.
- **Fuller.4D** — synergetics’ tetrahedral accounting of space using Quadrays (four non-negative coordinates with at least one zero after normalization) and the Isotropic Vector Matrix (IVM) = Cubic Close Packing (CCP) = Face-Centered Cubic (FCC) correspondence. This treats the regular tetrahedron as a natural unit container and emphasizes angle/shape relations independent of time/energy.

This paper unifies three threads:

- **Foundations**: Quadray coordinates and their relation to 4D modeling more generally, with explicit namespace usage (Coxeter.4D, Einstein.4D, Fuller.4D) to maintain clarity.
- **Optimization framework**: leverages integer volume quantization on tetrahedral lattices to achieve robust, discrete convergence.
- **Information geometry**: tools (e.g., Fisher Information, free-energy minimization) for interpreting optimization as geodesic motion on statistical manifolds.

Contributions:

- **Namespaces mapping**: Coxeter.4D (Euclidean E⁴), Einstein.4D (Minkowski spacetime), and Fuller.4D (Quadrays/IVM) → analytical tools and examples.
- **Quadray-adapted Nelder–Mead**: integer-lattice normalization and volume-level tracking.
- **Equations and methods**: comprehensive supplement with guidance for high-precision computation using `libquadmath`.
- **Discrete optimizer**: integer-valued variational descent over the IVM (`discrete_ivm_descent`) with animation tooling, connecting lattice geometry to information-theoretic objectives.

## Related work and background

Kirby Urner's expositions and implementations have been influential in making Quadray coordinates practical and accessible across multiple programming languages and educational contexts. The comprehensive [4dsolutions ecosystem](https://github.com/4dsolutions) provides extensive computational resources for Quadrays and synergetic geometry:

- **Foundational materials**: [Urner – Quadray intro](https://www.grunch.net/synergetics/quadintro.html), [Quadrays and XYZ](https://www.grunch.net/synergetics/quadxyz.html), [Quadrays and the Philosophy of Mathematics](https://www.grunch.net/synergetics/quadphil.html)
- **Python implementations**: Core modules [`qrays.py`](https://github.com/4dsolutions/m4w/blob/main/qrays.py) (Quadray vectors with SymPy support) and [`tetravolume.py`](https://github.com/4dsolutions/m4w/blob/main/tetravolume.py) (IVM volumes, BEAST modules, multiple algorithms)
- **Educational framework**: [School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow) with interactive tutorials and algorithm comparisons in [Qvolume.ipynb](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/Qvolume.ipynb) (Tom Ace 5×5) and [VolumeTalk.ipynb](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/VolumeTalk.ipynb) (bridging vs native)
- **Cross-language validation**: Independent implementations in [Rust](https://github.com/4dsolutions/rusty_rays), [Clojure](https://github.com/4dsolutions/synmods), POV-Ray pipelines ([quadcraft.py](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/quadcraft.py)), and [VPython animations](https://github.com/4dsolutions/BookCovers)
- **Historical context**: [Python edu-sig post (May 2000)](https://mail.python.org/pipermail/edu-sig/2000-May/000498.html)
- **Related work**: [QuadCraft](https://github.com/docxology/quadcraft/) is a tetrahedral voxel engine game using Quadray coordinates

The 4dsolutions organization spans 29+ repositories with implementations across Python, Rust, Clojure, POV-Ray, and VPython, providing extensive cross-language validation and educational resources. Core algorithmic modules include vector operations, volume calculations, visualization pipelines, and pedagogical frameworks that complement and validate the methods developed in this manuscript. See the comprehensive catalog in `07_resources.md`.

More background resources: 

- **Tetrahedron volume formulas**: length-based [Cayley–Menger determinant](https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant) and determinant-based expressions on vertex coordinates (see [Tetrahedron – volume](https://en.wikipedia.org/wiki/Tetrahedron#Volume)).
- **Exact determinants**: [Bareiss algorithm](https://en.wikipedia.org/wiki/Bareiss_algorithm), used in our integer tetravolume implementations.
- **Information geometry**: [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) and [natural gradient](https://en.wikipedia.org/wiki/Natural_gradient).
- **Optimization baseline**: the [Nelder–Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method), adapted here to the Quadray lattice.

## Companion code and tests

The manuscript is accompanied by a fully-tested Python codebase under `src/` with unit tests under `tests/`. Key artifacts used throughout the paper:

- **Quadray APIs**: `src/quadray.py` (`Quadray`, `integer_tetra_volume`, `ace_tetravolume_5x5`).
- **Determinant utilities**: `src/linalg_utils.py` (`bareiss_determinant_int`).
- **Length-based volume**: `src/cayley_menger.py` (`tetra_volume_cayley_menger`, `ivm_tetra_volume_cayley_menger`).
- **XYZ conversion**: `src/conversions.py` (`urner_embedding`, `quadray_to_xyz`).
- **Examples**: `src/examples.py` (`example_ivm_neighbors`, `example_volume`, `example_optimize`).

Figure 1 (graphical abstract): Panel A shows Quadray axes (A,B,C,D) under a symmetric embedding with wireframe context. Panel B shows close-packed spheres at the tetrahedron vertices (IVM/CCP/FCC, "twelve around one").

![**Figure 1: Quadray coordinate system overview (graphical abstract)**. **Panel A**: Four Quadray axes (A,B,C,D) rendered as colored directional arrows from the origin to the vertices of a regular tetrahedron under the default symmetric embedding. Each axis is distinctly colored (A=blue, B=orange, C=green, D=red) with axis labels positioned at the vertex endpoints. A light gray wireframe connects the four vertices to emphasize the tetrahedral geometry underlying the coordinate system. This panel illustrates the fundamental Fuller.4D direction-based structure where Quadrays represent four canonical directions in tetrahedral space rather than orthogonal Cartesian dimensions. **Panel B**: The same tetrahedral vertices shown as close-packed spheres with radius chosen so neighboring spheres kiss along tetrahedron edges, emphasizing the connection to the Isotropic Vector Matrix (IVM), Cubic Close Packing (CCP), and Face-Centered Cubic (FCC) arrangements. Each sphere is colored to match its corresponding axis from Panel A, with light edge wireframes providing geometric context. This visualization demonstrates how Quadray coordinates naturally align with dense sphere packing and the "twelve around one" coordination motif central to synergetics and Fuller.4D modeling.](../output/figures/graphical_abstract_quadray.png)

Tests illustrate expected behavior and edge cases (see `tests/`), and coverage is enforced at 100% for `src/`.
