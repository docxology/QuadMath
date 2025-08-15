# Resources

This section provides comprehensive resources for learning about and working with Quadrays, synergetics, and the computational methods discussed in this manuscript.

## Core Concepts and Background

### Information Geometry and Optimization
- **Fisher information**: [Fisher information (reference)](https://en.wikipedia.org/wiki/Fisher_information) — see also Eq. \eqref{eq:supp_fim} in the equations appendix
- **Natural gradient**: [Natural gradient (reference)](https://en.wikipedia.org/wiki/Natural_gradient) — see also Eq. \eqref{eq:supp_natgrad} in the equations appendix

### Active Inference and Free Energy
- **Active Inference Institute**: [Welcome to Active Inference Institute](https://welcome.activeinference.institute/)
- **Comprehensive review**: [Active Inference — recent review (UCL Discovery, 2023)](https://discovery.ucl.ac.uk/id/eprint/10176959/1/1-s2.0-S1571064523001094-main.pdf)

### Mathematical Foundations
- **Tetrahedron volume formulas**: length-based [Cayley–Menger determinant](https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant) and determinant-based expressions on vertex coordinates (see [Tetrahedron – volume](https://en.wikipedia.org/wiki/Tetrahedron#Volume))
- **Exact determinants**: [Bareiss algorithm](https://en.wikipedia.org/wiki/Bareiss_algorithm), used in our integer tetravolume implementations
- **Optimization baseline**: the [Nelder–Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method), adapted here to the Quadray lattice

## Quadrays and Synergetics (Core Starting Points)

### Introductory Materials
- **Quadray coordinates (intro and conversions)**: [Urner – Quadray intro](https://www.grunch.net/synergetics/quadintro.html), [Urner – Quadrays and XYZ](https://www.grunch.net/synergetics/quadxyz.html)
- **Quadrays and the Philosophy of Mathematics**: [Urner – Quadrays and the Philosophy of Mathematics](https://www.grunch.net/synergetics/quadphil.html)
- **Synergetics background and IVM**: [Synergetics (Fuller, overview)](https://en.wikipedia.org/wiki/Synergetics_(Fuller))
- **Quadray coordinates overview**: [Quadray coordinates (reference)](https://en.wikipedia.org/wiki/Quadray_coordinates)

### Historical and Background Materials
- **RW Gray projects — Synergetics text**: [rwgrayprojects.com (synergetics)](http://www.rwgrayprojects.com/synergetics/s00/p0000.html)
- **Fuller FAQ**: [C. J. Fearnley's Fuller FAQ](https://www.cjfearnley.com/fuller-faq.pdf)
- **Synergetics resource list**: [C. J. Fearnley's resource page](https://www.cjfearnley.com/fuller-faq-2.html)
- **Wikieducator**: [Synergetics hub](https://wikieducator.org/Synergetics)
- **Quadray animation**: [Quadray.gif (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Quadray.gif)
- **Fuller Institute**: [BFI — Big Ideas: Synergetics](https://www.bfi.org/about-fuller/big-ideas/synergetics/)

## 4dsolutions Ecosystem: Comprehensive Computational Framework

The [4dsolutions organization](https://github.com/4dsolutions) provides the most extensive computational framework for Quadrays and synergetic geometry, spanning 29+ repositories with implementations across multiple programming languages.

### Core Computational Modules

#### Primary Python Libraries
- **Math for Wisdom (m4w)**: [m4w (repo)](https://github.com/4dsolutions/m4w)
  - **Quadray vectors and conversions**: [`qrays.py` (Qvector, SymPy-aware)](https://github.com/4dsolutions/m4w/blob/main/qrays.py)
  - **Synergetic tetravolumes and modules**: [`tetravolume.py` with PdF-CM vs native IVM and BEAST algorithms](https://github.com/4dsolutions/m4w/blob/main/tetravolume.py)

#### Cross-Language Validation
- **Rust implementation**: [rusty_rays](https://github.com/4dsolutions/rusty_rays) (performance-oriented)
  - Sources: [Rust library implementation](https://github.com/4dsolutions/rusty_rays/blob/master/src/lib.rs), [Rust command-line interface](https://github.com/4dsolutions/rusty_rays/blob/master/src/main.rs)
- **Clojure implementation**: [synmods](https://github.com/4dsolutions/synmods) (functional paradigm)
  - Sources: [`qrays.clj`](https://github.com/4dsolutions/synmods/blob/master/qrays.clj), [`ramping_up.clj`](https://github.com/4dsolutions/synmods/blob/master/ramping_up.clj)

### Primary Hub: School_of_Tomorrow (Python + Notebooks)

**Repository**: [School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow)

#### Core Modules
- **`qrays.py`**: Quadray implementation with normalization, conversions, and vector ops ([source](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/qrays.py))
- **`quadcraft.py`**: POV-Ray scenes for CCP/IVM arrangements, animations, and tutorials ([source](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/quadcraft.py))
- **`flextegrity.py`**: Polyhedron framework, concentric hierarchy, POV-Ray export ([source](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/flextegrity.py))
- **Additional modules**: `polyhedra.py`, `identities.py`, `smod_play.py` (synergetic modules)

#### Key Notebooks
- **`Qvolume.ipynb`**: Tom Ace 5×5 determinant with random-walk demonstrations ([source](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/Qvolume.ipynb))
- **`VolumeTalk.ipynb`**: Comparative analysis of bridging vs native tetravolume formulations ([source](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/VolumeTalk.ipynb))
- **`QuadCraft_Project.ipynb`**: 1,255 lines of interactive CCP navigation and visualization tutorials ([source](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/QuadCraft_Project.ipynb))
- **Additional notebooks**: `TetraBook.ipynb`, `CascadianSynergetics.ipynb`, `Rendering_IVM.ipynb`, `SphereVolumes.ipynb` (visual and curricular materials)

### Additional Repositories

#### Tetravolumes (Algorithms and Pedagogy)
- **Repository**: [tetravolumes](https://github.com/4dsolutions/tetravolumes)
- **Code**: [`tetravolume.py`](https://github.com/4dsolutions/tetravolumes/blob/master/tetravolume.py)
- **Notebooks**: [Atoms R Us.ipynb](https://raw.githubusercontent.com/4dsolutions/tetravolumes/refs/heads/master/Atoms%20R%20Us.ipynb), [Computing Volumes.ipynb](https://raw.githubusercontent.com/4dsolutions/tetravolumes/refs/heads/master/Computing%20Volumes.ipynb)

#### Visualization and Rendering
- **BookCovers**: VPython for interactive educational animations ([repo](https://github.com/4dsolutions/BookCovers))
  - Examples: [`bookdemo.py`](https://github.com/4dsolutions/BookCovers/blob/master/bookdemo.py), [`stickworks.py`](https://github.com/4dsolutions/BookCovers/blob/master/stickworks.py), [`tetravolumes.py`](https://github.com/4dsolutions/BookCovers/blob/master/tetravolumes.py)

### Educational Framework and Curricula

#### Oregon Curriculum Network (OCN)
- **OCN portal**: [OCN portal](http://www.4dsolutions.net/ocn/)
- **Python for Everyone**: [pymath page](http://www.4dsolutions.net/ocn/pymath.html)

#### Historical Documentation
- **Python5 notebooks**: [Polyhedrons 101.ipynb](https://raw.githubusercontent.com/4dsolutions/Python5/master/Polyhedrons%20101.ipynb)
- **Historical variants**: `qrays.py` also appears in [Python5 (archive)](https://github.com/4dsolutions/Python5/blob/master/qrays.py)
- **Python edu-sig archives**: [Python edu-sig archives](https://mail.python.org/pipermail/edu-sig/2000-May/000498.html) tracing 25+ years of development

### Media and Publications
- **YouTube demonstrations**: [Synergetics talk 1](https://www.youtube.com/watch?v=g14mu4uWD4E), [Synergetics talk 2](https://www.youtube.com/watch?v=i9oij02oje0), [Additional](https://www.youtube.com/watch?v=D0M1h_gjA_w)
- **Academia profile**: [Kirby Urner at Academia.edu](https://princeton.academia.edu/kirbyurner)

## Community Discussions and Collaborative Platforms

### Active Platforms
- **Math4Wisdom**: [Collaborative platform](https://coda.io/@daniel-ari-friedman/math4wisdom/ivm-xyz-40) with curated IVM↔XYZ conversion resources and cross-reference materials
- **synergeo discussion archive**: [Groups.io platform](https://groups.io/g/synergeo/topics) with ongoing community discussions and technical exchanges

### Historical Archives
- **GeodesicHelp threads**: [GeodesicHelp computations archive (Google Groups)](https://groups.google.com/g/GeodesicHelp/) documenting computational approaches and problem-solving techniques

## Related Projects and Applications

### Tetrahedral Voxel Engines
- **QuadCraft**: [Tetrahedral voxel engine using Quadrays](https://github.com/docxology/quadcraft/)

### Academic Publications
- **Flextegrity**: [Generating the Flextegrity Lattice (academia.edu)](https://www.academia.edu/44531954/Generating_the_Flextegrity_Lattice)

## Tooling and Technical Resources

### High-Precision Arithmetic
- **GCC libquadmath (binary128)**: [Official GCC libquadmath documentation](https://gcc.gnu.org/onlinedocs/libquadmath/index.html)

## Cross-Language and Cross-Platform Validation

### Implementation Consistency
- **Rust (rusty_rays)** and **Clojure (synmods)** mirror the Python algorithms for vector ops and tetravolumes, serving as independent checks on correctness and performance comparisons.
- **POV-Ray (quadcraft.py)** and **VPython (BookCovers)** demonstrate rendering pipelines for CCP/IVM scenes and educational animations.

### Context and Integration
These materials popularize the IVM/CCP/FCC framing of space, integer tetravolumes, and projective Quadray normalization. They inform the methods in this paper and complement the `src/` implementations (see `quadray.py`, `cayley_menger.py`, `linalg_utils.py`).

The ecosystem provides extensive validation, pedagogical context, and practical implementations that complement and extend the methods developed in this manuscript. Cross-language implementations serve as independent verification of algorithmic correctness while educational materials demonstrate practical applications across diverse computational environments.
