# Resources (References and Further Reading)

## Quadrays and Synergetics (core starting points)

 - **Quadray coordinates (intro and conversions)**: [Urner – Quadray intro](https://www.grunch.net/synergetics/quadintro.html), [Urner – Quadrays and XYZ](https://www.grunch.net/synergetics/quadxyz.html)
- **Synergetics background and IVM**: [Synergetics (Fuller, overview)](https://en.wikipedia.org/wiki/Synergetics_(Fuller))

## 4dsolutions (Kirby Urner) — repositories and key artifacts

- **Organization overview**: [4dsolutions (GitHub org)](https://github.com/4dsolutions) — Python-centered explorations of Quadrays and synergetic geometry.
- **Math for Wisdom (m4w)**: [m4w (repo)](https://github.com/4dsolutions/m4w)
  - Quadray vectors and conversions: [`qrays.py` (Qvector, SymPy-aware)](https://github.com/4dsolutions/m4w/blob/main/qrays.py)
  - Synergetic tetravolumes and modules: [tetravolume.py - PdF/CM vs native IVM, BEAST](https://github.com/4dsolutions/m4w/blob/main/tetravolume.py)
- **School_of_Tomorrow (notebooks/code)**: [School_of_Tomorrow (repo)](https://github.com/4dsolutions/School_of_Tomorrow)
  - Tom Ace 5×5 determinant: [Qvolume.ipynb](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/Qvolume.ipynb)
  - Bridging vs native tetravolumes: [VolumeTalk.ipynb](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/VolumeTalk.ipynb)
- **Historical variants**: `qrays.py` also appears in [Python5 (archive)](https://github.com/4dsolutions/Python5/blob/master/qrays.py).

Context: These materials popularize the IVM/CCP/FCC framing of space, integer tetravolumes, and projective Quadray normalization. They inform the methods in this paper and complement the `src/` implementations (see `quadray.py`, `cayley_menger.py`, `linalg_utils.py`).

## Comprehensive index of 4dsolutions artifacts (selected)

### Primary hub: School_of_Tomorrow (Python + notebooks)

- **Repository**: [School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow)
- **Core modules**:
  - `qrays.py`: Quadray implementation with normalization, conversions, and vector ops ([qrays.py source — School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/qrays.py))
  - `quadcraft.py`: POV-Ray scenes for CCP/IVM arrangements, animations, and tutorials ([quadcraft.py source — School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/quadcraft.py))
  - `flextegrity.py`: Polyhedron framework, concentric hierarchy, POV-Ray export ([flextegrity.py source — School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/flextegrity.py))
  - Additional: `polyhedra.py`, `identities.py`, `smod_play.py` (synergetic modules)
- **Notebooks**:
  - `QuadCraft_Project.ipynb`: Interactive tutorials; CCP navigation and tetra demos ([QuadCraft_Project.ipynb — School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/QuadCraft_Project.ipynb))
  - `Qvolume.ipynb`: Tom Ace 5×5 determinant; random-walk IVM volumes ([Qvolume.ipynb — School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/Qvolume.ipynb))
  - `VolumeTalk.ipynb`: Bridging (CM/PdF) vs native (Ace/GdJ) tetravolumes ([VolumeTalk.ipynb — School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/VolumeTalk.ipynb))
  - `TetraBook.ipynb`, `CascadianSynergetics.ipynb`, `Rendering_IVM.ipynb`, `SphereVolumes.ipynb` (visual and curricular materials)

### Additional repositories

- **tetravolumes**: algorithms and pedagogy for tetra volumes
  - Repo: [tetravolumes](https://github.com/4dsolutions/tetravolumes)
  - Code: [`tetravolume.py`](https://github.com/4dsolutions/tetravolumes/blob/master/tetravolume.py)
  - Notebooks: [Atoms R Us.ipynb](https://raw.githubusercontent.com/4dsolutions/tetravolumes/refs/heads/master/Atoms%20R%20Us.ipynb), [Computing Volumes.ipynb](https://raw.githubusercontent.com/4dsolutions/tetravolumes/refs/heads/master/Computing%20Volumes.ipynb)

- **rusty_rays**: Rust port highlighting cross-language consistency
  - Repo: [rusty_rays](https://github.com/4dsolutions/rusty_rays)
  - Sources: [Rust library implementation](https://github.com/4dsolutions/rusty_rays/blob/master/src/lib.rs), [Rust command-line interface](https://github.com/4dsolutions/rusty_rays/blob/master/src/main.rs)

- **synmods**: Clojure/functional approach to Quadrays and synergetic modules
  - Repo: [synmods](https://github.com/4dsolutions/synmods)
  - Sources: [`qrays.clj`](https://github.com/4dsolutions/synmods/blob/master/qrays.clj), [`ramping_up.clj`](https://github.com/4dsolutions/synmods/blob/master/ramping_up.clj)

- **BookCovers**: VPython for interactive educational animations
  - Repo: [BookCovers](https://github.com/4dsolutions/BookCovers)
  - Examples: [`bookdemo.py`](https://github.com/4dsolutions/BookCovers/blob/master/bookdemo.py), [`stickworks.py`](https://github.com/4dsolutions/BookCovers/blob/master/stickworks.py), [`tetravolumes.py`](https://github.com/4dsolutions/BookCovers/blob/master/tetravolumes.py)

### Additional educational resources

- **Oregon Curriculum Network (OCN)**: [OCN portal](http://www.4dsolutions.net/ocn/)
- **Python for Everyone**: [pymath page](http://www.4dsolutions.net/ocn/pymath.html)
- **Python5 notebooks**: [Polyhedrons 101.ipynb](https://raw.githubusercontent.com/4dsolutions/Python5/master/Polyhedrons%20101.ipynb)

### Media and publications

- **YouTube demonstrations**: [Synergetics talk 1](https://www.youtube.com/watch?v=g14mu4uWD4E), [Synergetics talk 2](https://www.youtube.com/watch?v=i9oij02oje0), [Additional](https://www.youtube.com/watch?v=D0M1h_gjA_w)
- **Academia profile**: [Kirby Urner at Academia.edu](https://princeton.academia.edu/kirbyurner)
- **Fuller Institute**: [BFI — Big Ideas: Synergetics](https://www.bfi.org/about-fuller/big-ideas/synergetics/)

### Background and community materials

- **RW Gray projects — Synergetics text**: [rwgrayprojects.com (synergetics)](http://www.rwgrayprojects.com/synergetics/s00/p0000.html)
- **Fuller FAQ**: [C. J. Fearnley’s Fuller FAQ](https://www.cjfearnley.com/fuller-faq.pdf)
- **Synergetics resource list**: [C. J. Fearnley’s resource page](https://www.cjfearnley.com/fuller-faq-2.html)
- **Wikieducator**: [Synergetics hub](https://wikieducator.org/Synergetics)
- **Quadray animation**: [Quadray.gif (Wikimedia Commons)](https://commons.wikimedia.org/wiki/File:Quadray.gif)

## Geometry and volumes (Coxeter.4D context)

- **Regular polytopes (Euclidean E⁴)**: H. S. M. Coxeter, Regular Polytopes (Dover ed.), p. 119 clarifies Euclidean 4D vs spacetime.
- **Sphere packings and lattices**: J. H. Conway & N. J. A. Sloane, [Sphere Packings, Lattices and Groups (Springer)](https://link.springer.com/book/10.1007/978-1-4757-6568-7)
- **Cayley–Menger determinant**: [Cayley–Menger determinant (reference)](https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant)
- **Tetrahedron — volume**: [Tetrahedron: volume (reference)](https://en.wikipedia.org/wiki/Tetrahedron#Volume)
- **Bareiss algorithm (exact determinants)**: [Bareiss algorithm (reference)](https://en.wikipedia.org/wiki/Bareiss_algorithm)

## Optimization and information geometry

- **Nelder–Mead method**: [Nelder–Mead (reference)](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
- **Fisher information**: [Fisher information (reference)](https://en.wikipedia.org/wiki/Fisher_information) — see also Eq. \eqref{eq:supp_fim}
- **Natural gradient**: [Natural gradient (reference)](https://en.wikipedia.org/wiki/Natural_gradient) — see Eq. \eqref{eq:supp_natgrad}

## Active Inference

- **Free energy principle**: [Free energy principle (reference)](https://en.wikipedia.org/wiki/Free_energy_principle)
- **Comprehensive review**: [Active Inference — recent review (UCL Discovery, 2023)](https://discovery.ucl.ac.uk/id/eprint/10176959/1/1-s2.0-S1571064523001094-main.pdf)

## Community discussions and context

- **Math4Wisdom**: [IVM↔XYZ conversions (curated page)](https://coda.io/@daniel-ari-friedman/math4wisdom/ivm-xyz-40)
- **synergeo (groups.io)**: [Synergetics discussion archive](https://groups.io/g/synergeo/topics)
- **GeodesicHelp**: [Geodesic computations archive (Google Groups)](https://groups.google.com/g/GeodesicHelp/)

## Related projects and applications

- **QuadCraft**: [Tetrahedral voxel engine using Quadrays](https://github.com/docxology/quadcraft/)
- **Flextegrity**: [Generating the Flextegrity Lattice (academia.edu)](https://www.academia.edu/44531954/Generating_the_Flextegrity_Lattice)

## Tooling

- **GCC libquadmath (binary128)**: [Official GCC libquadmath documentation](https://gcc.gnu.org/onlinedocs/libquadmath/index.html)

## Cross-language and cross-platform validation

- **Rust (rusty_rays)** and **Clojure (synmods)** mirror the Python algorithms for vector ops and tetravolumes, serving as independent checks on correctness and performance comparisons.
- **POV-Ray (quadcraft.py)** and **VPython (BookCovers)** demonstrate rendering pipelines for CCP/IVM scenes and educational animations.
