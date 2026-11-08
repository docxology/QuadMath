# Quadray Analytical Details and Methods

## Coordinate, volume, and metric contracts

This section provides detailed analytical methods for working with Quadray coordinates, including coordinate conventions, volume calculations, and optimization approaches. We emphasize the distinction between different 4D frameworks and provide practical computational methods.

## Mathematical Foundations

The methods presented here rest on three interconnected mathematical frameworks. For comprehensive definitions, see [Section 2: 4D Namespaces](02_4d_namespaces.md).

### Framework Distinctions
- **Coxeter.4D**: Euclidean 4D geometry with standard metric tensor and volume forms
- **Einstein.4D**: Minkowski spacetime with indefinite metric for information geometry analogies  
- **Fuller.4D**: Synergetics/Quadray coordinates with integer lattice constraints and IVM unit conventions

### Key Mathematical Principles
- **Rational volume quantization**: Lattice constraints quantize tetrahedral volumes to exact quarter-unit (1/4-grain) rationals in IVM units — integral for unit-tetra tilings, e.g. $\tfrac14$ for the primitive tetrahedron
- **Coordinate bridges**: The declared rank-three embedding supplies distances and scale-dependent volume conversion; it is not an isometry from four independent Euclidean axes.
- **Information geometry**: Fisher metric provides Riemannian structure for optimization on parameter manifolds
- **Exact arithmetic**: Bareiss algorithm ensures determinant calculations remain exact for integer inputs

### Implementation Strategy
The implemented numerical methods use:
- **Exact arithmetic** where possible (integer determinants, symbolic computation)
- **Numerical stability** for floating-point operations (ridge regularization, condition number monitoring)
- **Cross-validation** between different formulations (Ace 5×5 vs Cayley–Menger, native vs bridging approaches)
- **Regression checks** for documented identities, numerical ranges and boundary cases

## Framework Integration

The three 4D frameworks serve distinct but complementary roles in our implementation:

- **Euclidean geometry supplies lengths**: XYZ determinants and tetrahedral Cayley–Menger formulas operate in ordinary three-space. The Coxeter.4D namespace distinguishes higher-dimensional Euclidean theory; it does not add a dimension to the Quadray image.

- **Einstein.4D provides the analogy**: Information geometry uses the positive-semidefinite Fisher metric on parameter space, not the indefinite Minkowski metric. Natural-gradient descent is metric-aware; its trajectories are not generally geodesics.

- **Fuller.4D provides the constraints**: The Quadray coordinate system and IVM lattice impose integer constraints that enable exact arithmetic and discrete optimization. The synergetics unit conventions (regular tetrahedron volume = 1) create a quantized geometry where volumes are exact rationals on the 1/4 grid — integral for tetrahedra tiling unit IVM tetras, $\tfrac14$-grain otherwise.

This multi-framework approach allows us to:
1. Use standard Euclidean methods for volume calculations where relevant or already in use (Coxeter.4D)
2. Apply information geometry principles for metric-aware optimization (not a geodesic integrator)
3. Maintain exact arithmetic in the Isotropic Vector Matrix (IVM) setting through integer lattice constraints (Fuller.4D)
4. Bridge and swap among frameworks via coordinate transformations and unit conversions

## Fuller.4D Coordinates and Normalization

- Quadray vector q = (a,b,c,d), a,b,c,d ≥ 0, with at least one coordinate zero under normalization.
- Projective normalization adds/subtracts `(k,k,k,k)` without changing the spatial vector; choose the minimum-zero representative. The quotient has three independent spatial coordinates, not four orthogonal dimensions.
- IVM/CCP sphere centers are the integer classes whose component sum is divisible by four; other residue classes are voids. The 12 permutations of `{2,1,1,0}` form the first cuboctahedral shell.
  - Integer-coordinate models: assigning unit IVM tetravolume to the regular tetrahedron yields integer coordinates for several familiar polyhedra (inverse tetrahedron, cube, octahedron, rhombic dodecahedron, cuboctahedron) when expressed as linear combinations of the four quadray basis vectors. See overview: [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).

## Conversions and Vector Operations: Quadray ↔ Cartesian (Fuller.4D ↔ Coxeter.4D/XYZ)

- **Embedding conventions** determine the linear maps between Quadray (Fuller.4D) and Cartesian XYZ (a 3D slice or embedding aligned with Coxeter.4D conventions).
- **References**: Urner provides practical conversion write-ups and matrices; see:
  - Quadrays and XYZ: [Urner – Quadrays and XYZ](https://www.grunch.net/synergetics/quadxyz.html)
  - Introduction with examples: [Urner – Quadray intro](https://www.grunch.net/synergetics/quadintro.html)
- **Implementation**: choose a fixed tetrahedral 3×4 embedding. Adding `(k,k,k,k)` preserves the point when the embedding annihilates the all-ones vector. The `S3=sqrt(9/8)` XYZ-to-IVM volume factor assumes sphere-radius-one units (unit tetrahedron edge 2); rescale `DEFAULT_EMBEDDING` by `1/sqrt(2)` before applying that convention.
- **Vector view**: treat `q` as a vector with magnitude and direction; define dot products and norms by pushing to XYZ via `M`.

The numerical Cayley–Menger routines normalize squared-distance scale before
evaluating determinants, radii and face areas, then restore physical units.
This avoids dimensionful degeneracy thresholds and avoidable determinant
overflow/underflow. It does not make nearly degenerate floating geometry exact.

`centroid` canonicalizes each input independently and rounds component means
half-up with integer arithmetic, preserving arbitrary-size integer inputs.
It is a quantized quadray centroid, not necessarily the XYZ-nearest point to
the continuous Euclidean centroid.

Use `centroid_exact(points)` for the continuous **vertex mean**: it returns four
`Fraction` components in minimum-zero gauge, without lattice snapping or
floating conversion. This is not generally the centroid of a solid's mass.
The matching symergetics method `centroid_exact()` returns the same vertex
mean; use its `volume_centroid_exact()` for a uniform closed solid. Do not
reconstruct either rational tuple through an integer coordinate constructor.

`Fraction` may retain fixed-width NumPy numerator/denominator fields. Quadray
construction and strict inverse conversion promote these fields to unbounded
Python integers before arithmetic; wrapping an integer in `Fraction` alone
is not sufficient protection against fixed-width overflow.

```python
from fractions import Fraction
from quadmath import Quadray, centroid_exact

points = [Quadray(0, 0, 0, 0), Quadray(1, 0, 0, 0),
          Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0)]
assert centroid_exact(points) == (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(0))
```

### Exact metric, affine and oriented primitives

`dot_exact`, `squared_norm_exact` and `squared_distance_exact` retain rational
coefficients and unbounded integer labels before any floating conversion.
`affine_combination_exact(points, weights)` returns a canonical rational tuple;
weights must sum to exactly one and may be negative. `signed_tetra_volume_exact`
retains orientation, and `triangle_area_squared_exact` avoids a premature
square root. These accept quadrays or four-component rational iterables.
Finite floating coefficients mean their exact stored binary rationals, not
guessed decimal fractions. Gauge invariance requires an embedding whose rows
annihilate the common-offset direction; arbitrary custom maps need not do so.

```python
from fractions import Fraction
from quadmath import (
    Quadray, affine_combination_exact, signed_tetra_volume_exact,
    squared_distance_exact, triangle_area_squared_exact,
)

o = Quadray(0, 0, 0, 0)
a, b, c = Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0)
midpoint = affine_combination_exact([o, a], [Fraction(1, 2)] * 2)
assert midpoint == (Fraction(1, 2), Fraction(0), Fraction(0), Fraction(0))
assert squared_distance_exact(a, b) == 8
assert triangle_area_squared_exact(o, a, b) == 2
assert signed_tetra_volume_exact(o, midpoint, b, c) == Fraction(1, 8)
```

Floating dot products round completed native expressions once; norms use
scale-safe evaluation and angles use a scaled `atan2` construction. Thus large
common offsets and resolved cancellation are not discarded before geometry.

Cross-project orientation, precision boundaries and executable comparisons are
documented in the [Synergetics suite guide](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md).

### Integer-coordinate constructions (compact derivation box)

- Under the synergetics convention (unit regular tetrahedron has tetravolume 1), many familiar solids admit Quadray integer coordinates. For example, the octahedron at the same edge length has tetravolume 4, and its vertices can be formed as integer linear combinations of the four axes A,B,C,D subject to the Quadray normalization rule.
- The cuboctahedron (vector equilibrium) arises as the shell of the 12 nearest IVM neighbors given by the permutations of \((2,1,1,0)\). The rhombic dodecahedron (tetravolume 6) is the Voronoi cell of the FCC/CCP packing centered at the origin under the same embedding.
- See the following figure for a schematic summary of these relationships.

| Object | Quadray construction (sketch) | IVM volume |
| --- | --- | --- |
| Regular tetrahedron | Vertices `o=(0,0,0,0)`, `p=(2,1,0,1)`, `q=(2,1,1,0)`, `r=(2,0,1,1)` | 1 |
| Cube (tetrahedron inscribed) | XYZ vertices $(\pm1,\pm1,\pm1)$; edge 2, shorter than the unit tetra edge $2\sqrt{2}$ | 3 |
| Octahedron (tetra edge) | XYZ vertices $\pm2 e_i$; twice the tetrahedron edge midpoints | 4 |
| Rhombic dodecahedron | Voronoi cell of FCC/CCP packing at origin (dual to cuboctahedron) | 6 |
| Cuboctahedron (vector equilibrium) | Shell of the 12 nearest IVM neighbors: permutations of `(2,1,1,0)` | 20 |
| Truncated octahedron (tetra edge) | All XYZ permutations of $(0,\pm2,\pm4)$; 24 vertices, 6 square and 8 hexagonal faces | 96 |

Small coordinate examples (subset):

- Cuboctahedron neighbors (representatives): `(2,1,1,0)`, `(2,1,0,1)`, `(2,0,1,1)`, `(1,2,1,0)`; the full shell is all distinct permutations.
- Tetrahedron: `[(0,0,0,0), (2,1,0,1), (2,1,1,0), (2,0,1,1)]`.

Short scripts:

```bash
uv run --python 3.13 python quadmath/scripts/polyhedra_quadray_constructions.py
```

Programmatic check (neighbors, equal radii, adjacency):

```python
import numpy as np
from quadmath.core.examples import example_cuboctahedron_vertices_xyz

xyz = np.array(example_cuboctahedron_vertices_xyz())
r = np.linalg.norm(xyz[0])
assert np.allclose(np.linalg.norm(xyz, axis=1), r)

# A cuboctahedron's nearest-neighbor edge equals its circumradius.
touch = []
for i in range(len(xyz)):
    for j in range(i + 1, len(xyz)):
        if np.isclose(np.linalg.norm(xyz[i] - xyz[j]), r):
            touch.append((i, j))
assert len(touch) == 24
```

### Example vertex lists and volume checks (illustrative)

The following snippets show executable decompositions for standard synergetics volumes. Each tetra volume is computed via `ace_tetravolume_5x5` and summed.

Octahedron (V = 4): the octahedron with vertices $\pm 2\hat e_i$ (edge $2\sqrt2$, matching the unit IVM tetra) decomposes into eight origin-apex orthant tetras, each of volume $\tfrac12$:

```python
from quadmath.core.quadray import Quadray, ace_tetravolume_5x5

o = Quadray(0,0,0,0)
axes = [
    Quadray(1,0,0,1), Quadray(0,1,1,0),   # +x, -x
    Quadray(1,1,0,0), Quadray(0,0,1,1),   # +y, -y
    Quadray(1,0,1,0), Quadray(0,1,0,1),   # +z, -z
]
V_oct = sum(
    ace_tetravolume_5x5(o, x, y, z)
    for x in axes[0:2] for y in axes[2:4] for z in axes[4:6]
)  # 8 * 1/2 = 4
```

Cube (V = 3): the cube with XYZ vertices $(\pm1,\pm1,\pm1)$ (edge 2) decomposes into the inscribed tetra on four alternating cube vertices (V = 1) plus four corner tetras (V = $\tfrac12$ each):

```python
from quadmath.core.quadray import Quadray, ace_tetravolume_5x5

inner = ace_tetravolume_5x5(
    Quadray(1,0,0,0), Quadray(0,1,0,0), Quadray(0,0,1,0), Quadray(0,0,0,1),
)  # 1
corner = ace_tetravolume_5x5(
    Quadray(0,1,1,1), Quadray(0,0,0,1), Quadray(0,1,0,0), Quadray(0,0,1,0),
)  # 1/2
V_cube = inner + 4 * corner  # 1 + 4*(1/2) = 3
```

Notes.

- The octahedron and cube snippets specify exact decompositions using `ace_tetravolume_5x5`; they are examples to reproduce, not a new verification receipt.
- Volumes are invariant to adding \((k,k,k,k)\) to each vertex of a tetra (projective normalization), which the 5×5 determinant respects.

## Quarter-Unit Rational Volume Quantization {#sec:integer_volume}

For a tetrahedron with vertices P₀..P₃ in the Quadray integer lattice (Fuller.4D), see Eq. \eqref{eq:lattice_det} in the equations appendix.

- With integer quadray coordinates the projected determinant is an exact integer, and the IVM tetravolume is the exact rational $|{\det}|/4$ (see Eq. \eqref{eq:gdj}), implemented with `fractions.Fraction` in `integer_tetra_volume` and `ace_tetravolume_5x5`. Volumes are integral for tetrahedra that tile into unit IVM tetras (determinant divisible by 4) and fractional otherwise: the primitive tetrahedron spanned by $(0,0,0,0)$, $(1,0,0,0)$, $(0,1,0,0)$, $(0,0,1,0)$ has determinant 1 and exact volume $\tfrac{1}{4}$.
- Unit convention: the particular regular tetrahedron listed above (origin and three mutually adjacent IVM neighbors) has determinant 4 and volume 1. Arbitrary triples of neighbor moves need not form this tetrahedron.

Notes.

- $P_0,\ldots,P_3$ are tetrahedron vertices in Quadray coordinates.
- $V_{xyz}$ in Eq. \eqref{eq:lattice_det} is the Euclidean (XYZ) volume; converting it to IVM tetra-units requires the synergetics scale factor $S3=\sqrt{9/8}$ in sphere-radius units (Eq. \eqref{eq:xyz_det}). For lattice tetrahedra, the direct quadray formula Eq. \eqref{eq:gdj} yields the IVM volume exactly, without unit conversion.
- Background and variations are discussed under Tetrahedron volume formulas: [Tetrahedron – volume](https://en.wikipedia.org/wiki/Tetrahedron#Volume).

Tom Ace 5×5 determinant (tetravolume directly from quadrays), see Eq. \eqref{eq:ace5x5} in the equations appendix.

This returns the same exact tetravolumes as `integer_tetra_volume` for every lattice tetrahedron: $|\det_{5\times5}|$ always equals the magnitude of the projected $3\times3$ determinant, so both implementations agree exactly (both return `fractions.Fraction` values $|\det|/4$).

Notes.

- Rows correspond to the Quadray 4-tuples of the four vertices with a final affine column of ones; the last row enforces projective normalization.
- The factor $\tfrac{1}{4}$ returns tetravolumes in IVM units consistent with synergetics. See also [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).

Equivalently, define the 5×5 matrix of quadray coordinates augmented with an affine 1 as shown in Eq. \eqref{eq:ace5x5_expanded} in the equations appendix.

Points vs vectors: subtracting points is shorthand for forming edge vectors. We treat quadray 4-tuples as vectors from the origin; differences like $(P_1-P_0)$ mean "edge vectors," avoiding ambiguity between "points" and "vectors."

Equivalently via Cayley–Menger determinant (Coxeter.4D/Euclidean lengths) ([Cayley–Menger determinant](https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant)), see Eq. \eqref{eq:cayley_menger} in the equations appendix.

References: [Cayley–Menger determinant](https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant), lattice tetrahedra discussions in geometry texts; see also [Tetrahedron – volume](https://en.wikipedia.org/wiki/Tetrahedron#Volume). Code: `integer_tetra_volume`, `ace_tetravolume_5x5`.

Notes.

- **Pairwise distances**: $d_{ij}$ are Euclidean distances between vertices $P_i$ and $P_j$.
- **Length-only formulation**: Cayley–Menger provides a length-only formula for simplex volumes, here specialized to tetrahedra; see the canonical reference above.

Table: Native placements in the closed-mesh figure; edge lengths are not all equal. {#tbl:polyhedra_volumes}

| Polyhedron | Scale-one XYZ edge squared | Volume (IVM tetra-units) |
| --- | --- | --- |
| Regular tetrahedron | 8 | 1 |
| Cube with tetrahedron inscribed | 4 | 3 |
| Octahedron | 8 | 4 |
| Rhombic dodecahedron (FCC Voronoi cell) | 3 | 6 |
| Cuboctahedron (vector equilibrium) | 8 | 20 |
| Truncated octahedron | 8 | 96 |

## Distances and Metrics

Distance definitions depend on the chosen embedding and normalization. For cross-references to information geometry, see [Eq. (FIM)](08_equations_appendix.md#eq:fim) and [natural gradient](08_equations_appendix.md#eq:natgrad) in the Equations appendix.

## XYZ determinant and S3 conversion {#sec:xyz_conversion}

Given XYZ coordinates of tetrahedron vertices (x_i, y_i, z_i), the Euclidean volume is computed as shown in Eq. \eqref{eq:xyz_det} in the equations appendix.

The conversion depends on the coordinate scale: `DEFAULT_EMBEDDING` has unit tetrahedron XYZ volume \(8/3\), so \(V_{\rm ivm}=(3/8)V_{\rm xyz}\). For a uniformly scaled Urner matrix \(M(c)\), the factor is \(3/(8|c|^3)\), \(c\ne0\). Only at \(c=1/\sqrt2\), with tetrahedron edge 2, does this become \(S3=\sqrt{9/8}\). Length-based formulas must use the same declared units.

### Alternative Volume Formulas

- **Piero della Francesca (PdF) formula**, which consumes edge lengths and returns Euclidean volumes. See Eq. \eqref{eq:pdf} in the equations appendix.

  Convert using \(3/8\) for default scale-one XYZ, or \(S3=\sqrt{9/8}\) for sphere-radius-one XYZ. Do not apply the latter to unrescaled default coordinates.

- **Gerald de Jong (GdJ) formula**, which natively returns tetravolumes. See Eq. \eqref{eq:gdj} in the equations appendix.

  In Quadray coordinates, one convenient native form uses edge-vector differences and an integer-preserving determinant (agreeing with Ace 5×5):

  where each column is formed from Quadray component differences of $P_1-P_0$, $P_2-P_0$, $P_3-P_0$ projected via $\pi(P) = (a - d,\, b - d,\, c - d)$ to the synergetics 3D slice (matching `integer_tetra_volume`); integer arithmetic is exact and the factor $\tfrac{1}{4}$ produces IVM tetravolumes. See de Jong's Quadray notes and Urner's implementations for derivations ([Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates)).

- Euclidean embedding distance via appropriate linear map from quadray to R³.
- Information geometry metric: Fisher Information Matrix (FIM)
  - $\mathrm{FIM}[i,j] = \mathbb{E}\big[\, \partial_{\theta_i} \log p(x;\theta)\,\partial_{\theta_j} \log p(x;\theta)\,\big]$
  - It is a Riemannian metric on identifiable parameter directions; redundant directions can make it singular. Natural-gradient increments solve a damped system.

## Fisher pullbacks, rank, and probability support {#fisher-geometry-in-quadray-space}

- A Cartesian score metric pulls back through the $3\times4$ embedding to a $4\times4$ matrix with the common-offset direction in its kernel. Tetrahedral coordinates alone do not imply a block-diagonal Fisher matrix.
- Eigenvalues describe score sensitivity; a loss-gradient Gram matrix is not automatically the Hessian of a loss.

The empirical estimator widens integral scores before products. Ordinary
batches use the floating Gram fast path; overflow-risk normalized batches use
exact accumulation before conversion, preserving unrelated small curvature.
Safe symmetrization retains finite large entries and representable subnormals.
Quadray Fisher information is estimated from the transformed scores, avoiding
an unnecessarily overflowing intermediate Cartesian pullback. These remain
floating outputs, not exact positive-semidefinite certificates; exceptional
exact accumulation costs more than the ordinary matrix-product path.

### Probability normalization and support

Entropy, divergence and information distances accept finite nonnegative
weights with positive total mass, including joint arrays. Scaling all weights
by a common positive factor does not change their normalized distribution.
Negative, complex, nonfinite, empty and zero-mass inputs are undefined and
raise `ValueError`.

Normalization retains log probabilities when positive weight ratios fall below
the floating probability range. KL therefore does not invent missing support,
and entropy, mutual information and likelihood expectations retain terms whose
final contribution is still representable. True zero support remains distinct
from numerical underflow.

`eps` is a finite nonnegative pseudocount added to each **normalized**
probability, followed by renormalization. This replaces inconsistent
logarithm/overlap offsets; the default `1e-15` remains negligible for ordinary
inputs. With `eps=0`, zero-probability entropy terms contribute zero, KL is
infinite when the reference lacks posterior support, and disjoint distributions
have Jensen--Shannon divergence `log(2)` and Fisher--Rao distance `pi`.
Mutual information smooths the joint distribution once before deriving its
marginals. Free energy ignores zero-posterior likelihood terms, but retains
infinite or undefined likelihood contributions on positive support; arbitrary
supplied log likelihoods do not guarantee a nonnegative free energy.

## Native volume recipes and scale-aware comparisons {#practical-methods}

### Tetravolumes with Quadrays {#sec:tetravolumes_quadrays}

Native `integer_tetra_volume` and `ace_tetravolume_5x5` return exact rational
IVM volumes. The signed and rational-affine primitives above preserve
orientation and fractional locations. PdF and Cayley–Menger instead consume
Euclidean lengths; their output must use the scale-specific bridge in
[XYZ conversion](#sec:xyz_conversion).

The following native-versus-length comparison uses sphere-radius-one lengths
before multiplying by \(S3\). Default scale-one lengths require the \(3/8\)
factor instead.

![**Native versus length-based tetrahedral volumes.** Exact Ace determinants agree with symbolic Cayley–Menger calculations after converting default squared distances to sphere-radius-one units before applying S3. Cases include the unit tetrahedron (1), doubled edges (8), mixed coordinates, the centered example (2) and a larger mixed tetrahedron. Values are native IVM rationals, not necessarily integers; the numerical data are exported alongside the figure.](../output/figures/bridging_vs_native.png)

![**Volume scaling in sphere-radius-one XYZ units.** Euclidean regular-tetrahedron volume is \(\sqrt2 L^3/12\); native IVM volume in this convention is \(L^3/8\). Their ratio is \(S3=\sqrt{9/8}\). Both describe the same Euclidean geometry with different volume units; default scale-one Urner coordinates instead use the \(3/8\) bridge.](../output/figures/volumes_scale_plot.png)

![**Closed polyhedra in shared scale-one XYZ units.** Six panels use identical axes, camera and unit aspect. Face cycles come from supporting planes; native tetrahedral decompositions measure the displayed volumes: 1, 3, 4, 6, 20 and 96. Edge-squared labels distinguish the cube and FCC Voronoi cell from the four solids with tetrahedron-length edges. The truncated octahedron has all 24 vertices and 14 faces; no independent display rescaling or guessed volume network is used.](../output/figures/polyhedra_quadray_constructions.png)

### Exact and length-based Python volume recipes {#short-python-snippets}

```python
from quadmath.core.quadray import Quadray, ace_tetravolume_5x5

o = Quadray(0,0,0,0)
p = Quadray(2,1,0,1)
q = Quadray(2,1,1,0)
r = Quadray(2,0,1,1)
assert ace_tetravolume_5x5(o,p,q,r) == 1  # unit IVM tetra
```

```python
import numpy as np
from quadmath.core.cayley_menger import ivm_tetra_volume_cayley_menger

# Edge length 1 in sphere-radius-one XYZ units, not default Urner units.
d2 = np.ones((4,4)) - np.eye(4)  # squared distances
V_ivm = ivm_tetra_volume_cayley_menger(d2)   # = 1/8 in IVM tetra-units
```

```python
# SymPy implementation of Tom Ace 5x5 (symbolic determinant)
from sympy import Matrix

def qvolume(q0, q1, q2, q3):
    M = Matrix([
        q0 + (1,),
        q1 + (1,),
        q2 + (1,),
        q3 + (1,),
        [1, 1, 1, 1, 0],
    ])
    return abs(M.det()) / 4
```

```python
# Symbolic variant with SymPy (exact radicals)
from sympy import Matrix, sqrt, simplify
from quadmath.core.symbolic import cayley_menger_volume_symbolic, convert_xyz_volume_to_ivm_symbolic

d2 = Matrix([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
V_xyz_sym = cayley_menger_volume_symbolic(d2)      # sqrt(2)/12
V_ivm_sym = simplify(convert_xyz_volume_to_ivm_symbolic(V_xyz_sym))  # 1/8
```

### Random tetrahedra in the IVM (integer volumes)

- The 12 CCP directions are the permutations of \((2,1,1,0)\). Random walks on this move set generate integer-coordinate Quadrays; resulting tetrahedra have integer tetravolumes.

```python
from random import Random
from quadmath.core.quadray import IVM_NEIGHBOR_MOVES, Quadray, ace_tetravolume_5x5

rng = Random(7)
moves = [Quadray(*move) for move in IVM_NEIGHBOR_MOVES]

def random_walk(start: Quadray, steps: int) -> Quadray:
    cur = start
    for _ in range(steps):
        m = rng.choice(moves)
        cur = Quadray(cur.a+m.a, cur.b+m.b, cur.c+m.c, cur.d+m.d)
    return cur

A = random_walk(Quadray(0,0,0,0), 1000)
B = random_walk(Quadray(0,0,0,0), 1000)
C = random_walk(Quadray(0,0,0,0), 1000)
D = random_walk(Quadray(0,0,0,0), 1000)
V = ace_tetravolume_5x5(A,B,C,D)            # exact IVM volume as a Fraction
```

### Algebraic precision

- Determinants via floating-point introduce rounding noise. Bareiss and symbolic arithmetic preserve exact values. Random walks on FCC moves stay in one FCC coset, so their tetrahedral volumes are integral `Fraction` values. Arbitrary integer quadray vertices can include other cosets and need only have quarter-unit volumes.
- When computing via XYZ determinants, high-precision floats (e.g., `gmpy2.mpfr`) or symbolic matrices avoid vestigial errors; round at the end if the underlying result is known to be integral.

### XYZ determinant and the S3 conversion

- Use Eq. \eqref{eq:xyz_det} with the declared scale: native factor $3/8$ for default XYZ, or S3 only for sphere-radius-one lengths.

### D^3 vs R^3: 60° “closing the lid” vs orthogonal “cubing”

- **60-degree corner heuristic**: Three edges of lengths $A,B,C$ meeting pairwise at 60 degrees enclose a tetrahedron of Euclidean volume $ABC/(6\sqrt2)$. Dividing by the regular reference tetrahedron's Euclidean volume $L_0^3/(6\sqrt2)$ gives native volume $ABC/L_0^3$. Thus the product rule holds for edge ratios measured against $L_0$, not for arbitrary unscaled XYZ lengths or for the four origin-to-vertex basis rays.
- **Pedagogical note**: Adopt a vector-first approach. Differences like $(P_i-P_0)$ denote edge vectors; Quadrays and Cartesian can be taught in parallel as vector languages on the same Euclidean container.

Reference notebook with worked examples and code: See the [Resources](07_resources.md) section for comprehensive educational materials and computational implementations.

See implementation: `tetra_volume_cayley_menger`.

- Lattice snapping is implementation-specific: `quadray_from_xyz` uses half-up component rounding with coherent recovery of resolvable lattice images. It is not generally XYZ-nearest; see [canonical inversion and snapping](14_conversions_spec.md#exact-canonical-inversion).

## Reproducing native and numerical calculations {#practical-implementation-workflow}

1. Declare the representation, embedding, and length units before comparing formulas.
2. Use native integer determinants for exact rational tetravolumes; use the scale-specific XYZ bridge for length-based formulas.
3. Distinguish score-based Fisher information from loss-gradient Gram surrogates. Damping is numerical regularization, not a proof of geometric invariance.
4. Run examples in the supported Python 3.13 environment and route generated artifacts with `QUADMATH_OUTPUT_DIR` when preserving a working output tree.
5. Use the [learning evaluation module](15_learning_evaluation.md) for held-out predictions and the [benchmark module](17_benchmarks_statistics.md) for measurements. A fixed workload seed does not make timing deterministic.

The source index below preserves explicit code anchors. Formulas discussed as proposals in [extensions](05_extensions.md) are not implemented promises.

## Source-linked mathematical API index {#code-methods-anchors}

### Core Quadray operations {#code:core_quadray}

#### `Quadray` {#code:Quadray}

Source: `src/quadmath/core/quadray.py` — raw integer four-component vector class; `normalize()` selects its minimum-zero physical representative.

#### `DEFAULT_EMBEDDING` {#code:DEFAULT_EMBEDDING}

Source: `src/quadmath/core/quadray.py` — canonical 3×4 symmetric embedding matrix for Quadray to XYZ conversion.

#### `to_xyz` {#code:to_xyz}

Source: `src/quadmath/core/quadray.py` — map quadray to R³ via a 3×4 embedding matrix (Fuller.4D → Coxeter.4D slice).

#### `magnitude` {#code:magnitude}

Source: `src/quadmath/core/quadray.py` — return the Euclidean magnitude of `q` under the given embedding (vector norm).

#### `dot` {#code:dot}

Source: `src/quadmath/core/quadray.py` — return Euclidean dot product <q1,q2> under the given embedding.

#### `centroid_exact` {#code:centroid_exact}

Source: `src/quadmath/core/quadray.py` — exact continuous centroid as a canonical four-`Fraction` tuple; distinct from the half-up snapped `centroid`.

#### Exact metric and affine primitives

Source: `src/quadmath/core/quadray.py` — `dot_exact`, `squared_norm_exact`,
`squared_distance_exact`, `affine_combination_exact`,
`signed_tetra_volume_exact` and `triangle_area_squared_exact` retain native
rational arithmetic. See the executable examples above and the generated API
[glossary](10_symbols_glossary.md).

### Quaternion operations {#code:quaternion_operations}

#### `qmul` {#code:qmul}

Source: `src/quadmath/core/quadray.py` — Hamilton product of two quaternions in scalar-first $(w, x, y, z)$ order ($q = w + x\,i + y\,j + z\,k$), an independent representation from the Quadray lattice components $(a, b, c, d)$ and from the XYZ triples produced by `to_xyz`.

#### `qconjugate` {#code:qconjugate}

Source: `src/quadmath/core/quadray.py` — conjugate $(w, -x, -y, -z)$ of a quaternion; for a unit quaternion the conjugate is the multiplicative inverse, so `q * qconjugate(q)` equals $(1, 0, 0, 0)$ up to floating-point error.

#### `qrotate` {#code:qrotate}

Source: `src/quadmath/core/quadray.py` — Rodrigues rotation of a 3-vector by a unit quaternion via $v' = q\,(0, v)\,q^*$, composed with `qmul` and `qconjugate`; `q` must be unit (within 1e-9) and the rotation magnitude it encodes, $2\,\mathrm{atan2}(\lVert (x, y, z) \rVert, w)$, must match $|\mathrm{angle}|$ modulo $2\pi$ within 1e-9, so a mismatched (q, angle) pair raises instead of silently rotating by the wrong amount.

#### `slerp` {#code:slerp}

Source: `src/quadmath/core/quadray.py` — shortest-arc spherical linear interpolation between unit quaternions: when $\langle q_a, q_b\rangle < 0$ the second quaternion is negated first ($q$ and $-q$ encode the same rotation) so the path always takes the shorter arc on the rotation sphere; nearly parallel inputs (including exactly antipodal pairs after sign alignment) fall back to normalized lerp to avoid dividing by $\sin\theta \approx 0$, and the endpoints are exact. The site path traced by these interpolations is visualized by `plot_slerp_path` in `src/quadmath/viz/plots.py` (figure below).

#### `rotate_about_axis` {#code:rotate_about_axis}

Source: `src/quadmath/core/quadray.py` — axis-angle convenience wrapper: builds the unit quaternion $q = (\cos(\theta/2),\, \sin(\theta/2)\,\hat{u})$ from the normalized rotation axis (any non-zero scale accepted; the zero vector is rejected) and delegates to `qrotate`, reducing the angle to $(-\pi, \pi]$ first so rotation is right-handed about the axis and `qrotate`'s encoded-angle validation stays exact.

The property checks in `src/quadmath/validate/validate.py` — normalization, conjugate-inverse, double cover, slerp midpoint, and associativity — consume these operations over quaternion sequences via `run_validation`, which collects the resulting deterministic validation reports.

![**Shortest-arc quaternion interpolation and its rotated site path.** A unit quaternion interpolates from identity to a right-handed $\pi/2$ rotation about Z over sixteen samples. The displayed site starts at $(2,2,2)$ and follows a quarter-circle of radius $2\sqrt2$ at height $z=2$. Slerp is a shortest arc in unit-quaternion rotation geometry; the displayed Euclidean site path is not a straight-line XYZ geodesic or a natural-gradient trajectory. Generated by `uv run --python 3.13 python quadmath/scripts/quaternion_gallery.py`.](../output/figures/quaternion_slerp_path.png)

### Quaternion metrics {#code:quaternion_metrics}

#### `angle_error` {#code:angle_error}

Source: `src/quadmath/core/metrics.py` — geodesic rotation angle $2\arccos(|\langle q_1, q_2\rangle|)$ between two quaternions, in radians and confined to $[0, \pi]$ by clamping the acos argument; the metric is symmetric in its arguments and invariant under $q \to -q$ (the quaternion double cover), with inputs normalized internally so non-unit but non-zero quaternions are accepted.

#### `quat_log_euclidean_dispersion` {#code:quat_log_euclidean_dispersion}

Source: `src/quadmath/core/metrics.py` — root-mean-square chordal dispersion of a quaternion set about its normalized component-wise mean, $\sqrt{\mathrm{mean}_i\,\|q_i - m\|^2}$ in $\mathbb{R}^4$, with signs aligned to the first quaternion before averaging because $q$ and $-q$ encode the same rotation; returns 0 for a single quaternion and raises if the sign-aligned mean vanishes.

### Volume calculations {#code:volume_calculations}

#### `integer_tetra_volume` {#code:integer_tetra_volume}

Source: `src/quadmath/core/quadray.py` — exact projected $3\times3$ determinant over 4, i.e. the IVM tetravolume $|\det|/4$ as a `fractions.Fraction`.

#### `ace_tetravolume_5x5` {#code:ace_tetravolume_5x5}

Source: `src/quadmath/core/quadray.py` — Tom Ace 5×5 determinant in IVM units.

#### `tetra_volume_cayley_menger` {#code:tetra_volume_cayley_menger}

Source: `src/quadmath/core/cayley_menger.py` — length-based formula (XYZ units).

#### `ivm_tetra_volume_cayley_menger` {#code:ivm_tetra_volume_cayley_menger}

Source: `src/quadmath/core/cayley_menger.py` — Cayley–Menger volume converted to IVM units.

### Coordinate conversions {#code:coordinate_conversions}

#### `urner_embedding` {#code:urner_embedding}

Source: `src/quadmath/lattice/conversions.py` — canonical XYZ embedding.

#### `quadray_to_xyz` {#code:quadray_to_xyz}

Source: `src/quadmath/lattice/conversions.py` — apply embedding matrix to map Quadray to XYZ.

### Linear algebra utilities {#code:linear_algebra}

#### `bareiss_determinant_int` {#code:bareiss_determinant_int}

Source: `src/quadmath/core/linalg_utils.py` — exact integer Bareiss determinant.

### Optimization methods {#code:optimization}

#### `nelder_mead_quadray` {#code:nelder_mead_quadray}

Source: `src/quadmath/optimize/nelder_mead_quadray.py` — Nelder–Mead optimization adapted to the integer quadray lattice.

#### `discrete_ivm_descent` {#code:discrete_ivm_descent}

Source: `src/quadmath/optimize/discrete_variational.py` — greedy descent over integer FCC locations using canonical neighbor moves; objective values may be real. Returns a `DiscretePath` with visited Quadrays and objective values.

#### `neighbor_moves_ivm` {#code:neighbor_moves_ivm}

Source: `src/quadmath/optimize/discrete_variational.py` — return the 12 canonical IVM neighbor moves as Quadray deltas.

#### `apply_move` {#code:apply_move}

Source: `src/quadmath/optimize/discrete_variational.py` — apply a lattice move and normalize to the canonical representative.

### Information geometry methods {#code:information_geometry}

#### `fisher_information_matrix` {#code:fisher_information_matrix}

Source: `src/quadmath/inference/information.py` — empirical outer-product estimator.

#### `fisher_information_quadray` {#code:fisher_information_quadray}

Source: `src/quadmath/inference/information.py` — compute Fisher information matrix in both Cartesian and Quadray coordinates.

#### `natural_gradient_step` {#code:natural_gradient_step}

Source: `src/quadmath/inference/information.py` — damped inverse-Fisher step.

#### `free_energy` {#code:free_energy}

Source: `src/quadmath/inference/information.py` — discrete-state variational free energy.

#### `expected_free_energy` {#code:expected_free_energy}

Source: `src/quadmath/inference/information.py` — the implemented toy objective is `free_energy(...) - log_p_o`; it is not a complete policy-conditioned expected-free-energy solver.

#### `active_inference_step` {#code:active_inference_step}

Source: `src/quadmath/inference/information.py` — joint perception-action update step in Active Inference.

#### `information_geometric_distance` {#code:information_geometric_distance}

Source: `src/quadmath/inference/information.py` — evaluate $\sqrt{\max(0,\Delta^\mathsf{T}F\Delta)}$ for one supplied, fixed matrix. For a positive-definite constant metric this is its distance; it does not integrate a position-dependent Fisher geodesic.

#### `perception_update` {#code:perception_update}

Source: `src/quadmath/inference/information.py` — return the scaled increment $\eta(D\mu-\partial_\mu F)$ using a finite-difference gradient; the caller adds it to the state.

#### `action_update` {#code:action_update}

Source: `src/quadmath/inference/information.py` — return the scaled increment $-\eta\,\partial_a F$; the caller performs the state update.

#### `finite_difference_gradient` {#code:finite_difference_gradient}

Source: `src/quadmath/inference/information.py` — compute numerical gradient of a scalar function via central differences.

### Metrics and analysis {#code:metrics}

#### `shannon_entropy` {#code:shannon_entropy}

Source: `src/quadmath/core/metrics.py` — Shannon entropy H(p) for a discrete distribution.

#### `information_length` {#code:information_length}

Source: `src/quadmath/core/metrics.py` — path length in information space via gradient-weighted arc length.

#### `fim_eigenspectrum` {#code:fim_eigenspectrum}

Source: `src/quadmath/core/metrics.py` — eigen-decomposition of a Fisher information matrix.

#### `fisher_condition_number` {#code:fisher_condition_number}

Source: `src/quadmath/core/metrics.py` — compute the condition number of the Fisher information matrix.

#### `fisher_curvature_analysis` {#code:fisher_curvature_analysis}

Source: `src/quadmath/core/metrics.py` — comprehensive analysis of Fisher information matrix curvature.

#### `fisher_quadray_comparison` {#code:fisher_quadray_comparison}

Source: `src/quadmath/core/metrics.py` — compare Fisher information matrices between coordinate systems.

### Examples and utilities {#code:examples}

#### `example_ivm_neighbors` {#code:example_ivm_neighbors}

Source: `src/quadmath/core/examples.py` — return the 12 nearest IVM neighbors as permutations of {2,1,1,0} (neighbor-move-set role).

#### `example_cuboctahedron_neighbors` {#code:example_cuboctahedron_neighbors}

Source: `src/quadmath/core/examples.py` — return twelve-around-one IVM neighbors (vector-equilibrium-shell role). Same set as `example_ivm_neighbors`; both public names are documented API sharing one sorted, deterministic implementation.

#### `example_cuboctahedron_vertices_xyz` {#code:example_cuboctahedron_vertices_xyz}

Source: `src/quadmath/core/examples.py` — return XYZ coordinates for the twelve-around-one neighbors.

#### `example_partition_tetra_volume` {#code:example_partition_tetra_volume}

Source: `src/quadmath/core/examples.py` — construct a tetrahedron from the four-fold partition and return the exact IVM tetravolume (`fractions.Fraction`).

### Symbolic computation {#code:symbolic}

#### `cayley_menger_volume_symbolic` {#code:cayley_menger_volume_symbolic}

Source: `src/quadmath/core/symbolic.py` — return symbolic Euclidean tetrahedron volume from squared distances.

#### `convert_xyz_volume_to_ivm_symbolic` {#code:convert_xyz_volume_to_ivm_symbolic}

Source: `src/quadmath/core/symbolic.py` — convert a symbolic Euclidean volume to IVM tetravolume via S3.

### Visualization and animation {#code:visualization}

#### `animate_discrete_path` {#code:animate_discrete_path}

Source: `src/quadmath/viz/visualize.py` — animate a `DiscretePath` to MP4; saves CSV/NPZ trajectory to `quadmath/output/`.

#### `plot_ivm_neighbors` {#code:plot_ivm_neighbors}

Source: `src/quadmath/viz/visualize.py` — scatter the 12 IVM neighbor points in 3D.

#### `plot_partition_tetrahedron` {#code:plot_partition_tetrahedron}

Source: `src/quadmath/viz/visualize.py` — plot the four-fold partition as a labeled tetrahedron in 3D.

#### `animate_simplex` {#code:animate_simplex}

Source: `src/quadmath/viz/visualize.py` — animate simplex evolution across iterations.

#### `plot_simplex_trace` {#code:plot_simplex_trace}

Source: `src/quadmath/viz/visualize.py` — plot per-iteration diagnostics for Nelder–Mead.

### Path and file utilities {#code:utilities}

#### `get_repo_root` {#code:get_repo_root}

Source: `src/quadmath/paths.py` — heuristically find repository root by walking up from start.

#### `get_output_dir` {#code:get_output_dir}

Source: `src/quadmath/paths.py` — return `quadmath/output` path at the repo root and ensure it exists.

#### `get_data_dir` {#code:get_data_dir}

Source: `src/quadmath/paths.py` — return `quadmath/output/data` path and ensure it exists.

#### `get_figure_dir` {#code:get_figure_dir}

Source: `src/quadmath/paths.py` — return `quadmath/output/figures` path and ensure it exists.

### Additional Nelder–Mead components {#code:nelder_mead_components}

#### `SimplexState` {#code:SimplexState}

Source: `src/quadmath/optimize/nelder_mead_quadray.py` — optimization trajectory state containing vertices, values, volume, and history.

#### `order_simplex` {#code:order_simplex}

Source: `src/quadmath/optimize/nelder_mead_quadray.py` — sort vertices by objective value ascending and return paired lists.

#### `centroid_excluding` {#code:centroid_excluding}

Source: `src/quadmath/optimize/nelder_mead_quadray.py` — integer centroid of three vertices, excluding the specified index.

#### `project_to_lattice` {#code:project_to_lattice}

Source: `src/quadmath/optimize/nelder_mead_quadray.py` — project a quadray to the canonical lattice representative via normalize.

#### `compute_volume` {#code:compute_volume}

Source: `src/quadmath/optimize/nelder_mead_quadray.py` — exact IVM tetra-volume (`fractions.Fraction`, $|\det|/4$) from the first four vertices.

### Discrete variational components {#code:discrete_variational}

#### `DiscretePath` {#code:DiscretePath}

Source: `src/quadmath/optimize/discrete_variational.py` — optimization trajectory on the integer quadray lattice.

### Glossary generation {#code:glossary}

#### `build_api_index` {#code:build_api_index}

Source: `src/quadmath/tools/glossary_gen.py` — build API index from source directory.

#### `generate_markdown_table` {#code:generate_markdown_table}

Source: `src/quadmath/tools/glossary_gen.py` — generate markdown table from API entries.

#### `inject_between_markers` {#code:inject_between_markers}

Source: `src/quadmath/tools/glossary_gen.py` — inject payload between markers in markdown text.

### Geometry utilities {#code:geometry}

#### `minkowski_interval` {#code:minkowski_interval}

Source: `src/quadmath/core/geometry.py` — return the Minkowski interval squared ds² (Einstein.4D).

Relevant tests (`tests/`):

- `test_quadray.py` (unit IVM tetra, primitive tetra = 1/4, exact scaled volume, Ace vs. integer agreement)
- `test_quadray_cov.py` (Ace determinant basic check)
- `test_cayley_menger.py` (regular tetra volume in XYZ units)
- `test_linalg_utils.py` (Bareiss determinant behavior)
- `test_examples.py`, `test_examples_cov.py` (neighbors, examples)
- `test_metrics.py`, `test_metrics_cov.py`, `test_information.py`, `test_paths.py`, `test_paths_cov.py`

### Regression coverage and evidence boundaries

The numerical regressions under `tests/unit/` cover coordinate operations, rational determinants, optimization trajectories, information metrics, plotting, and symbolic comparisons. Their existence is not a current pass or coverage result. Consult [the operational workflow](../../WORKFLOW.md) for the integrated verification procedure.

## Reproducibility checklist

- Keep exact integer/rational and floating calculations distinct; symbolic optional dependencies must be installed for symbolic examples.
- Compare native volumes with length formulas only after declaring the embedding scale.
- Record seeds, parameters, dependency versions, and output locations with numerical results.
- Treat simplex volume as a diagnostic, not a monotone objective or a convergence certificate.
- Preserve generated evidence as a dated result. Re-running a script may change outputs when its implementation or scientific environment changes.
- Consult [the conversion specification](../../SPEC.md) for integer quotient and formalization scope; it does not verify floating simulation or application-level claims.
