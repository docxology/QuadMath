# 4D Namespaces: Coxeter.4D, Einstein.4D, Fuller.4D

This section provides the definitive reference for the three 4D frameworks used throughout this manuscript. Each namespace represents a distinct mathematical framework with specific applications in our Quadray-based computational system.

## Coxeter.4D (Euclidean E⁴)

**Definition**: Four-dimensional Euclidean space E⁴ with mutually orthogonal axes and a positive-definite metric — the native setting of classical regular polytopes, and, per Coxeter (Regular Polytopes, Dover ed., p. 119), explicitly not spacetime. Lattice/packing discussions connect to Conway & Sloane's systematic treatment of higher-dimensional sphere packings and lattices ([Sphere Packings, Lattices and Groups (Springer)](https://link.springer.com/book/10.1007/978-1-4757-6568-7)).

**Usage**: Distinguish actual Euclidean four-dimensional polytopes from the rank-three Quadray image. The implemented Quadray embedding lands in $\mathbb R^3$; ordinary tetrahedral length formulas do not require an extra physical dimension.

**Simplexes**: Simplex structures extend naturally to 4D and beyond (e.g., pentachora).

**Mathematical context**: This framework is appropriate for standard Euclidean geometry, including the Cayley–Menger determinant for computing volumes from edge lengths.

## Einstein.4D (Relativistic spacetime)

**Definition**: Minkowski spacetime — three space dimensions joined to one time dimension by an indefinite metric of signature $(-,+,+,+)$ (mostly-plus convention) — the geometric arena of special relativity; the sign of the squared line element classifies separations as timelike, lightlike, or spacelike.

**Spacetime**: Minkowski metric signature.

**Line element** (mostly-plus convention; see [Minkowski space](https://en.wikipedia.org/wiki/Minkowski_space)): see Eq. \eqref{eq:minkowski_line_element} in the equations appendix.

**Optimization analogy**: Information geometry also uses a metric, but Fisher geometry is positive semidefinite, not Minkowskian. Natural-gradient descent and metric geodesics are different constructions. See [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) and [natural gradient](https://en.wikipedia.org/wiki/Natural_gradient).

**Important note**: This namespace is used ONLY as a metric/geodesic analogy when discussing information geometry. Physical constants G, c, Λ do not appear in Quadray lattice methods and should not be mixed with IVM unit conventions.

## Fuller.4D (Synergetics / Quadrays)

**Definition**: The synergetic account of space built on Quadray coordinates — four non-negative components (A, B, C, D) giving directions from the center of a reference regular tetrahedron to its vertices — together with the IVM = CCP = FCC lattice correspondence and the regular tetrahedron as the unit of volume. Exact tetravolumes are rational: integer quadray vertices give multiples of one quarter, not necessarily integers. It tracks shape and angle relations among containers, independent of time and energy.

**Representation**: Four components modulo a common offset; canonical representatives are nonnegative with minimum zero. This quotient has rank three. Overview: [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).

**Geometry**: The regular unit tetrahedron has native volume 1. Only integer classes with component sum divisible by four are FCC sphere centers; the other classes describe packing voids. See [static lattice membership](11_ivm_field_learning.md#the-ivm-lattice-in-quadray-coordinates).

**Distances**: Apply the declared embedding to differences of points. Normalization selects a representative but is not itself a metric. See [Gram identities](14_conversions_spec.md#gram-identities-and-exact-distances).

**Implementation heritage**: Kirby Urner's [4dsolutions ecosystem](https://github.com/4dsolutions) supplies related implementations and educational context; the [resource bibliography](99_resources.md) does not constitute an executed cross-project validation report.

### Directions, not dimensions (language and models)

**Vector-first framing**: Treat Quadrays as four canonical directions ("spokes" to the vertices of a regular tetrahedron from its center), not as four orthogonal dimensions. The methane molecule (CH₄) and caltrop shape are helpful mental models.

**Origins outside Synergetics**: Quadrays did not originate with Fuller; we adopt the coordinate system within the IVM context. See [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).

**Language games**: Quadrays and Cartesian are parallel vector languages on the same Euclidean container; teaching them together avoids oscillating between "points now, vectors later."

### FCC coordination and embedding comparison

![**IVM neighbors and coordination patterns (2×2 panel layout)**. **Panel A**: The twelve nearest IVM neighbors plotted as blue points in 3D space under the default embedding, showing the positions corresponding to permutations of the Quadray integer coordinates {2,1,1,0}. These points form the vertices of a cuboctahedron (vector equilibrium) centered at the origin with uniform radial distances. **Panel B**: The same neighbor points with radial edges (light lines) connecting each neighbor to the central origin, emphasizing the spoke-like radial symmetry and equal distances from center to shell. **Panel C**: Twelve-around-one close-packed spheres configuration where each neighbor position hosts a sphere with radius chosen so neighboring spheres kiss along cuboctahedron edges, illustrating the fundamental CCP/FCC/IVM correspondence. The central gray sphere represents the "one" in Fuller's "twelve around one" motif. **Panel D**: Adjacency graph showing strut connections (solid lines) between touching neighbor spheres, revealing the cuboctahedron's edge structure, plus light radial cables to the origin representing a stylized tensegrity interpretation of the vector equilibrium geometry.](figures/ivm_neighbors_edges.png)

![**Default, scaled, and equivalent Urner point clouds.** The same 200 seeded integer Quadrays are normalized and mapped into XYZ. Left: `DEFAULT_EMBEDDING`. Center: that matrix scaled by 0.75. Right: `urner_embedding(scale=1.0)`, which is entry-for-entry identical to the default, not a different geometric convention. Thus left and right are the same points; the center changes physical coordinate scale only. The samples include non-FCC residue classes, so not every point is a close-packed sphere center. Generated by `quadmath/scripts/quadray_clouds.py`; raw coordinates and matrices are saved in `quadray_clouds_data.npz`.](figures/quadray_clouds.png)

In the previous figure, we show the twelve nearest IVM neighbors with coordination patterns and vector equilibrium geometry; the current figure illustrates random Quadray clouds under several embeddings.

**Vector equilibrium (cuboctahedron)**: The shell formed by the 12 nearest IVM neighbors is the cuboctahedron, also called the vector equilibrium in synergetics. All 12 vertices are equidistant from the origin with equal edge lengths, modeling a balanced local packing. This geometry underlies the "twelve around one" close-packing motif and appears in tensegrity discussions as a canonical balanced structure. See background: [Cuboctahedron (vector equilibrium)](https://en.wikipedia.org/wiki/Cuboctahedron) and synergetics references. Computational demonstrations include related visualizations in the 4dsolutions ecosystem. See the [Resources](99_resources.md) section for comprehensive details.

### Clarifying remarks

"A time machine is not a tesseract." [KU on synergeo](https://groups.io/g/synergeo/topic/my_take_on_close_pack/114531919) The tesseract is a Euclidean 4D object (Coxeter.4D), while Minkowski spacetime (Einstein.4D) is indefinite and not Euclidean; conflating the two leads to category errors. Fuller.4D, in turn, is a tetrahedral, mereological framing of ordinary space emphasizing shape/angle relations and IVM quantization. Each namespace carries distinct assumptions and should be used accordingly in analysis.

## Practical usage guide

- Use **Fuller.4D** when working with Quadrays, exact rational tetravolumes, and IVM neighbors (native lattice calculations).
- Use **Coxeter.4D** for Euclidean length-based formulas, higher-dimensional polytopes, or comparisons in E⁴ (including Cayley–Menger).
- Use **Einstein.4D** as a metric analogy when discussing geodesics or time-evolution; do not mix with synergetic unit conventions.
