# 4D Namespaces: Coxeter.4D, Einstein.4D, Fuller.4D

In this section, we clarify the three internal meanings of "4D," following a dot-notation that avoids cross-domain confusion. Each namespace represents a distinct mathematical framework with specific applications in our quadray-based computational system.

## Coxeter.4D (Euclidean E⁴)

- **Definition**: Standard E⁴ with orthogonal axes and Euclidean metric; the proper setting for classical regular polytopes. As Coxeter notes (Regular Polytopes, Dover ed., p. 119), this Euclidean 4D is not spacetime. Lattice/packing discussions connect to Conway & Sloane's systematic treatment of higher-dimensional sphere packings and lattices ([Sphere Packings, Lattices and Groups (Springer)](https://link.springer.com/book/10.1007/978-1-4757-6568-7)).
- **Usage**: Embed Quadray configurations or compare alternative parameterizations when a strictly Euclidean 4D setting is desired.
- **Simplexes**: Simplex structures extend naturally to 4D and beyond (e.g., pentachora).
- **Mathematical context**: This framework is appropriate for standard Euclidean geometry, including the Cayley-Menger determinant for computing volumes from edge lengths.

## Einstein.4D (Relativistic spacetime)

- **Spacetime**: Minkowski metric signature.
- **Line element** (mostly-plus convention; see [Minkowski space](https://en.wikipedia.org/wiki/Minkowski_space)):

  \begin{equation}\label{eq:einstein_line_element}
  ds^2 = -c^2\,dt^2 + dx^2 + dy^2 + dz^2
  \end{equation}

- **Optimization analogy**: Metric-aware geodesics generalize to information geometry where the Fisher metric replaces the physical metric. See [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) and [natural gradient](https://en.wikipedia.org/wiki/Natural_gradient).
- **Important note**: This namespace is used ONLY as a metric/geodesic analogy when discussing information geometry. Physical constants G, c, Λ do not appear in Quadray lattice methods and should not be mixed with IVM unit conventions.

## Fuller.4D (Synergetics / Quadrays)

- **Basis**: Four non-negative components A,B,C,D with at least one zero post-normalization, treated as a vector (direction and magnitude), not merely a point. Overview: [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).
- **Geometry**: Tetrahedral; unit tetrahedron volume = 1; integer lattice aligns with close-packed spheres (IVM). Background: [Synergetics](https://en.wikipedia.org/wiki/Synergetics_(Fuller)).
- **Distances**: Computed via appropriate projective normalization; edges align with tetrahedral axes. The IVM = CCP = FCC shortcut allows working in 3D embeddings for visualization while preserving the underlying Fuller.4D tetrahedral accounting.
- **Implementation heritage**: Extensive computational validation through Kirby Urner's [4dsolutions ecosystem](https://github.com/4dsolutions). See the [Resources](07_resources.md) section for comprehensive details on computational implementations and educational materials.

### Directions, not dimensions (language and models)

- **Vector-first framing**: Treat Quadrays as four canonical directions ("spokes" to the vertices of a regular tetrahedron from its center), not as four orthogonal dimensions. The methane molecule (CH₄) and caltrop shape are helpful mental models.
- **Origins outside Synergetics**: Quadrays did not originate with Fuller; we adopt the coordinate system within the IVM context. See [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).
- **Language games**: Quadrays and Cartesian are parallel vector languages on the same Euclidean container; teaching them together avoids oscillating between "points now, vectors later."

### Figures

![**IVM neighbors and coordination patterns (2×2 panel layout)**. **Panel A**: The twelve nearest IVM neighbors plotted as blue points in 3D space under the default embedding, showing the positions corresponding to permutations of the Quadray integer coordinates {2,1,1,0}. These points form the vertices of a cuboctahedron (vector equilibrium) centered at the origin with uniform radial distances. **Panel B**: The same neighbor points with radial edges (light lines) connecting each neighbor to the central origin, emphasizing the spoke-like radial symmetry and equal distances from center to shell. **Panel C**: Twelve-around-one close-packed spheres configuration where each neighbor position hosts a sphere with radius chosen so neighboring spheres kiss along cuboctahedron edges, illustrating the fundamental CCP/FCC/IVM correspondence. The central gray sphere represents the "one" in Fuller's "twelve around one" motif. **Panel D**: Adjacency graph showing strut connections (solid lines) between touching neighbor spheres, revealing the cuboctahedron's edge structure, plus light radial cables to the origin representing a stylized tensegrity interpretation of the vector equilibrium geometry.](../output/figures/ivm_neighbors_edges.png)

![**Random Quadray point clouds under different embeddings (3-panel comparison)**. Each panel shows 200 randomly sampled integer Quadray coordinates with components in {0,1,2,3,4,5} projected to 3D space using different embedding matrices. **Left panel (Default embedding)**: Points (blue) under the default symmetric embedding matrix showing the natural tetrahedral-symmetric distribution of normalized Quadrays in 3D space. **Center panel (Scaled embedding, 0.75×)**: The same Quadray points (orange) under a uniformly scaled version of the default embedding, demonstrating how the point cloud structure scales proportionally while preserving relative geometries. **Right panel (Urner embedding)**: The same points (purple) projected through the canonical Urner embedding matrix, illustrating how different linear mappings from Fuller.4D to Coxeter.4D (3D slice) affect the spatial distribution while preserving the underlying discrete lattice relationships. This comparison demonstrates the flexibility in choosing embeddings for visualization and analysis while maintaining the fundamental Quadray coordinate relationships.](../output/figures/quadray_clouds.png)

In the previous figure, we show the twelve nearest IVM neighbors with coordination patterns and vector equilibrium geometry; the current figure illustrates random Quadray clouds under several embeddings.

Vector equilibrium (cuboctahedron). The shell formed by the 12 nearest IVM neighbors is the cuboctahedron, also called the vector equilibrium in synergetics. All 12 vertices are equidistant from the origin with equal edge lengths, modeling a balanced local packing. This geometry underlies the "twelve around one" close-packing motif and appears in tensegrity discussions as a canonical balanced structure. See background: [Cuboctahedron (vector equilibrium)](https://en.wikipedia.org/wiki/Cuboctahedron) and synergetics references. Computational demonstrations include related visualizations in the 4dsolutions ecosystem. See the [Resources](07_resources.md) section for comprehensive details.

### Clarifying remarks

- "A time machine is not a tesseract." [KU on synergeo](https://groups.io/g/synergeo/topic/my_take_on_close_pack/114531919) The tesseract is a Euclidean 4D object (Coxeter.4D), while Minkowski spacetime (Einstein.4D) is indefinite and not Euclidean; conflating the two leads to category errors. Fuller.4D, in turn, is a tetrahedral, mereological framing of ordinary space emphasizing shape/angle relations and IVM quantization. Each namespace carries distinct assumptions and should be used accordingly in analysis.

## Practical usage guide

- Use **Fuller.4D** when working with Quadrays, integer tetravolumes, and IVM neighbors (native lattice calculations).
- Use **Coxeter.4D** for Euclidean length-based formulas, higher-dimensional polytopes, or comparisons in E⁴ (including Cayley–Menger).
- Use **Einstein.4D** as a metric analogy when discussing geodesics or time-evolution; do not mix with synergetic unit conventions.
