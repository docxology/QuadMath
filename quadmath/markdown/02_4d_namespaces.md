# 4D Namespaces: Coxeter.4D, Einstein.4D, Fuller.4D

In this section, we clarify the three internal meanings of “4D,” following a dot-notation that avoids cross-domain confusion. First we briefly review the Coxeter.4D and Einstein.4D name spaces, which should be familiar to most readers. We then review and highlight the Fuller.4D name space, which is the focus of this manuscript.

## Coxeter.4D (Euclidean E⁴)

- **Definition**: standard E⁴ with orthogonal axes and Euclidean metric; the proper setting for classical regular polytopes. As Coxeter notes (Regular Polytopes, Dover ed., p. 119), this Euclidean 4D is not spacetime. Lattice/packing discussions connect to Conway & Sloane’s systematic treatment of higher-dimensional sphere packings and lattices ([Sphere Packings, Lattices and Groups (Springer)](https://link.springer.com/book/10.1007/978-1-4757-6568-7)).
- **Usage**: embed Quadray configurations or compare alternative parameterizations when a strictly Euclidean 4D setting is desired.
- **Simplexes**: simplex structures extend naturally to 4D and beyond (e.g., pentachora).

## Einstein.4D (Relativistic spacetime)

- **Spacetime**: Minkowski metric signature.
- **Line element** (mostly-plus convention; see [Minkowski space](https://en.wikipedia.org/wiki/Minkowski_space)):

  \begin{equation}\label{eq:einstein_line_element}
  ds^2 = -c^2\,dt^2 + dx^2 + dy^2 + dz^2
  \end{equation}

- **Optimization analogy**: metric-aware geodesics generalize to information geometry where the Fisher metric replaces the physical metric. See [Fisher information](https://en.wikipedia.org/wiki/Fisher_information) and [natural gradient](https://en.wikipedia.org/wiki/Natural_gradient).

## Fuller.4D (Synergetics / Quadrays)

- **Basis**: four non-negative components A,B,C,D with at least one zero post-normalization, treated as a vector (direction and magnitude), not merely a point. Overview: [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).
- **Geometry**: tetrahedral; unit tetrahedron volume = 1; integer lattice aligns with close-packed spheres (IVM). Background: [Synergetics](https://en.wikipedia.org/wiki/Synergetics_(Fuller)).
- **Distances**: computed via appropriate projective normalization; edges align with tetrahedral axes. The IVM = CCP = FCC shortcut allows working in 3D embeddings for visualization while preserving the underlying Fuller.4D tetrahedral accounting.
- **Implementation heritage**: Extensive computational validation through Kirby Urner's [4dsolutions ecosystem](https://github.com/4dsolutions), particularly [`qrays.py` (vector operations)](https://github.com/4dsolutions/m4w/blob/main/qrays.py) and educational materials in [School_of_Tomorrow](https://github.com/4dsolutions/School_of_Tomorrow).

### Directions, not dimensions (language and models)

- **Vector-first framing**: Treat Quadrays as four canonical directions (“spokes” to the vertices of a regular tetrahedron from its center), not as four orthogonal dimensions. The methane molecule (CH₄) and caltrop shape are helpful mental models.
- **Origins outside Synergetics**: Quadrays did not originate with Fuller; we adopt the coordinate system within the IVM context. See [Quadray coordinates](https://en.wikipedia.org/wiki/Quadray_coordinates).
- **Language games**: Quadrays and Cartesian are parallel vector languages on the same Euclidean container; teaching them together avoids oscillating between “points now, vectors later.”

### Figures

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figures/ivm_neighbors_edges.png}
\caption{IVM neighbors and radial edges (2×2 panels). A: neighbor points; B: points with radial edges from origin; C: twelve-around-one close-packed spheres (vector equilibrium); D: adjacency graph among touching neighbors.}
\label{fig:ivm_neighbors_edges}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figures/quadray_clouds.png}
\caption{Random Quadray clouds under embeddings (projections of uniformly sampled integer Quadrays through canonical maps).}
\label{fig:quadray_clouds}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figures/vector_equilibrium_panels.png}
\caption{Vector equilibrium panels. A: twelve-around-one close-packed spheres at IVM neighbor positions plus central sphere (kissing spheres). B: adjacency (struts) among touching neighbors with light radial cables to origin (stylized tensegrity).}
\label{fig:vector_equilibrium}
\end{figure}

In Figure \ref{fig:ivm_neighbors_edges}, we show the twelve nearest IVM neighbors with radial edges under a standard Urner embedding; Figure \ref{fig:quadray_clouds} illustrates random Quadray clouds under several embeddings.

Vector equilibrium (cuboctahedron). The shell formed by the 12 nearest IVM neighbors is the cuboctahedron, also called the vector equilibrium in synergetics. All 12 vertices are equidistant from the origin with equal edge lengths, modeling a balanced local packing. This geometry underlies the "twelve around one" close-packing motif and appears in tensegrity discussions as a canonical balanced structure. See background: [Cuboctahedron (vector equilibrium)](https://en.wikipedia.org/wiki/Cuboctahedron) and synergetics references. Computational demonstrations include [`ivm_neighbors.py`](https://github.com/4dsolutions/School_of_Tomorrow/blob/master/quadcraft.py) and related visualizations in the 4dsolutions ecosystem.

### Clarifying remarks

- “A time machine is not a tesseract.” The tesseract is a Euclidean 4D object (Coxeter.4D), while Minkowski spacetime (Einstein.4D) is indefinite and not Euclidean; conflating the two leads to category errors. Fuller.4D, in turn, is a tetrahedral, mereological framing of ordinary space emphasizing shape/angle relations and IVM quantization. Each namespace carries distinct assumptions and should be used accordingly in analysis.

## Practical usage guide

- Use **Fuller.4D** when working with Quadrays, integer tetravolumes, and IVM neighbors (native lattice calculations).
- Use **Coxeter.4D** for Euclidean length-based formulas, higher-dimensional polytopes, or comparisons in E⁴ (including Cayley–Menger).
- Use **Einstein.4D** as a metric analogy when discussing geodesics or time-evolution; do not mix with synergetic unit conventions.
