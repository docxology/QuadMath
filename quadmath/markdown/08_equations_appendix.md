# Equations and Math Supplement (Appendix)

## Volume of a Tetrahedron (Lattice)

\begin{equation}\label{eq:lattice_det}
V = \tfrac{1}{6}\,\left|\det\,[\,P_1 - P_0,\; P_2 - P_0,\; P_3 - P_0\,]\right|
\end{equation}

Notes.

- $P_0,\ldots,P_3$ are vertex coordinates; the determinant computes the volume of the parallelepiped spanned by edge vectors, with the $1/6$ factor converting to tetra volume.

Tom Ace 5×5 tetravolume (IVM units):

\begin{equation}\label{eq:ace5x5}
V_{ivm} = \tfrac{1}{4} \left| \det \begin{pmatrix}
 a_0 & a_1 & a_2 & a_3 & 1 \\
 b_0 & b_1 & b_2 & b_3 & 1 \\
 c_0 & c_1 & c_2 & c_3 & 1 \\
 d_0 & d_1 & d_2 & d_3 & 1 \\
  1 & 1 & 1 & 1 & 0
\end{pmatrix} \right|
\end{equation}

Notes.

- Rows correspond to Quadray 4-tuples of the vertices; the last row encodes the affine constraint. Division by 4 returns IVM tetravolume.

## Expanded Ace 5×5 Matrix

The expanded form of the Ace 5×5 matrix with explicit Quadray coordinates:

\begin{equation}\label{eq:ace5x5_expanded}
M(q_0,q_1,q_2,q_3) = \begin{bmatrix}
q_{01} & q_{02} & q_{03} & q_{04} & 1 \\
q_{11} & q_{12} & q_{13} & q_{14} & 1 \\
q_{21} & q_{22} & q_{23} & q_{24} & 1 \\
q_{31} & q_{32} & q_{33} & q_{34} & 1 \\
1 & 1 & 1 & 1 & 0
\end{bmatrix}, \qquad V_{ivm} = \tfrac{1}{4}\,\big|\det M(q_0,q_1,q_2,q_3)\big|
\end{equation}

Notes.

- **Matrix structure**: Each row represents a vertex with its Quadray coordinates plus affine coordinate 1.
- **Last row**: Enforces projective normalization constraint.
- **Volume computation**: Determinant divided by 4 gives IVM tetravolume.

XYZ determinant volume and S3 conversion:

\begin{equation}\label{eq:xyz_det}
V_{xyz} = \tfrac{1}{6} \left| \det \begin{pmatrix}
 x_a & y_a & z_a & 1 \\
 x_b & y_b & z_b & 1 \\
 x_c & y_c & z_c & 1 \\
  x_d & y_d & z_d & 1
\end{pmatrix} \right|, \qquad V_{ivm} = S3\, V_{xyz},\quad S3=\sqrt{\tfrac{9}{8}}
\end{equation}

Notes.

- Homogeneous determinant in Cartesian coordinates for tetra volume; conversion to IVM units uses $S3=\sqrt{9/8}$ as used throughout.

## Cayley-Menger Determinant (Coxeter.4D)

For tetrahedron volume from edge lengths (Coxeter.4D approach):

\begin{equation}\label{eq:cayley_menger}
288\,V^2 = \det\begin{pmatrix}
  0 & 1 & 1 & 1 & 1 \\
  1 & 0 & d_{01}^2 & d_{02}^2 & d_{03}^2 \\
  1 & d_{10}^2 & 0 & d_{12}^2 & d_{13}^2 \\
  1 & d_{20}^2 & d_{21}^2 & 0 & d_{23}^2 \\
  1 & d_{30}^2 & d_{31}^2 & d_{32}^2 & 0
\end{pmatrix}
\end{equation}

Notes.

- **Pairwise distances**: $d_{ij}$ are Euclidean distances between vertices $P_i$ and $P_j$.
- **Length-only formulation**: Cayley–Menger provides a length-only formula for simplex volumes, here specialized to tetrahedra.
- **Conversion to IVM**: Use $V_{ivm} = S3 \cdot V_{xyz}$ with $S3=\sqrt{9/8}$ to convert to IVM units.

## Piero della Francesca Formula (PDF)

For tetrahedron volume from edge lengths meeting at a vertex:

\begin{equation}\label{eq:pdf}
144\,V_{xyz}^2 = 4 a^2 b^2 c^2 - a^2\,(b^2 + c^2 - f^2)^2 - b^2\,(c^2 + a^2 - e^2)^2 - c^2\,(a^2 + b^2 - d^2)^2 + (b^2 + c^2 - f^2)(c^2 + a^2 - e^2)(a^2 + b^2 - d^2)
\end{equation}

Notes.

- **Edge lengths**: $a,b,c$ are edges meeting at a vertex, $d,e,f$ are opposite edges.
- **Conversion to IVM**: Use $V_{ivm} = S3 \cdot V_{xyz}$ with $S3=\sqrt{9/8}$.

## Gerald de Jong Formula (GdJ)

Native Quadray formula for tetrahedron volume:

\begin{equation}\label{eq:gdj}
V_{ivm} = \frac{1}{4}\,\left|\det \begin{pmatrix}
a_1-a_0 & a_2-a_0 & a_3-a_0 \\
b_1-b_0 & b_2-b_0 & b_3-b_0 \\
c_1-c_0 & c_2-c_0 & c_3-c_0 \\
\end{pmatrix}\right|
\end{equation}

Notes.

- **Quadray differences**: Each column represents edge vectors $P_1-P_0$, $P_2-P_0$, $P_3-P_0$ in Quadray coordinates.
- **Native IVM**: Returns tetravolume directly in IVM units without conversion.
- **Integer arithmetic**: Exact for integer Quadray coordinates.

See code: [`tetra_volume_cayley_menger`](03_quadray_methods.md#code:tetra_volume_cayley_menger). For tetrahedron volume background, see [Tetrahedron – volume](https://en.wikipedia.org/wiki/Tetrahedron#Volume). Exact integer determinants in code use the [Bareiss algorithm](https://en.wikipedia.org/wiki/Bareiss_algorithm). External validation: these formulas align with implementations in the 4dsolutions ecosystem. See the [Resources](07_resources.md) section for comprehensive details.

## Fisher Information Matrix (FIM) {#eq:fim}

Background: [Fisher information](https://en.wikipedia.org/wiki/Fisher_information).

\begin{equation}\label{eq:fim}
F_{i,j} = \mathbb{E}\left[ \frac{\partial \, \log p(x;\theta)}{\partial \theta_i}\, \frac{\partial \, \log p(x;\theta)}{\partial \theta_j} \right]
\end{equation}

Notes.

- Defines the Fisher information matrix as the expected outer product of score functions; see [Fisher information](https://en.wikipedia.org/wiki/Fisher_information).

Figure: empirical estimate shown in the FIM heatmap figure. See code: [`fisher_information_matrix`](03_quadray_methods.md#code:fisher_information_matrix).

See `src/information.py` — empirical outer-product estimator (`fisher_information_matrix`).

## Empirical Fisher Information Matrix

For empirical estimation from data, the Fisher Information Matrix is computed as:

\begin{equation}\label{eq:fim_empirical}
F_{i,j} = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial \, \log p(x_n;\theta)}{\partial \theta_i}\, \frac{\partial \, \log p(x_n;\theta)}{\partial \theta_j}
\end{equation}

Notes.

- Empirical estimate of the FIM from $N$ data samples; converges to the theoretical FIM as $N \to \infty$.
- Used in natural gradient descent and information geometry applications.

## Natural Gradient {#eq:natgrad}

Background: [Natural gradient](https://en.wikipedia.org/wiki/Natural_gradient) (Amari).

\begin{equation}\label{eq:natural_gradient}
\theta \leftarrow \theta - \eta\, F(\theta)^{-1}\, \nabla_{\theta} L(\theta)
\end{equation}

Explanation.

- Natural gradient update: right-precondition the gradient by the inverse of the Fisher metric (Amari); see [Natural gradient](https://en.wikipedia.org/wiki/Natural_gradient).

See code: [`natural_gradient_step`](03_quadray_methods.md#code:natural_gradient_step).

See `src/information.py` — damped inverse-Fisher step (`natural_gradient_step`).

## Free Energy (Active Inference) {#eq:free_energy}

\begin{equation}\label{eq:free_energy}
\mathcal{F} = -\log P(o\mid s) + \mathrm{KL}\big[ Q(s)\;\|\; P(s) \big]
\end{equation}

Explanation.

- **Partition**: variational free energy decomposes into expected negative log-likelihood and KL between approximate posterior and prior; see [Free energy principle](https://en.wikipedia.org/wiki/Free_energy_principle).

See code: [`free_energy`](03_quadray_methods.md#code:free_energy).

See `src/information.py` — discrete-state variational free energy (`free_energy`).

**Note**: The main figures demonstrating natural gradient trajectories and free energy landscapes are shown in [Section 4: Optimization in 4D](04_optimization_in_4d.md). The appendix focuses on unique figures specific to mathematical formulations and validation.

## Quadray Normalization (Fuller.4D)

Given $q=(a,b,c,d)$, choose $k=\min(a,b,c,d)$ and set $q' = q - (k,k,k,k)$ to enforce at least one zero with non-negative entries.

## Distance (Embedding Sketch; Coxeter.4D slice)

Choose linear map $M$ from quadray to $\mathbb{R}^3$ (or $\mathbb{R}^4$) consistent with tetrahedral axes; then $d(q_1,q_2) = \lVert M(q_1) - M(q_2) \rVert_2$.

## Minkowski Line Element (Einstein.4D analogy)

\begin{equation}\label{eq:minkowski_line_element}
ds^2 = -c^2\,dt^2 + dx^2 + dy^2 + dz^2
\end{equation}

Background: [Minkowski space](https://en.wikipedia.org/wiki/Minkowski_space).

## High-Precision Arithmetic Note

When evaluating determinants, FIMs, or geodesic distances for sensitive problems, use quad precision (binary128) via GCC's `libquadmath` (`__float128`, functions like `expq`, `sqrtq`, and `quadmath_snprintf`). See [GCC libquadmath](https://gcc.gnu.org/onlinedocs/libquadmath/index.html). Where possible, it is useful to use symbolic math libraries like SymPy to compute exact values.

### Reproducibility artifacts and external validation

- **This manuscript's artifacts**: Raw data in `quadmath/output/` for reproducibility and downstream analysis:
  - `fisher_information_matrix.csv` / `.npz`: empirical Fisher matrix and inputs
  - `fisher_information_eigenvalues.csv` / `fisher_information_eigensystem.npz`: eigenspectrum and eigenvectors
  - `natural_gradient_path.png` with `natural_gradient_path.csv` / `.npz`: projected trajectory and raw coordinates
  - `ivm_neighbors_data.csv` / `ivm_neighbors_edges_data.npz`: neighbor coordinates (Quadray and XYZ)

  - `polyhedra_quadray_constructions.png`: synergetics volume relationships schematic

- **External validation resources**: The [4dsolutions ecosystem](https://github.com/4dsolutions) provides extensive cross-validation. See the [Resources](07_resources.md) section for comprehensive details on computational implementations and validation.

## Namespaces summary (notation)

- Coxeter.4D: Euclidean E⁴; regular polytopes; not spacetime (cf. Coxeter, Regular Polytopes, Dover ed., p. 119). Connections to higher-dimensional lattices and packings as in Conway & Sloane.
- Einstein.4D: Minkowski spacetime; indefinite metric; used here only as a metric analogy when discussing geodesics and information geometry.
- Fuller.4D: Quadrays/IVM; tetrahedral lattice with integer tetravolume; unit regular tetrahedron has volume 1; synergetics scale relations (e.g., S3).
