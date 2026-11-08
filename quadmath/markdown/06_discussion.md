# Interpretation, Evidence, and Limits of Quadray Computation {#discussion}

The implemented contribution is a coherent set of exact native geometric primitives and numerical demonstrations. Common-offset equivalence gives rank-three physical coordinates; integer labels yield rational quarter-unit tetravolumes. These facts do not establish physical energy levels, automatic robustness, faster optimization, or geodesic natural-gradient trajectories. The [methods](03_quadray_methods.md) and [optimization chapter](04_optimization_in_4d.md) specify the narrower computational contracts.

## Scope and Limitations

- **Embedding and units**: Distances depend on the declared map. The default scale-one map is the canonical comparison convention here, not a claim that every arbitrary embedding preserves physical geometry.
- **Exact versus floating arithmetic**: Native determinants preserve integer/rational information. Floating objectives, dense solvers, statistical estimators, and displays retain numerical limitations.
- **Experiments versus guarantees**: Seeded toy examples illustrate behavior. They do not prove application-level improvement or global convergence.
- **Proposals versus implementations**: Hybrid optimizers, GPU acceleration, consensus security, and geospatial applications remain [research extensions](05_extensions.md).

## Fisher Information and Curvature

Population Fisher information is a positive-semidefinite score metric, positive definite only on identifiable directions. The regression figures use sample loss-gradient Gram matrices; their eigenvalues describe the displayed gradient sensitivities and are not automatically loss-Hessian eigenvalues. The [equations appendix](08_equations_appendix.md#eq:fim) separates the population and empirical definitions.

`natural_gradient_step` returns a damped metric increment. Its continuous undamped ideal is coordinate-covariant steepest descent, not generally a geodesic. `information_geometric_distance` evaluates one fixed quadratic metric; `information_length` is a gradient-weighted path-length proxy. Neither computes a general position-dependent Fisher geodesic. Comparisons with tensegrity or least action are analogies, not equivalences proved by these functions.

## Quadray Coordinates and 4D Structure (Fuller.4D vs Coxeter.4D vs Einstein.4D)

Four Quadray labels do not give four independent Euclidean dimensions. A Cartesian metric pulled back through the $3\times4$ embedding has a common-offset null direction; no generic block-diagonal statistical structure follows from tetrahedral symmetry alone. The [namespace reference](02_4d_namespaces.md) distinguishes ordinary XYZ geometry, true Euclidean four-space, and Minkowski spacetime.

## Integrating FIM with Quadray Models

- Supply actual per-observation likelihood scores when estimating Fisher information; otherwise describe the matrix as a gradient Gram surrogate.
- Inspect null directions and conditioning rather than assuming invertibility.
- Add the increment returned by `natural_gradient_step` to the current parameters; damping and finite step size change the ideal invariant flow.
- An explicit hybrid projection scheme needs its own objective and evaluation. The existing discrete optimizer is not automatically Fisher-guided.

## Implications for Optimization and Estimation

### Clarifications on "frequency/time" dimensions

Graph-shell frequency counts lattice steps. It is not physical temporal frequency or an extra Euclidean dimension. Simulated time is an explicit update index; no physical dynamics is implied without model assumptions and units.

### On distance-based tetravolume formulas (clarification)

Use native determinants for integer Quadray input. For length-based Cayley–Menger or PdF formulas, convert Euclidean volume by $3/8$ in default scale-one XYZ, or by $S3=\sqrt{9/8}$ only in sphere-radius-one XYZ. The [conversion contract](03_quadray_methods.md#sec:xyz_conversion) is authoritative. Arbitrary integer Quadray vertices may have fractional native volumes.

### Symbolic analysis (bridging vs native) (Results linkage)

The manuscript's native-versus-symbolic comparison is reproducible through its figure script and saved data. Agreement on the examples is finite numerical/symbolic evidence; the exact determinant identity and declared scale explain the relationship. This chapter does not claim a new execution of those artifacts.

K-FAC is external research on structured curvature approximations, not an implemented QuadMath optimizer; see [the K-FAC paper](https://arxiv.org/abs/1503.05671). Near-zero metric modes can diagnose redundancy, but reliability depends on the model, scores, and sampling regime.

## Community Ecosystem and Validation

The [4dsolutions references](07_resources.md) provide independent implementations and pedagogy. Cross-implementation agreement can be useful evidence only when versions, units, cases, and actual comparisons are recorded. A repository link alone is not a validation result. Likewise, [Lean proofs](../../SPEC.md#73-lean-4-mirror-status) cover the stated integer quotient/metric/volume/bound theorems, not floating solvers, universal shell counts, gameplay, or empirical application claims.
