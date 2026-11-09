# Dynamics and Learning on the IVM Lattice

## Field evolution and inverse coupling estimation

Where [Section 4](04_optimization_in_4d.md) descends a *single point* along the
IVM lattice, this section evolves a *field* — one scalar per lattice site —
and then asks the inverse question: given an observed field trajectory, what
coupling produced it? Everything here is implemented in `ivm_dynamics.py`
(`ball_sites`, `make_lattice`, `heat_step`, `conservative_heat_step`,
`graph_laplacian`, `dirichlet_energy`, `majority_step`, `step`, `simulate`,
`fit_trajectory`, `sum_of_squares`, `is_nonincreasing`) on top of
the `Quadray` class and `to_xyz` embedding in `quadray.py`; the demo figure is
produced by `quadmath/scripts/ivm_dynamics_demo.py` (which delegates to
`render_dynamics_demo` in `src/quadmath/lattice/ivm_dynamics.py`, per the thin-orchestrator
contract in `quadmath/scripts/AGENTS.md`).

## Lattice sites, shells, and the 12-around-one move graph

A *site* is a canonical quadray representative — non-negative integer
components with at least one zero, selected by `Quadray.normalize` (see
[Quadray Methods](03_quadray_methods.md)). The radial observable is the
squared embedding norm `site_radius_sq(q)`; because the shift vector
\((1,1,1,1)\) lies in the kernel of the `DEFAULT_EMBEDDING` matrix, the radius
is invariant under quadray normalization. `ball_sites(R)` enumerates the
canonical FCC sites with squared radius at most \(R^2\). Its finite scan
bounds are conservative; the Euclidean radius here differs from the
graph-shell radius used by `IVMField.lattice_ball`.

`make_lattice(R)` joins two sites when one of the 12 canonical IVM neighbor
shifts (`neighbor_shifts`, all permutations of \((2,1,1,0)\), each at squared
radius 8 — the close-packing distance) maps one site onto the other after
renormalization. Normalization never moves the embedded point, so adjacency
is exactly "embedding difference is one of the 12 shifts". Lattice sites are
the quadrays whose coordinate sum is divisible by 4; the remaining residue
classes are IVM voids and are excluded. On the \(R = 3\) ball (13 sites) the
graph is a single connected component: the close-packed **12-around-one
cluster**, i.e. the origin (degree 12) and its twelve cuboctahedron neighbors
(degree 5). The update laws below act independently on each connected
component of the ball.

## Symmetric-normalized heat and its squared-norm bound {#heat-update-and-the-monotonicity-lemma}

The heat (diffusion) update blends a field \(u_t \in \mathbb{R}^{N}\) — one
value per lattice site, \(N\) the site count — with its symmetric-normalized
adjacency average \(S = D^{-1/2} A D^{-1/2}\), where \(A\) is the adjacency
matrix of the move graph, \(D\) the diagonal degree matrix, and rows of
\(S\) are zero at isolated sites:

\begin{equation}
\label{eq:ivmdyn-heat}
u_{t+1} \;=\; (1-\alpha)\, u_t \;+\; \alpha\, S\, u_t ,
\qquad \alpha \in [0,1] .
\end{equation}

**Lemma (heat averaging is non-increasing in sum of squares).** For every
\(\alpha \in [0,1]\), every lattice built by `make_lattice`, and every state,

\begin{equation}
\label{eq:ivmdyn-lemma}
\lVert u_{t+1} \rVert_2^2 \;\le\; \lVert u_t \rVert_2^2 .
\end{equation}

*Proof sketch.* On non-isolated vertices, \(S\) is symmetric and similar to
the random-walk matrix, with spectrum in \([-1,1]\). Isolated rows contribute
zero eigenvalues of \(S\). Thus \((1-\alpha)I+\alpha S\) has eigenvalues in
\([1-2\alpha,1]\subseteq[-1,1]\) and is non-expansive in the Euclidean norm.
\(\square\)

This is exactly the claim the test suite asserts — *per step*, across seeds
and the full coupling range (`test_heat_update_l2_nonincreasing_per_step` in
`tests/unit/lattice/test_ivm_dynamics.py`). Two scope restrictions are deliberate and
tested:

1. The guarantee is specific to the symmetric normalization \(S\). Plain
   row-average diffusion \(P = D^{-1}A\) on a truncated ball is **not** covered
   (boundary rows make \(P\) non-doubly-stochastic); the module neither uses
   nor claims it.
2. This operator does **not** conserve ordinary scalar mass on an irregular
   graph. For \(\alpha>0\), its stationary profiles on a non-isolated component are
   proportional to \(\sqrt{\deg(s)}\), not generally constant. Isolated values
   decay by \(1-\alpha\). Use the conservative operator for concentrations.

## Conservative graph transport

For the same undirected graph, define \(L=D-A\) and let \(d_{\max}\) be its
maximum degree. `conservative_heat_step` applies synchronous edge flux:

\begin{equation}
\label{eq:ivmdyn-conservative}
u_{t+1}=u_t-\frac{\alpha}{d_{\max}}Lu_t,\qquad 0\leq\alpha\leq1 .
\end{equation}

An edgeless graph is the identity. Otherwise the update matrix is symmetric,
non-negative and doubly stochastic: every row and column sums to one.
Consequently it preserves each component's scalar mass, constant fields and
the initial range. Its eigenvalues lie in \([-1,1]\), so both squared norm
and graph Dirichlet energy are non-increasing:

\begin{equation}
\label{eq:ivmdyn-dirichlet}
E(u)=\tfrac12 u^{\mathsf T}Lu
     =\tfrac12\sum_{\{i,j\}\in\mathcal E}(u_i-u_j)^2 .
\end{equation}

`graph_laplacian` and `dirichlet_energy` expose these observables. The flux
implementation costs \(O(N+|\mathcal E|)\); accumulation precedes the final
state addition. Exceptional floating ranges use exact binary-rational fluxes
before rounding, preserving recoverable subnormal flow and finite constants.
For maximum absolute input $M>\mathrm{MAX}/4$, exact accumulation also avoids
overflow of the total incident flux plus the old state, not just individual
edge differences. On the ordinary path the incident absolute flux is bounded
by $2M$ and the old state plus flux by $3M$.
Conservation for ordinary floating trajectories holds up to roundoff, not
bitwise equality. At \(\alpha=1\), bipartite components can oscillate, so
non-increasing energy is not a claim of strict convergence.

For an independent hand calculation, put 12 at the center of the 13-site
cluster and zero elsewhere. A conservative half-step gives center 6 and
each shell site \(1/2\): mass stays 12 and \(E\) falls from 864 to 181.5.

```python
import numpy as np
from quadmath import DynamicsParams, conservative_heat_step, make_lattice, simulate

lattice = make_lattice(3)
initial = np.zeros(lattice.size)
initial[0] = 12
assert conservative_heat_step(initial, lattice, 0.5)[0] == 6
trajectory = simulate(12, DynamicsParams(kind="conservative", alpha=0.35),
                      lattice, initial)
```

The six-panel comparison in [Lattice Visualization](16_lattice_gallery.md)
uses one color normalization for conservative and symmetric-normalized
transport and displays their different equilibria. QuadCraft's generic
`ConservativeGraphField` implements the same edge operator; SimAnt supplies
an **eight-neighbor raw-label graph**, not this twelve-neighbor FCC graph,
and accounts for evaporation and sources separately. See the
[suite contract](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md).

## Majority update on integer states

For integer-valued fields the rounded-averaging ("majority") update is

\begin{equation}
\label{eq:ivmdyn-majority}
u_{t+1}(s) \;=\; \operatorname{rint}\!\Big( (1-\alpha)\, u_t(s)
\;+\; \alpha\, \tfrac{1}{\deg(s)} \!\!\sum_{n \in N(s)} u_t(n) \Big),
\end{equation}

with \(N(s)\) the neighbor set of site \(s\) and \(\deg(s) = |N(s)|\) its
degree, identity rows at isolated sites, and round-half-to-even ties.
`majority_step` evaluates the binary floating coupling as an exact rational,
combines unbounded native integer counts, then rounds once. Output uses
`int64` when representable and native-integer object arrays otherwise.
Every new value is a rounded convex combination, so the extremal structure
is preserved:

**Proposition (range containment).** Under \eqref{eq:ivmdyn-majority} the
field maximum is non-increasing and the minimum non-decreasing, at every step,
for every \(\alpha \in [0,1]\).

We deliberately make **no** sum-of-squares claim for this update: rounding can
increase it, and the test suite pins a demonstrating case (origin at 0 with
its twelve packers at 1 maps to thirteen sites at value 1, raising the sum of
squares from 12 to 13 — `test_majority_step_sum_of_squares_can_increase`).
The distinctions matter: symmetric-normalized heat is \(L_2\)-monotone;
conservative heat additionally preserves concentration mass and range;
rounded majority preserves integer range but can increase squared norm.
`sum_of_squares` remains exact for integral fields, including counts beyond
binary64 and int64 range.

## Learning the coupling from an observed trajectory

Given observed snapshots \(u_{\mathrm{obs}}(0..T)\) on a known lattice, the
coupling \(\alpha\) is identified gradient-free: `fit_trajectory` re-simulates
the field from the observed initial condition for each candidate coupling and
minimizes the trajectory mean squared error over the \(T\) post-initial
snapshots and the \(N\) sites, where \(u_{\alpha}(t)\) is the re-simulated
field under candidate coupling \(\alpha\):

\begin{equation}
\label{eq:ivmdyn-mse}
\mathcal{L}(\alpha) \;=\; \frac{1}{T\,N} \sum_{t=1}^{T}
\big\lVert u_{\alpha}(t) - u_{\mathrm{obs}}(t) \big\rVert_2^2 ,
\end{equation}

scanning the supplied grid, then re-gridding the interval between the best
candidate's grid neighbors `refine_rounds` times (grid/coordinate search;
ties resolve to the smallest \(\alpha\); fully deterministic). The result is
a global optimum *over the evaluated candidates only* — no convergence
beyond the evaluated grid resolution is claimed. The recorded seeded
demonstration generates data at \(\alpha=0.3\), with \(R=3\) and \(T=40\),
and reports recovery near 0.300. This is synthetic parameter recovery, not
general identifiability: a stationary initial field can contain no coupling
information. Off-grid recovery is limited by the supplied candidates and
refinement policy.

Count residuals are subtracted, squared and averaged exactly before the final
floating score. This includes integral floating observations and fractional
measurement noise: a single-count error at an offset \(2^{2000}\) is not
rounded away. An incorrect candidate with an unrepresentably large MSE scores
infinity without preventing evaluation of a correct candidate. Real-valued
scores use scaled residuals to avoid premature squaring overflow.

`trajectory_train_test` in `quadmath.learn.learning_eval` uses the same score
for a temporal training prefix, held-out multi-step continuation and
teacher-forced one-step prediction. It does not shuffle time or fit held-out
rows.

## Demo figure

Generate with `uv run --python 3.13 python quadmath/scripts/ivm_dynamics_demo.py`.
The seeded numerical inputs are reproducible in a fixed environment; this
is not a promise of identical PNG bytes across scientific-library versions.

![**Dynamics and learning on the 13-site IVM cluster**. Top: symmetric-normalized heat snapshots at \(t=0,20,40\), \(\alpha=0.35\), seed 7, on one color scale; the nonuniform-degree equilibrium is proportional to \(\sqrt{\deg}\), not a homogeneous concentration. Bottom: heat/majority squared norms, coupling-identification MSE and the final integer field. Conservative transport and its constant equilibrium are compared separately in the lattice gallery.](figures/ivm_dynamics_demo.png)

## Reproducibility and test contract

- Run the numerical regressions with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  uv run --python 3.13 pytest tests/unit/lattice/test_ivm_dynamics.py
  tests/unit/learn/test_learning_eval.py -q`.
- Markdown: `uv run --python 3.13 python quadmath/scripts/validate_markdown.py --strict`.
- The regressions exercise hand-computed flux/energy, component mass and range,
  isolated nodes, floating extremes, exact large counts and temporal scoring.
  The normalized-heat and rounded-majority non-claims remain explicit.
- These spectral arguments are mathematical derivations and executable
  numerical checks, not Lean proofs of the floating simulation implementation.

## Cross-references

- Site geometry and normalization: [Quadray Methods](03_quadray_methods.md).
- Discrete point descent on the same move set:
  [Optimization in 4D](04_optimization_in_4d.md).
- Field/energy vocabulary (free energy, Fisher information):
  [Appendix B](09_free_energy_active_inference.md); equation index:
  [Appendix A](08_equations_appendix.md).