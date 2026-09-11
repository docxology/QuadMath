# Dynamics and Learning on the IVM Lattice

## Overview

Where [Section 4](04_optimization_in_4d.md) descends a *single point* along the
IVM lattice, this section evolves a *field* — one scalar per lattice site —
and then asks the inverse question: given an observed field trajectory, what
coupling produced it? Everything here is implemented in `ivm_dynamics.py`
(`ball_sites`, `make_lattice`, `heat_step`, `majority_step`, `step`,
`simulate`, `fit_trajectory`, `sum_of_squares`, `is_nonincreasing`) on top of
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
canonical sites with squared radius at most \(R^2\): for a canonical site the
squared radius is at least twice the square of its largest component, so
components in \([0, R]\) cover the ball.

`make_lattice(R)` joins two sites when one of the 12 canonical IVM neighbor
shifts (`neighbor_shifts`, all permutations of \((2,1,1,0)\), each at squared
radius 8 — the close-packing distance) maps one site onto the other after
renormalization. Normalization never moves the embedded point, so adjacency
is exactly "embedding difference is one of the 12 shifts". On the \(R = 3\)
ball (27 sites) this graph decomposes into:

- the close-packed **12-around-one cluster**: the origin (degree 12) and its
  twelve cuboctahedron neighbors (degree 5);
- two regular tetrahedra on the odd-parity sites \((\pm1,\pm1,\pm1)\) (degree 3);
- one octahedron on the axis sites \((\pm2,0,0)\) and permutations (degree 4).

The move graph is *not* connected across these cosets — a structural fact of
the \((2,1,1,0)\) move set, not a defect: the update laws below act
independently on each component.

## Heat update and the monotonicity lemma

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

*Proof sketch.* \(S\) is symmetric with spectrum in \([-1,1]\) (it is similar
to the random-walk matrix \(D^{-1}A\)). The update operator
\((1-\alpha)I + \alpha S\) is therefore symmetric with eigenvalues in
\([1-2\alpha,\,1] \subseteq [-1,1]\), hence non-expansive in the 2-norm.
\(\square\)

This is exactly the claim the test suite asserts — *per step*, across seeds
and the full coupling range (`test_heat_update_l2_nonincreasing_per_step` in
`tests/test_ivm_dynamics.py`). Two scope restrictions are deliberate and
tested:

1. The guarantee is specific to the symmetric normalization \(S\). Plain
   row-average diffusion \(P = D^{-1}A\) on a truncated ball is **not** covered
   (boundary rows make \(P\) non-doubly-stochastic); the module neither uses
   nor claims it.
2. In the demo run (\(\alpha = 0.35\), \(T = 40\), seed 7) the sum of squares
   decreases monotonically from \(37.04\) to \(3.93\) while never increasing
   at any single step.

## Majority update on integer states

For integer-valued fields the rounded-averaging ("majority") update is

\begin{equation}
\label{eq:ivmdyn-majority}
u_{t+1}(s) \;=\; \operatorname{rint}\!\Big( (1-\alpha)\, u_t(s)
\;+\; \alpha\, \tfrac{1}{\deg(s)} \!\!\sum_{n \in N(s)} u_t(n) \Big),
\end{equation}

with \(N(s)\) the neighbor set of site \(s\) and \(\deg(s) = |N(s)|\) its
degree, identity rows at isolated sites, and round-half-to-even ties
(`numpy.rint`), computed by `majority_step`. Every new value is a rounded convex
combination of the site and its neighbors, so the extremal structure is
preserved:

**Proposition (range containment).** Under \eqref{eq:ivmdyn-majority} the
field maximum is non-increasing and the minimum non-decreasing, at every step,
for every \(\alpha \in [0,1]\).

We deliberately make **no** sum-of-squares claim for this update: rounding can
increase it, and the test suite pins a demonstrating case (origin at 0 with
its twelve packers at 1 maps to thirteen sites at value 1, raising the sum of
squares from 12 to 13 — `test_majority_step_sum_of_squares_can_increase`).
The honest summary: heat is \(L_2\)-monotone, majority is range-preserving,
and neither claim is stretched to cover the other update.

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
beyond the refinement resolution is claimed. In the demo configuration
(synthetic trajectory generated at \(\alpha_{\text{true}} = 0.3\) on the
\(R=3\) lattice, \(T = 40\)), a 21-point grid plus refinement (33 MSE
evaluations) recovers \(\alpha \approx 0.300\) with
\(\mathcal{L} \approx 4.5 \times 10^{-33}\) — machine precision for a
bitwise-deterministic re-simulation. When the true coupling lies off-grid,
refinement converges to within a grid-spacing fraction of it
(`test_fit_refinement_finds_offgrid_coupling`: 0.37 recovered to
\(\le 0.0125\) from a coarse \(\{0, 0.25, 0.5, 0.75, 1\}\) grid).

## Demo figure

Generated by `uv run python quadmath/scripts/ivm_dynamics_demo.py`
(`MPLBACKEND=Agg`); all randomness is seeded (`DynamicsParams.seed`), so the
figure is reproducible byte-for-byte up to PNG encoding.

![**Dynamics and learning on the IVM lattice (\(R=3\) ball, 27 sites)**. Top row: heat-field snapshots at \(t = 0, 20, 40\) (\(\alpha = 0.35\), seed 7) over the embedded quadray sites (XYZ axes in embedding units); dot color encodes field value on a shared symmetric scale, and diffusion homogenizes the field by \(t = 40\). Bottom left: sum-of-squares histories over \(t = 0 \ldots 40\) — the heat curve decreases monotonically (lemma \eqref{eq:ivmdyn-lemma}); the majority curve falls steeply and plateaus, but unlike heat it carries no monotonicity guarantee, since rounding can increase the sum of squares (`test_majority_step_sum_of_squares_can_increase`). Bottom middle: trajectory MSE versus coupling \(\alpha\) on a logarithmic axis over the search grid, with the identified optimum (`fit 0.3000`) coinciding with the true generating \(\alpha = 0.3\) (dotted). Bottom right: the final integer majority field at \(t = 40\). Generated by `uv run python quadmath/scripts/ivm_dynamics_demo.py` (`MPLBACKEND=Agg`, seeded via `DynamicsParams.seed`).](figures/ivm_dynamics_demo.png)

## Reproducibility and test contract

- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest
  tests/test_ivm_dynamics.py -q` then `uv run coverage report` — 40 tests,
  100% statement and branch coverage of `src/quadmath/lattice/ivm_dynamics.py`, no mocks, all
  examples real numerics with fixed seeds.
- Markdown: `uv run python quadmath/scripts/validate_markdown.py`.
- The two provable claims (lemma \eqref{eq:ivmdyn-lemma}, the range
  proposition under \eqref{eq:ivmdyn-majority}) are asserted per step; the
  non-claims (row-average diffusion, majority \(L_2\) behavior) are pinned by
  the honesty tests described above rather than left implicit.

## Cross-references

- Site geometry and normalization: [Quadray Methods](03_quadray_methods.md).
- Discrete point descent on the same move set:
  [Optimization in 4D](04_optimization_in_4d.md).
- Field/energy vocabulary (free energy, Fisher information):
  [Appendix B](09_free_energy_active_inference.md); equation index:
  [Appendix A](08_equations_appendix.md).