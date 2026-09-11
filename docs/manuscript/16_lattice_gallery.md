# Lattice Visualization Gallery

## Overview

This section is the figure surface of the lattice layer: every rendering
primitive used by the IVM chapters lives in `vis_lattice.py`
(`shell_scatter`, `field_slice`, `dynamics_strip`, `gallery`).  Each
primitive receives its matplotlib axes explicitly, so panels compose inside
caller-owned figures; only `gallery` creates figures (three, written as
PNGs).  Nothing renders at import time, and all randomness is drawn from a
seeded `numpy.random.default_rng`, so the gallery is deterministic.  The
command-line entry is `quadmath/scripts/lattice_gallery.py` — a thin
orchestrator that sets a headless backend and a fixed seed, delegates to
`gallery` (thin-orchestrator contract of `quadmath/scripts/AGENTS.md`), and
prints each written path on its own line: those stdout lines are the
`make_all_figures` manifest contract.

## Frequency shells

`shell_scatter(ax, sites, k)` embeds the sites of shell `k` (from
:func:`omni_numbering.generate_shell`, which enumerates the
\(10k^2 + 2\) sites of shell \(k\); see
[Lattice Tooling](13_lattice_tooling.md)) with `to_xyz` under
`DEFAULT_EMBEDDING` and scatters them in 3D.  With `axis_hints` it overlays
four dashed rays from the origin toward the embedding matrix columns — the
images of the quadray basis directions \(A, B, C, D\), which are the
vertices of a regular tetrahedron (pairwise dot product \(-1\), edge
length \(2\sqrt{2}\); see [Quadray Methods](03_quadray_methods.md)).  The
hints make the tetrahedral axis frame of the lattice readable in print.

![**Frequency shells of the omnidirectional close packing.** Shell 1 (12 sites) and shell 2 (42 sites) of the IVM lattice under `DEFAULT_EMBEDDING`, rendered by `shell_scatter`; dashed rays mark the four tetrahedral quadray axes labeled A, B, C, D.](figures/vis_gallery_shell.png)

## Lattice-plane slices

`field_slice(ax, field, sites, plane)` renders a scalar
`ivm_field.IVMField` as a heatmap restricted to a lattice plane.  A plane
is parametrized by an origin \(q_0\) and two independent integer
translation vectors \(u\), \(v\) — by default two of the twelve IVM
neighbor moves, both permutations of \((2,1,1,0)\) — and its points are

\begin{equation}
\label{eq:vis-plane}
q(i, j) \;=\; \operatorname{normalize}\!\big(q_0 + i\,u + j\,v\big),
\qquad (i, j) \in \mathbb{Z}^2 ,
\end{equation}

with `normalize` the quadray canonical representative.  Because each
neighbor move has component sum \(4\), the sum stays divisible by \(4\)
along the whole plane, so every \(q(i,j)\) is again a lattice site.  Two
coordinate facts make the rendering exact: the embedding is linear, so the
xyz image is the planar set \(t(q_0) + i\,t(u) + j\,t(v)\); and the shift
\((1,1,1,1)\) lies in the embedding kernel, so plane membership is tested
in integer quadray space — a site lies on the plane of \eqref{eq:vis-plane}
iff \(q - q_0 = i\,u + j\,v + m\,(1,1,1,1)\) for integers \(i, j, m\).
`field_slice` decides membership by an exact least-squares solve followed
by an integer verification, so no site is ever misassigned by floating
point.  Sites off the plane are ignored, plane cells outside the field's
ball are masked, and the heatmap axes are the integer plane indices \(i\)
(steps along \(u\)) and \(j\) (steps along \(v\)).

The gallery figure learns a field on the radius-3 ball (147 sites) from
noisy observations of a linear synthetic truth (25% of sites observed,
noise level \(0.1\), regularization `lam=0.01`, kernel width `0.5`) and
slices it through the default plane
\(\big((2,1,1,0), (1,2,1,0)\big)\); the learned surface follows the
planar lattice, with masked tiles where the ball ends.  Field learning
itself is the subject of [Static IVM Field Learning](11_ivm_field_learning.md).

![**Learned scalar field sliced along a lattice plane.** Heatmap of a Laplacian-regularized `IVMField` over the radius-3 ball, restricted by `field_slice` to the plane spanned by the neighbor moves u = (2,1,1,0) and v = (1,2,1,0); masked tiles mark plane cells outside the ball.](figures/vis_gallery_field.png)

## Dynamics strips

`dynamics_strip(axs, trajectory, t_indices)` renders selected snapshots of
an `ivm_dynamics.simulate` trajectory (see
[Dynamics and Learning on the IVM Lattice](12_ivm_dynamics.md)) as a strip
of 3D scatter panels over the embedded lattice.  All panels share one
symmetric color scale \([-v_{\max}, v_{\max}]\) with
\(v_{\max} = \max_t |u_t|\) over the selected snapshots, so evolution is
directly comparable across panels.  The gallery strip shows heat diffusion
(\(\alpha = 0.45\), horizon \(T = 20\)) on the radius-3 lattice (27 sites)
at \(t = 0, 10, 20\): the seeded random initial field relaxes toward its
lattice average — the visually flat mid panel and right panel are the
\(L_2\)-monotone decay of the heat lemma in action.

![**Heat-diffusion evolution strip.** Snapshots of a seeded heat run at t = 0, 10, 20 on the radius-3 IVM lattice, rendered by `dynamics_strip` on one shared symmetric color scale; the field visibly relaxes as the sum of squares decays monotonically.](figures/vis_gallery_dynamics.png)

## Reproducibility and test contract

- `gallery(paths_out_dir, seed=12)` writes exactly the three files
  `vis_gallery_shell.png`, `vis_gallery_field.png`,
  `vis_gallery_dynamics.png` (module constant `GALLERY_FILES`, in that
  order) into `paths_out_dir`, creating it when missing, and returns their
  paths.  Every panel is fully seeded, and the PNGs carry no timestamp
  metadata: the test suite asserts byte-identical re-renders for a fixed
  seed.
- Tests: `tests/test_vis_lattice.py` — 19 tests, no mocks, deterministic
  assertions without pixel diffs (artist placement on caller-provided
  axes, exact field values at known lattice cells, byte-level
  reproducibility).  Scope with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest
  tests/test_vis_lattice.py -q` and `uv run coverage report` — 100%
  statement and branch coverage of `src/vis_lattice.py`.
- Figure regeneration:
  `uv run python quadmath/scripts/lattice_gallery.py` (the script sets
  `MPLBACKEND=Agg` itself); stdout is exactly the three written paths.
- Markdown: `uv run python quadmath/scripts/validate_markdown.py`.

## Cross-references

- Site geometry, normalization, and the embedding:
  [Quadray Methods](03_quadray_methods.md).
- Shell enumeration and nearest-site search:
  [Lattice Tooling](13_lattice_tooling.md).
- The sliced learner: [Static IVM Field
  Learning](11_ivm_field_learning.md).
- The stripped dynamics: [Dynamics and Learning on the IVM
  Lattice](12_ivm_dynamics.md).
