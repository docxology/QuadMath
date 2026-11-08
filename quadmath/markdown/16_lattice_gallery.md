# Lattice Visualization Gallery

## Composable lattice views and owned figure outputs

The IVM rendering surface is `vis_lattice.py`: `shell_scatter`,
`field_slice`, `dynamics_strip`, `boundary_comparison` and
`dynamics_comparison` compose on caller-owned matplotlib axes. Only
`gallery` creates figures, writing three deterministic PNGs. The thin
command-line entry `quadmath/scripts/lattice_gallery.py` sets a headless
backend, delegates to `gallery`, and prints the three output paths in
manifest order. Seeded boundary noise and initial fields make repeated
runs reproducible.

## Frequency shells

`shell_scatter(ax, sites, k)` embeds the graph shell `k` from
`omni_numbering.generate_shell`. For \(k>0\) its cardinality is
\(10k^2+2\); shell zero is the origin. The label includes both the graph
shell index and site count. A graph-shell radius counts neighbor moves:
it is **not** an XYZ Euclidean radius. With `axis_hints`, dashed rays
mark the four embedding columns, the tetrahedral quadray directions.
`shell_scatter` and `dynamics_strip` enforce equal physical XYZ metric
aspect using the shared `set_axes_equal` helper.

![**IVM graph shells.** Shell 1 (12 sites) and shell 2 (42 sites) under the scale-1 `DEFAULT_EMBEDDING`, with tetrahedral axis hints and equal XYZ metric aspect.](../output/figures/vis_gallery_shell.png)

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

with `normalize` the canonical quadray representative. Membership means
\(q-q_0=i u+j v+m(1,1,1,1)\) for **integer** \(i,j,m\).
The implementation first subtracts each vector's fourth component,
removing the common-offset gauge. A nonzero two-by-two minor of the
resulting three-by-two integer basis gives Cramer's numerators;
divisibility and all three quotient equations are checked with Python
integers. Independence is required **modulo the common-offset direction**,
not merely in raw four-space. Arbitrarily large integer gauges and
ill-conditioned integer bases therefore do not pass through a floating
rank test or least-squares membership solve.

Each occupied cell has unit width at its actual integer index. Missing
rows, columns and cells remain neutral gaps rather than being stretched
into observations. Consecutive absent indices are represented by one
masked band. To avoid a quadratic Cartesian-product allocation, large
sparse products use a `PolyCollection` of occupied cells instead of a
`QuadMesh`; storage is bounded linearly in sample count, independent of
empty index extent. `field_slice` returns either artist type. These axes
are lattice **indices**, not orthonormal physical coordinates.

For huge in-plane offsets, exact integer local origins are subtracted
before float display conversion. The labels explicitly state
`i - (origin)` / `j - (origin)` and the returned artist records the exact
pair as `plane_index_origin`. Small separations are retained without
claiming that arbitrary huge absolute indices are representable floats.
An optional shared `norm` supports honest cross-panel color comparison;
an identically zero field maps to the colormap midpoint.

## Hard boundary versus soft fitting

`boundary_comparison(axs, hard, soft, boundary_sites, boundary_values)`
uses four caller-owned 2D axes: two matched-color plane slices,
prescribed-versus-fitted boundary values, and absolute interior residuals
\(|Lf|\). It returns the slice artists, boundary-error arrays and interior
residual arrays derived from the supplied model fields.

The gallery supplies the same noisy, spatially varying observations to
`IVMField.harmonic_extension` and `IVMField.learn` on the **graph-shell
radius-3 ball (147 sites)**. Boundary sites are those with fewer than
twelve in-domain neighbors. Hard extension pins those values and solves
\(L_{UU}f_U=-L_{UB}f_B\), so the interior is harmonic up to floating solver
residual. Soft learning instead solves its regularized data-fit system
(`lam=0.4`, kernel width `1.5`); it need not interpolate observations or
have a harmonic interior. Neither fit is labeled exact recovery of an
unknown truth. See [Static IVM Field Learning](11_ivm_field_learning.md).

![**Hard boundary versus soft fit.** Matched-color plane slices, boundary agreement and interior graph-Laplacian residuals on the graph-shell-radius-3 domain. Both methods receive identical observations; their constraints are deliberately different.](../output/figures/vis_gallery_field.png)

## Dynamics and finite-boundary diagnostics

`dynamics_strip(axs, trajectory, t_indices, norm=None)` renders selected
snapshots on a symmetric color scale shared across panels, or an explicit
shared `Normalize` supplied by the caller. A zero trajectory uses
\([-1,1]\), putting zero at the neutral midpoint rather than a colormap
endpoint.

`dynamics_comparison(axs, conservative, normalized)` takes six axes:
three 3D axes for graph degree and the two final fields, followed by
three 2D axes for changes in conserved quantities, the trajectories'
recorded `sos`, and distance from each stationary projection. Both
trajectories must share the same graph, initial state and horizon.
Returned diagnostics include componentwise stationary fields and
residual norms, not an ordinary-mean interpretation of normalized heat.

The gallery compares forty updates at \(\alpha=0.45\) on the
**Euclidean-radius-3 domain (13 sites)**. Conservative transport uses the
unnormalized Laplacian with the model's degree-bound step size, preserves
\(\sum_i u_i\), and has a constant stationary field on each connected
component. Symmetric-normalized heat uses
\(S=D^{-1/2}AD^{-1/2}\); its connected nonisolated stationary mode is
\(\sqrt{d}\), with conserved projection \(\sqrt{d}^{\,T}u\), generally
**not** conserved ordinary sum. Isolated sites retain their concentration
under conservative transport but have zero stationary projection for
positive-coupling normalized heat. The degree panel exposes the finite
boundary that produces this distinction. Stationary references are
computed separately for every connected component.

![**Conservative versus symmetric-normalized diffusion.** Equal-metric final fields and finite-domain degree, changes in ordinary mass and normalized stationary projection, recorded sum of squares, and distance to the operator-specific stationary field. Both operators use the same seeded initial condition on the Euclidean-radius-3 domain.](../output/figures/vis_gallery_dynamics.png)

## Deterministic animations and gallery plots {#sec:animation_frames}

`src/quadmath/viz/animations.py` returns `Frame` objects containing a 2D
brightness array and title. Rendering uses a fixed 48-by-48 orthographic
XYZ-to-XY grid over $[-8,8]^2$, with upward positive Y, maximum brightness
for colliding projections, and border clamping outside the camera.
The default graph-shell-radius-3 ball has coordinate extrema $\pm6$;
its Euclidean radius is not 6.

- `simplex_frames(q0, q1, n=16)` rotates the 13-site **graph-shell-radius-1**
  ball by shortest-arc unit-quaternion slerp. Despite its name, it does not
  animate the optimizing simplex of Section 4.
- `lattice_frames(shells=3, n=12)` pulses `shell_ball_sites(shells)` by
  $1+0.25\sin(2\pi i/n)$; shell 3 contains 147 sites.
- `diffusion_frames(n_steps=12, seed=0)` uses the 147-site graph-shell ball
  and a seeded one-hot source. It applies row-average diffusion at coupling
  0.25, clips at zero, and rescales every displayed frame by its own maximum.
  It is **neither** `heat_step`'s symmetric-normalized operator **nor**
  `conservative_heat_step`'s mass-preserving operator. Brightness across
  different frames is not an absolute concentration scale.

`src/quadmath/viz/plots.py` supplies `plot_loss_history`,
`plot_shell_growth`, `plot_error_histogram`, `plot_lattice_shell_3d`, and
`plot_slerp_path`. These functions own and close their figures; they return
a saved path or an empty string when saving is disabled, not a live figure.
Bare filenames use the configured figure directory, and paths with a
directory component are used as supplied.

`frames_to_gif(frames, out_path, fps=8, scale=8)` assembles actual frame data into
an animated GIF. Float frames are rounded from $255v$ to `uint8`; integer
frames retain their gray levels. Integer `scale` uses nearest-neighbor
resampling. The requested duration is `1000 // fps` milliseconds per frame,
but GIF timing has centisecond resolution. PIL is imported lazily. Numerical
frame reproducibility does not by itself promise identical encoded files
across encoder versions.

`frames_strip(frames, out_path=None, labels=None, save=True)` creates one
grayscale panel per actual `Frame`, using brightness range $[0,1]$ and mapping
`uint8` values through $v/255$. Labels are supplied explicitly or taken from
frame titles. The strip follows the [standalone plot](../../src/quadmath/viz/plots.py)
save convention: a bare name uses the configured figure directory; a path with
a directory component is used as supplied. It returns the saved path, or `""`
when saving is disabled or no name is supplied. Empty frames and mismatched
labels raise `ValueError`. Encoded PNG bytes depend on the rendering toolchain.

![**Animation stills with distinct model semantics.** Left: a pulsing graph-shell-radius-2 ball. Center: a quaternion rotation of the thirteen-site graph-shell-radius-1 cluster. Right: row-average diffusion on the graph-shell-radius-3 ball, normalized by that frame's maximum. These are fixed-camera grayscale projections, not metric-preserving flattened geometries or matched absolute diffusion scales. Generated by `uv run --python 3.13 python quadmath/scripts/animation_stills.py`.](../output/figures/animation_frames_strip.png)

`quadmath/scripts/animation_gallery.py` writes `animation_lattice.gif`,
`animation_simplex.gif`, and `animation_diffusion.gif`; the still-strip
script writes `animation_frames_strip.png`. Fixed seeds control numerical
inputs. Byte reproducibility requires a fixed rendering/encoder environment,
not merely the same seed across library versions.

## Reproducibility and test contract

- `gallery(paths_out_dir, seed=12)` writes exactly the three files
  `vis_gallery_shell.png`, `vis_gallery_field.png`,
  `vis_gallery_dynamics.png` (module constant `GALLERY_FILES`, in that
  order) into `paths_out_dir`, creating it when missing, and returns their
  paths. Seeds control the numerical boundary data and initial fields, not
  cross-version image encoding.
- Tests: `tests/unit/viz/test_vis_lattice.py` covers exact quotient rank,
  gauge invariance, integer divisibility, ill-conditioned bases, sparse
  gap placement and bounded storage, huge-offset local display, zero
  normalization, equal metric aspect, model-derived boundary and dynamics
  diagnostics, and preservation of caller figures during gallery rendering. Run the focused
  module with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 python -m pytest
  tests/unit/viz/test_vis_lattice.py -q` in the supported environment.
- Tests: `tests/unit/viz/test_animations.py` covers the frame module
  (`Frame` validation, the three renderers, `frames_to_gif`) and strip
  rendering: invalid labels/empty data, grayscale range, uint8/float
  equivalence, layout and fixed-toolchain encoding. Such checks do not prove
  cross-platform byte identity or physical scale preserved by a 2-D projection.
- Figure regeneration:
  `uv run --python 3.13 python quadmath/scripts/lattice_gallery.py` (the script sets
  `MPLBACKEND=Agg` itself); stdout is exactly the three written paths.
- Still-strip regeneration:
  `uv run --python 3.13 python quadmath/scripts/animation_stills.py` (the script sets
  `MPLBACKEND=Agg` itself); stdout is exactly the one written strip path,
  `quadmath/output/figures/animation_frames_strip.png`.
- Markdown: `uv run --python 3.13 python quadmath/scripts/validate_markdown.py`.

## Cross-references

- Site geometry, normalization, and the embedding:
  [Quadray Methods](03_quadray_methods.md).
- Shell enumeration and nearest-site search:
  [Lattice Tooling](13_lattice_tooling.md).
- The sliced learner: [Static IVM Field
  Learning](11_ivm_field_learning.md).
- The stripped dynamics: [Dynamics and Learning on the IVM
  Lattice](12_ivm_dynamics.md).
