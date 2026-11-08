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

`frames_to_gif(frames, out_path, fps=8, scale=8)` assembles a frame sequence into an animated GIF: float frames are quantized to `uint8` gray levels by rounding \(255\,v\), each frame is upscaled by the integer factor `scale` with nearest-neighbor resampling, and the sequence is saved with per-frame duration \(1000\,/\,\texttt{fps}\) milliseconds and infinite looping.  PIL is imported lazily inside the function (Pillow stays optional at import time), and because the PIL GIF encoder is deterministic and embeds no timestamps, identical `(frames, fps, scale)` inputs render byte-identical GIF files — the same byte-stability contract the PNG gallery asserts.

`frames_strip(frames, out_path=None, labels=None, save=True)` renders the print still-counterpart of `frames_to_gif`: a single-row matplotlib strip with one grayscale `imshow` panel per `Frame`, figure width scaling with the panel count while the height stays one panel plus the title band, each panel pinned to the unit brightness interval (`vmin = 0`, `vmax = 1`; `uint8` frames are first mapped by \(v/255\), so both `Frame` dtypes share the GIF's gray-level convention), per-panel titles taken from `labels` or, by default, from each frame's own `title`, and axes switched off with zero inter-panel spacing so consecutive panels butt together.  Matplotlib is imported lazily inside the function (the module stays matplotlib-free at import time), and saving follows the `plots.py` convention: a bare `out_path` name is written into `quadmath/output/figures/` via `quadmath.paths.get_figure_dir`, a path carrying a directory component is used verbatim, and the returned string is the written path (`""` when `save` is off or no name is given).  An empty `frames` sequence or a `labels` length mismatch raises `ValueError`; the render has no RNG and no wall-clock input, so identical inputs give byte-identical PNGs — the test suite asserts the byte stability, the full \([0, 1]\) grayscale excursion, the uint8/float equivalence, and the panel-count scaling.

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
  paths.  Every panel is fully seeded, and the PNGs carry no timestamp
  metadata: the test suite asserts byte-identical re-renders for a fixed
  seed.
- Tests: `tests/unit/viz/test_vis_lattice.py` covers exact quotient rank,
  gauge invariance, integer divisibility, ill-conditioned bases, sparse
  gap placement and bounded storage, huge-offset local display, zero
  normalization, equal metric aspect, model-derived boundary and dynamics
  diagnostics, and byte-level gallery reproducibility. Run the focused
  module with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 python -m pytest
  tests/unit/viz/test_vis_lattice.py -q` in the supported environment.
- Tests: `tests/unit/viz/test_animations.py` covers the frame module
  (`Frame` validation, the three renderers, `frames_to_gif`) and the strip
  renderer — `frames_strip` label and emptiness validation, byte-identical
  re-renders in temporary directories, uint8/float equivalence, the full
  grayscale excursion, and panel-count scaling — headless via
  `MPLBACKEND=Agg` (set in `tests/conftest.py`).
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
