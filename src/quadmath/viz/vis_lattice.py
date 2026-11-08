"""Deterministic visualization gallery for the IVM lattice methods.

This module is the figure surface for the lattice layer (``quadmath.core.quadray``,
``quadmath.lattice.omni_numbering``, ``quadmath.lattice.ivm_field``, ``quadmath.lattice.ivm_dynamics``).  Every plotting
primitive receives its matplotlib axes explicitly, so panels compose inside
caller-owned figures; only :func:`gallery` creates figures (three of them,
written as PNG files).  No figure is created at import time.

Pieces:

- :func:`shell_scatter` — 3D scatter of one IVM frequency shell with
  tetrahedral axis hinting.  Under ``DEFAULT_EMBEDDING`` the four quadray
  basis directions map to the columns of the embedding matrix, which are
  the vertices of a regular tetrahedron in R^3 (pairwise dot product -1).
- :func:`field_slice` — heatmap of a scalar :class:`quadmath.lattice.ivm_field.IVMField`
  restricted to a lattice plane spanned by two lattice translation
  vectors ``u`` and ``v`` (see the plane parametrization in the
  docstring below).
- :func:`dynamics_strip` — row of 3D snapshots of an
  :class:`quadmath.lattice.ivm_dynamics.Trajectory` rendered on one shared symmetric
  color scale so panels are directly comparable.
- :func:`boundary_comparison` — matched-color hard/soft fields with measured
  boundary errors and interior harmonic residuals.
- :func:`dynamics_comparison` — finite-graph degree, conservation, sum-of-squares
  and componentwise stationary-projection diagnostics for two heat operators.
- :func:`gallery` — composes the three gallery figures deterministically
  (fixed file order :data:`GALLERY_FILES`) and returns their paths.

Seeded boundary observations and initial fields make the numerical examples
reproducible. Encoded image bytes depend on the rendering toolchain; they
are not a portable mathematical contract.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np

from quadmath.lattice.ivm_dynamics import DynamicsParams, Trajectory, simulate
from quadmath.lattice.ivm_field import IVMField
from quadmath.lattice.omni_numbering import generate_shell
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, to_xyz
from quadmath.paths import get_figure_dir
from quadmath.viz._common import figure_scope, resolve_output_path, save_figure, set_axes_equal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    from matplotlib.collections import PolyCollection, QuadMesh

__all__ = [
    "DEFAULT_PLANE",
    "GALLERY_FILES",
    "boundary_comparison",
    "dynamics_comparison",
    "dynamics_strip",
    "field_slice",
    "gallery",
    "shell_scatter",
]

#: Output file names of the three gallery figures, in composition order.
GALLERY_FILES: Tuple[str, str, str] = (
    "vis_gallery_shell.png",
    "vis_gallery_field.png",
    "vis_gallery_dynamics.png",
)

#: Default lattice plane: two independent IVM translation vectors
#: (both shell-1 neighbor moves of the (2,1,1,0) family).
DEFAULT_PLANE: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]] = (
    (2, 1, 1, 0),
    (1, 2, 1, 0),
)

#: Fixed labels of the four tetrahedral axis hints (quadray basis directions).
_TETRA_LABELS: Tuple[str, str, str, str] = ("A", "B", "C", "D")

SiteLike = Union[Quadray, Sequence[int]]


def _as_quadray_rows(sites: Sequence[SiteLike]) -> List[Quadray]:
    """Coerce a site sequence to a list of :class:`Quadray` points.

    Accepts either ``Quadray`` instances or integer 4-tuples / array rows
    (as returned by :func:`omni_numbering.generate_shell`).

    Parameters
    - sites: Sequence of quadray-like rows.

    Returns
    - List[Quadray]: One point per input row, in input order.

    Raises
    - ValueError: If ``sites`` is empty.
    """
    if len(sites) == 0:
        raise ValueError("sites must be non-empty")
    out: List[Quadray] = []
    for site in sites:
        if isinstance(site, Quadray):
            out.append(site)
        else:
            out.append(Quadray(int(site[0]), int(site[1]), int(site[2]), int(site[3])))
    return out


def _tetra_directions(
    embedding: "np.ndarray",
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Return the four unit tetrahedral directions (embedding columns).

    Parameters
    - embedding: (3, 4) embedding matrix; column ``k`` is the xyz image of
      the quadray basis direction ``(1 in slot k)``.

    Returns
    - Tuple of four unit-length xyz direction arrays, one per column.
    """
    cols = []
    for k in range(4):
        col = embedding[:, k]
        cols.append(col / float(np.linalg.norm(col)))
    return (cols[0], cols[1], cols[2], cols[3])


def shell_scatter(
    ax: "Axes",
    sites: Iterable[SiteLike],
    k: int,
    *,
    embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING,
    color: str = "tab:blue",
    size: float = 18.0,
    axis_hints: bool = True,
    title: Optional[str] = None,
) -> "Path3DCollection":
    """Scatter one IVM frequency shell in 3D with tetrahedral axis hints.

    The shell sites are embedded with :func:`quadray.to_xyz` and drawn as
    a 3D point cloud on ``ax``.  With ``axis_hints`` four dashed rays from
    the origin toward the tetrahedral vertex directions (the embedding
    matrix columns) are overlaid and labeled ``A, B, C, D`` — the hint
    makes the quadray axis frame of the lattice readable in print.

    Parameters
    - ax: Matplotlib 3D axes owned by the caller.
    - sites: Shell-``k`` sites as ``Quadray`` points or integer rows; any
      iterable (including a one-shot generator) is accepted.  It is
      materialized once at entry.
    - k: Shell index (non-negative); used for the default title only.
    - embedding: 3x4 embedding matrix (defaults to ``DEFAULT_EMBEDDING``).
    - color: Scatter color.
    - size: Scatter marker size.
    - axis_hints: Draw the four tetrahedral axis rays when True.
    - title: Panel title; defaults to ``"IVM graph shell k (n sites)"``.

    Returns
    - Path3DCollection: The scatter handle placed on ``ax``.

    Raises
    - ValueError: If ``sites`` is empty or ``k`` is negative.
    """
    site_list = list(sites)
    if len(site_list) == 0:
        raise ValueError("sites must be non-empty")
    if k < 0:
        raise ValueError(f"shell index must be non-negative, got {k}")
    quads = _as_quadray_rows(site_list)
    arr = np.asarray(embedding, dtype=float)
    xyz = np.array([to_xyz(q, arr) for q in quads], dtype=float)
    handle = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], c=color, s=size, depthshade=False
    )
    if axis_hints:
        span = float(np.max(np.linalg.norm(xyz, axis=1)))
        tip = 1.25 * span if span > 0.0 else 1.0
        for j, direction in enumerate(_tetra_directions(arr)):
            ax.plot(
                [0.0, tip * direction[0]],
                [0.0, tip * direction[1]],
                [0.0, tip * direction[2]],
                ls="--",
                lw=0.8,
                color="0.55",
            )
            ax.text(
                tip * 1.08 * direction[0],
                tip * 1.08 * direction[1],
                tip * 1.08 * direction[2],
                _TETRA_LABELS[j],
                color="0.35",
                fontsize=9,
                ha="center",
                va="center",
            )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is None:
        title = f"IVM graph shell {k} ({len(quads)} sites)"
    ax.set_title(title)
    set_axes_equal(ax)
    return handle


def _plane_minor(
    u: Tuple[int, ...], v: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int]:
    """Find an exact nonzero minor after quotienting by the common offset."""
    a = tuple(u[k] - u[3] for k in range(3))
    b = tuple(v[k] - v[3] for k in range(3))
    for r, s in ((0, 1), (0, 2), (1, 2)):
        det = a[r] * b[s] - a[s] * b[r]
        if det:
            return a, b, r, s, det
    raise ValueError("plane vectors u and v must be linearly independent modulo (1,1,1,1)")


def _plane_coordinates(
    site: Quadray,
    q0: Tuple[int, ...],
    u: Tuple[int, ...],
    v: Tuple[int, ...],
) -> Optional[Tuple[int, int]]:
    """Solve integer plane membership exactly in the rank-three quotient.

    Subtracting the fourth component removes arbitrary input gauges before
    Cramer's rule. Divisibility and all three equations are checked with
    Python integers; neither large offsets nor ill-conditioning lose members.
    """
    a, b, r, s, det = _plane_minor(u, v)
    w = tuple(site.as_tuple()[k] - q0[k] for k in range(4))
    c = tuple(w[k] - w[3] for k in range(3))
    ni = c[r] * b[s] - c[s] * b[r]
    nj = a[r] * c[s] - a[s] * c[r]
    if ni % det or nj % det:
        return None
    i, j = ni // det, nj // det
    return (i, j) if all(c[k] == i * a[k] + j * b[k] for k in range(3)) else None


def _cell_edges(values: Sequence[int]) -> "np.ndarray":
    """Preserve unit sample cells and represent each empty index band once."""
    return np.array(sorted({edge for i in values for edge in (i - 0.5, i + 0.5)}))


def field_slice(
    ax: "Axes",
    field: IVMField,
    sites: Sequence[SiteLike],
    plane: Sequence[Sequence[int]],
    *,
    q0: Sequence[int] = (0, 0, 0, 0),
    cmap: str = "viridis",
    title: Optional[str] = None,
    colorbar: bool = True,
    norm: Optional["mpl.colors.Normalize"] = None,
) -> Union["QuadMesh", "PolyCollection"]:
    """Heatmap of a scalar IVM field restricted to a lattice plane.

    Plane parametrization.  With plane origin ``q0`` and independent
    integer translation vectors ``u``, ``v``, the plane points are

    ``q(i, j) = normalize(q0 + i*u + j*v)``, ``(i, j) in Z^2``,

    where ``normalize`` is the quadray canonical representative
    (:meth:`quadray.Quadray.normalize`).  Choosing ``u`` and ``v`` as
    differences of neighboring lattice sites (e.g. two of the twelve
    permutations of ``(2,1,1,0)``) keeps every ``q(i, j)`` a lattice
    site: each move has component sum 4, so the sum stays divisible by
    4 under the membership rule of :mod:`ivm_field`.  Because the
    embedding is linear and ``(1,1,1,1)`` lies in its kernel, the xyz
    image is the planar point set ``t(q0) + i*t(u) + j*t(v)``; the
    integer class shift ``(1,1,1,1)`` is invisible in xyz.  Membership
    of each candidate site is decided by the exact integer solve of
    :func:`_plane_coordinates`. Sites off the plane are ignored. Missing
    candidate cells remain neutral gaps, not interpolated observations.
    Unit-width cells keep their true index positions, including sparse bands.
    Storage is bounded linearly in candidate count: an exact integer span
    selects a masked mesh only below a fixed cell-count bound; otherwise a
    shared local unit-square path and per-sample offsets draw occupied cells.

    Axes show integer plane indices, not an orthonormal physical frame.
    For indices beyond 2**40, exact integer offsets are subtracted before
    float conversion and recorded in axis labels and ``artist.plane_index_origin``.
    ``artist.plane_indices`` retains the exact occupied integer index pairs
    in candidate order (duplicates occur only once). Sparse display offsets
    are floating approximations; a finite-pixel plot cannot resolve unit cells
    across arbitrarily large index spans, but their local paths stay unit-sized.

    Parameters
    - ax: Matplotlib 2D axes owned by the caller.
    - field: Scalar field over a lattice ball (values via
      :meth:`ivm_field.IVMField.predict`).
    - sites: Candidate lattice sites to project (typically the field's
      own ``field.sites`` ball enumeration).
    - plane: Pair ``(u, v)`` of integer 4-vectors.
    - q0: Plane origin 4-tuple (defaults to the lattice origin).
    - cmap: Colormap name (resolved from the matplotlib registry on a
      private copy; masked cells render in light gray).
    - title: Panel title; defaults to naming the plane origin.
    - colorbar: Attach a colorbar to the figure when True.
    - norm: Optional shared color normalization for cross-panel comparisons.

    Returns
    - QuadMesh or PolyCollection: Colored cells on ``ax``; the latter is
      used when the exact integer bounding span exceeds bounded mesh storage.

    Raises
    - ValueError: If a plane vector lacks 4 components, the plane
      vectors are linearly dependent, or no candidate site lies on the
      plane.
    """
    u = tuple(int(c) for c in plane[0])
    v = tuple(int(c) for c in plane[1])
    for name, vec in (("u", u), ("v", v)):
        if len(vec) != 4:
            raise ValueError(
                f"plane vector {name} must have 4 components, got {len(vec)}"
            )
    _plane_minor(u, v)
    if len(q0) != 4:
        raise ValueError("plane origin q0 must have 4 components")
    q0t = tuple(int(c) for c in q0)

    cells: List[Tuple[int, int, Quadray]] = []
    for site in _as_quadray_rows(sites):
        coords = _plane_coordinates(site, q0t, u, v)
        if coords is None:
            continue
        cells.append((coords[0], coords[1], site))
    if not cells:
        raise ValueError("no lattice site from `sites` lies on the requested plane")

    # Never convert the absolute indices to float before removing the offset.
    i_min = min(i for i, _, _ in cells)
    j_min = min(j for _, j, _ in cells)
    i_origin = i_min if max(abs(i) for i, _, _ in cells) > 2**40 else 0
    j_origin = j_min if max(abs(j) for _, j, _ in cells) > 2**40 else 0
    values = {(i - i_origin, j - j_origin): field.predict(site) for i, j, site in cells}
    i_vals = sorted({i for i, _ in values})
    j_vals = sorted({j for _, j in values})
    span_cells = (i_vals[-1] - i_vals[0] + 1) * (j_vals[-1] - j_vals[0] + 1)
    data = np.array(list(values.values()))
    if norm is None:
        lo, hi = float(data.min()), float(data.max())
        if lo == hi:
            pad = max(abs(lo) * 0.05, 1.0)
            lo, hi = lo - pad, hi + pad
        norm = mpl.colors.Normalize(lo, hi)
    cmap_obj = mpl.colormaps[cmap].copy()
    cmap_obj.set_bad("0.88")
    ax.set_facecolor("0.88")
    # Decide before any half-integer edges are rounded to float. A sparse
    # plane may span far more than 2**53 even after removing its origin.
    if span_cells <= max(4096, 8 * len(values)):
        x_edges, y_edges = _cell_edges(i_vals), _cell_edges(j_vals)
        grid = np.ma.masked_all((len(y_edges) - 1, len(x_edges) - 1))
        i_pos = {i: int(np.searchsorted(x_edges, i - 0.5)) for i in i_vals}
        j_pos = {j: int(np.searchsorted(y_edges, j - 0.5)) for j in j_vals}
        for (i, j), value in values.items():
            grid[j_pos[j], i_pos[i]] = value
        mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap_obj, norm=norm, shading="flat")
    else:
        from matplotlib.collections import PolyCollection
        from matplotlib.transforms import AffineDeltaTransform

        unit_square = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        mesh = PolyCollection(
            [unit_square], offsets=list(values), offset_transform=ax.transData,
            transform=AffineDeltaTransform(ax.transData),
            array=data, cmap=cmap_obj, norm=norm, edgecolors="none",
        )
        ax.add_collection(mesh, autolim=False)
        # Collection autoscaling sees offsets alone, omitting the unit path.
        ax.update_datalim([
            (i_vals[0] - 0.5, j_vals[0] - 0.5),
            (i_vals[-1] + 0.5, j_vals[-1] + 0.5),
        ])
        ax.autoscale_view()
    mesh.plane_index_origin = (i_origin, j_origin)
    mesh.plane_indices = tuple((i + i_origin, j + j_origin) for i, j in values)
    i_label = "i" if not i_origin else f"i - ({i_origin})"
    j_label = "j" if not j_origin else f"j - ({j_origin})"
    ax.set_xlabel(f"{i_label} (steps along u = {list(u)})")
    ax.set_ylabel(f"{j_label} (steps along v = {list(v)})")
    if title is None:
        title = f"IVM field slice through q0 = {list(q0t)}"
    ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label="field value")
    return mesh


def dynamics_strip(
    axs: Iterable["Axes"],
    trajectory: Trajectory,
    t_indices: Iterable[int],
    *,
    embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING,
    cmap: str = "coolwarm",
    titles: Optional[Sequence[str]] = None,
    norm: Optional["mpl.colors.Normalize"] = None,
) -> List[Any]:
    """Render evolution snapshots of a trajectory as a strip of 3D panels.

    Each selected snapshot index ``t`` is scattered over the embedded
    lattice sites of ``trajectory.lattice`` on one shared symmetric
    color scale ``[-vmax, vmax]`` with ``vmax = max |u_t|`` over the
    selected snapshots, so panels are directly comparable.

    Parameters
    - axs: One 3D axes per snapshot, owned by the caller; any iterable
      (including a one-shot generator) is accepted.
    - trajectory: Simulation record from :func:`ivm_dynamics.simulate`.
    - t_indices: Snapshot indices into ``trajectory.fields``; any iterable
      (including a one-shot generator) is accepted.
    - embedding: 3x4 embedding matrix (defaults to ``DEFAULT_EMBEDDING``).
    - cmap: Diverging colormap name.
    - titles: Optional per-panel titles; defaults to
      ``"{kind}, t = {t}"``.
    - norm: Shared normalization across calls; by default symmetric about
      zero, with [-1, 1] for identically zero snapshots.

    Returns
    - List: One scatter handle per panel, in panel order.

    Raises
    - ValueError: If ``t_indices`` is empty, lengths of ``axs`` and
      ``t_indices`` differ, or an index is outside the trajectory.
    """
    axs = list(axs)
    t_indices = list(t_indices)
    if len(t_indices) == 0:
        raise ValueError("t_indices must be non-empty")
    if len(axs) != len(t_indices):
        raise ValueError("axs and t_indices must have the same length")
    arr = np.asarray(embedding, dtype=float)
    xyz = np.array([to_xyz(q, arr) for q in trajectory.lattice.sites], dtype=float)
    fields = trajectory.fields
    for t in t_indices:
        if not 0 <= t < len(fields):
            raise ValueError(f"snapshot index {t} outside 0..{len(fields) - 1}")
    vmax = max(float(np.max(np.abs(fields[t]))) for t in t_indices)
    if norm is None:
        vmax = vmax or 1.0
        norm = mpl.colors.Normalize(-vmax, vmax)
    handles: List[Any] = []
    for n, (ax, t) in enumerate(zip(axs, t_indices)):
        handle = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=fields[t],
            cmap=cmap,
            norm=norm,
            s=26,
            depthshade=False,
        )
        if titles is None:
            label = f"{trajectory.params.kind}, t = {t}"
        else:
            label = titles[n]
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_axes_equal(ax)
        handles.append(handle)
    return handles


def boundary_comparison(
    axs: Sequence["Axes"],
    hard: IVMField,
    soft: IVMField,
    boundary_sites: Sequence[Quadray],
    boundary_values: Sequence[float],
    *,
    plane: Sequence[Sequence[int]] = DEFAULT_PLANE,
) -> dict[str, Any]:
    """Compare hard Dirichlet extension and soft learning on four caller axes.

    Parameters
    - axs: Two slice axes, a boundary-fit axis and an interior-residual axis.
    - hard, soft: Fitted fields on the same finite graph and site order.
    - boundary_sites, boundary_values: Common observations used by both fits.
    - plane: Integer quotient plane for the shared-color slice panels.

    Returns
    - dict: Slice artists, boundary errors and interior graph-Laplacian residuals.
    """
    if len(axs) != 4:
        raise ValueError("boundary_comparison needs four axes")
    if hard.sites != soft.sites or hard.adjacency != soft.adjacency:
        raise ValueError("comparison fields must have identical sites and adjacency")
    targets = np.asarray(boundary_values, dtype=float)
    if not boundary_sites or targets.shape != (len(boundary_sites),):
        raise ValueError("boundary sites and values must have the same nonzero length")
    boundary = np.array([hard.site_index[q.normalize()] for q in boundary_sites])
    interior = np.ones(len(hard.sites), dtype=bool)
    interior[boundary] = False
    amplitude = max(float(np.max(np.abs(f.values))) for f in (hard, soft)) or 1.0
    norm = mpl.colors.Normalize(-amplitude, amplitude)
    artists = [
        field_slice(ax, f, f.sites, plane, norm=norm, cmap="coolwarm", title=label)
        for ax, f, label in zip(axs[:2], (hard, soft), ("Hard boundary: harmonic interior", "Soft fit: regularized observations"))
    ]
    errors: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    for label, f in (("hard", hard), ("soft", soft)):
        predictions = f.values[boundary]
        errors[label] = predictions - targets
        residual = np.array([
            len(neighbors) * f.values[i] - sum(f.values[j] for j in neighbors)
            for i, neighbors in enumerate(f.adjacency)
        ])[interior]
        residuals[label] = residual
        axs[2].scatter(targets, predictions, label=label, s=16, alpha=0.7)
        axs[3].plot(np.arange(len(residual)), np.abs(residual), label=label)
    lo, hi = float(targets.min()), float(targets.max())
    axs[2].plot([lo, hi], [lo, hi], color="0.4", ls="--", label="exact boundary")
    axs[2].set(xlabel="prescribed boundary value", ylabel="fitted boundary value", title="Boundary agreement")
    axs[3].set(xlabel="interior site index", ylabel="|L f|", title="Interior harmonic residual")
    for ax in axs[2:]:
        ax.legend()
    return {"artists": artists, "boundary_errors": errors, "interior_residuals": residuals}


def _stationary_field(trajectory: Trajectory) -> np.ndarray:
    """Project the initial field onto each component's stationary subspace."""
    lattice = trajectory.lattice
    initial = np.asarray(trajectory.fields[0], dtype=float)
    if trajectory.params.alpha == 0:
        return initial.copy()
    result = np.zeros_like(initial)
    unseen = set(range(len(initial)))
    while unseen:
        seed = unseen.pop()
        component = [seed]
        for i in component:
            for j in np.flatnonzero(lattice.adjacency[i]):
                if int(j) in unseen:
                    unseen.remove(int(j))
                    component.append(int(j))
        indices = np.array(component)
        if trajectory.params.kind == "conservative":
            result[indices] = np.mean(initial[indices])
        else:
            mode = np.sqrt(lattice.degrees[indices])
            denominator = float(mode @ mode)
            if denominator:
                result[indices] = mode * float(mode @ initial[indices]) / denominator
    return result


def dynamics_comparison(
    axs: Sequence["Axes"], conservative: Trajectory, normalized: Trajectory
) -> dict[str, Any]:
    """Show finite-boundary operators and their measured invariants on six axes.

    Parameters
    - axs: Three 3D axes (degree and final fields), then three 2D axes
      (conserved quantities, recorded sum of squares, stationary residuals).
    - conservative, normalized: Same graph, initial state and snapshot count;
      kinds must be ``conservative`` and ``heat``, respectively.

    Returns
    - dict: Artists, componentwise stationary references and residual norms.
      Normalized heat preserves the sqrt(degree) projection, not ordinary mass.
    """
    if len(axs) != 6:
        raise ValueError("dynamics_comparison needs six axes")
    if conservative.params.kind != "conservative" or normalized.params.kind != "heat":
        raise ValueError("expected conservative and heat trajectories")
    lattice = conservative.lattice
    if (lattice.sites != normalized.lattice.sites
            or not np.array_equal(lattice.adjacency, normalized.lattice.adjacency)
            or len(conservative.fields) != len(normalized.fields)
            or not np.array_equal(conservative.fields[0], normalized.fields[0])):
        raise ValueError("comparison trajectories must share graph, initial state and horizon")
    xyz = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in lattice.sites])
    degree_artist = axs[0].scatter(*xyz.T, c=lattice.degrees, cmap="viridis", depthshade=False)
    axs[0].set(title="Finite-graph degree (boundary)", xlabel="x", ylabel="y", zlabel="z")
    set_axes_equal(axs[0])
    axs[0].figure.colorbar(degree_artist, ax=axs[0], label="neighbors in domain", shrink=0.6)
    amplitude = max(float(np.max(np.abs(tr.fields[-1]))) for tr in (conservative, normalized)) or 1.0
    norm = mpl.colors.Normalize(-amplitude, amplitude)
    artists = [degree_artist]
    references: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    for ax, tr, label in zip(axs[1:3], (conservative, normalized), ("conservative", "symmetric normalized")):
        artists.extend(dynamics_strip([ax], tr, [len(tr.fields) - 1], norm=norm, titles=[label + ": final field"]))
        fields = np.asarray(tr.fields, dtype=float)
        mass = fields.sum(axis=1)
        axs[3].plot(mass - mass[0], label=f"{label}: change in sum(u)")
        axs[4].plot(tr.sos, label=label)
        reference = _stationary_field(tr)
        references[label] = reference
        residuals[label] = np.linalg.norm(fields - reference, axis=1)
        axs[5].plot(residuals[label], label=label)
    weighted = np.asarray(normalized.fields) @ np.sqrt(lattice.degrees)
    axs[3].plot(weighted - weighted[0], ls="--", label="normalized: change in sqrt(d) dot u")
    axs[1].figure.colorbar(artists[1], ax=list(axs[1:3]), label="field value", shrink=0.6)
    axs[3].set(title="Conservation versus boundary bias", ylabel="change from initial value", xlabel="tick")
    axs[4].set(title="Recorded sum of squares", ylabel="sum(u²)", xlabel="tick")
    axs[5].set(title="Distance from stationary projection", ylabel="||u - stationary(u0)||₂", xlabel="tick")
    for ax in axs[3:]:
        ax.legend(fontsize=7)
    return {"artists": artists, "stationary_fields": references, "stationary_residuals": residuals}


def gallery(paths_out_dir: str, seed: int = 12) -> List[str]:
    """Compose the three lattice-gallery figures deterministically.

    Panels:

    1. ``vis_gallery_shell.png`` — frequency shells 1 and 2 of the
       omnidirectional close packing (:func:`shell_scatter` with
       tetrahedral axis hints; sites from
       :func:`omni_numbering.generate_shell`).
    2. ``vis_gallery_field.png`` — hard-boundary harmonic extension versus
       soft regularized learning on the graph-shell-radius-3 ball, with
       matched-color plane slices, boundary agreement and interior residuals.
    3. ``vis_gallery_dynamics.png`` — conservative versus symmetric-normalized
       diffusion on one Euclidean-radius-3 domain, with degree, conservation,
       recorded sum-of-squares and stationary-projection diagnostics.

    Boundary observations and the shared dynamics initial condition are
    seeded. Numerical inputs are reproducible; encoded PNG bytes are not promised.

    Parameters
    - paths_out_dir: Directory the three PNGs are written to (created
      when missing).  A bare directory name is resolved under
      ``quadmath/output/figures/``; a path with a directory component (or an
      absolute path) is used as given.
    - seed: Seed for the field observations and the dynamics run.

    Returns
    - List[str]: The three written paths, in :data:`GALLERY_FILES` order.
    """
    paths_out_dir = resolve_output_path(paths_out_dir, get_figure_dir)
    os.makedirs(paths_out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Figure 1: frequency shells 1 and 2 with tetrahedral axis hints.
    with figure_scope(figsize=(11.0, 4.6)) as fig:
        for n, k in enumerate((1, 2), start=1):
            ax = fig.add_subplot(1, 2, n, projection="3d")
            shell_scatter(ax, generate_shell(k), k)
        fig.suptitle("IVM frequency shells of the omnidirectional close packing")
        fig.tight_layout()
        shell_path = os.path.join(paths_out_dir, GALLERY_FILES[0])
        save_figure(fig, shell_path, dpi=160)

    # Figure 2: identical prescribed boundary data, distinct fitting contracts.
    radius = 3
    hard = IVMField.lattice_ball(radius)
    soft = IVMField.lattice_ball(radius)
    boundary_sites = [q for q, neighbors in zip(hard.sites, hard.adjacency) if len(neighbors) < 12]
    coords = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in boundary_sites])
    boundary_values = coords[:, 0] + 0.15 * coords[:, 1] ** 2 + rng.normal(0, 0.1, len(coords))
    hard.harmonic_extension(boundary_sites, boundary_values)
    soft.learn(boundary_sites, boundary_values, lam=0.4, kernel_width=1.5)
    with figure_scope(figsize=(12.0, 9.0)) as fig:
        axs = [fig.add_subplot(2, 2, n) for n in range(1, 5)]
        boundary_comparison(axs, hard, soft, boundary_sites, boundary_values)
        fig.suptitle(f"Hard boundary versus soft fit: graph-shell radius {radius}, {len(hard.sites)} sites")
        fig.tight_layout()
        field_path = os.path.join(paths_out_dir, GALLERY_FILES[1])
        save_figure(fig, field_path, dpi=160)

    # Figure 3: the same finite graph and seeded initial state, two operators.
    conservative = simulate(40, DynamicsParams(kind="conservative", alpha=0.45, seed=seed))
    normalized = simulate(
        40, DynamicsParams(kind="heat", alpha=0.45, seed=seed),
        lattice=conservative.lattice, u0=conservative.fields[0],
    )
    with figure_scope(figsize=(15.0, 9.0), layout="constrained") as fig:
        fig.get_layout_engine().set(h_pad=0.3, hspace=0.1)
        axs = [fig.add_subplot(2, 3, n, projection="3d" if n <= 3 else None) for n in range(1, 7)]
        dynamics_comparison(axs, conservative, normalized)
        fig.suptitle(
            f"Conservative versus symmetric-normalized diffusion: Euclidean radius "
            f"{conservative.lattice.radius}, {len(conservative.lattice.sites)} sites"
        )
        dynamics_path = os.path.join(paths_out_dir, GALLERY_FILES[2])
        save_figure(fig, dynamics_path, dpi=160)

    return [shell_path, field_path, dynamics_path]
