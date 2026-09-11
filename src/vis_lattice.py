"""Deterministic visualization gallery for the IVM lattice methods.

This module is the figure surface for the lattice layer (``quadray``,
``omni_numbering``, ``ivm_field``, ``ivm_dynamics``).  Every plotting
primitive receives its matplotlib axes explicitly, so panels compose inside
caller-owned figures; only :func:`gallery` creates figures (three of them,
written as PNG files).  No figure is created at import time.

Pieces:

- :func:`shell_scatter` — 3D scatter of one IVM frequency shell with
  tetrahedral axis hinting.  Under ``DEFAULT_EMBEDDING`` the four quadray
  basis directions map to the columns of the embedding matrix, which are
  the vertices of a regular tetrahedron in R^3 (pairwise dot product -1).
- :func:`field_slice` — heatmap of a scalar :class:`ivm_field.IVMField`
  restricted to a lattice plane spanned by two lattice translation
  vectors ``u`` and ``v`` (see the plane parametrization in the
  docstring below).
- :func:`dynamics_strip` — row of 3D snapshots of an
  :class:`ivm_dynamics.Trajectory` rendered on one shared symmetric
  color scale so panels are directly comparable.
- :func:`gallery` — composes the three gallery figures deterministically
  (fixed file order :data:`GALLERY_FILES`) and returns their paths.

All randomness used by :func:`gallery` comes from a seeded
``numpy.random.default_rng(seed)``; the rendered PNGs carry no timestamp
metadata, so a re-run with the same seed reproduces the figures
byte-for-byte.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ivm_dynamics import DynamicsParams, Trajectory, simulate
from ivm_field import IVMField
from omni_numbering import generate_shell
from quadray import DEFAULT_EMBEDDING, Quadray, to_xyz

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d.art3d import Path3DCollection
    from matplotlib.collections import QuadMesh

__all__ = [
    "DEFAULT_PLANE",
    "GALLERY_FILES",
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
    sites: Sequence[SiteLike],
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
    - sites: Shell-``k`` sites as ``Quadray`` points or integer rows.
    - k: Shell index (non-negative); used for the default title only.
    - embedding: 3x4 embedding matrix (defaults to ``DEFAULT_EMBEDDING``).
    - color: Scatter color.
    - size: Scatter marker size.
    - axis_hints: Draw the four tetrahedral axis rays when True.
    - title: Panel title; defaults to ``"IVM shell k (n sites)"``.

    Returns
    - Path3DCollection: The scatter handle placed on ``ax``.

    Raises
    - ValueError: If ``sites`` is empty or ``k`` is negative.
    """
    if len(sites) == 0:
        raise ValueError("sites must be non-empty")
    if k < 0:
        raise ValueError(f"shell index must be non-negative, got {k}")
    quads = _as_quadray_rows(sites)
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
        title = f"IVM shell {k} ({len(quads)} sites)"
    ax.set_title(title)
    return handle


def _plane_coordinates(
    site: Quadray,
    q0: Tuple[int, ...],
    u: Tuple[int, ...],
    v: Tuple[int, ...],
) -> Optional[Tuple[int, int]]:
    """Exact integer plane coordinates of a site, or None.

    A site lies on the plane through ``q0`` spanned by the integer
    translation vectors ``u`` and ``v`` iff its quadray difference from
    ``q0`` is an integer combination of ``u``, ``v`` and the kernel
    direction ``(1,1,1,1)`` of the embedding:

    ``site - q0 = i*u + j*v + m*(1,1,1,1)`` with integers ``i, j, m``.

    The system is solved in least-squares sense over the 4x3 integer
    basis and then verified exactly; because a member has an exact
    integer solution, rounding recovers it and the exact check rejects
    every non-member.

    Parameters
    - site: Quadray site (canonical representative).
    - q0: Plane origin as an integer 4-tuple.
    - u, v: Independent integer translation vectors.

    Returns
    - Optional[Tuple[int, int]]: ``(i, j)`` plane indices, or ``None``
      when the site is off-plane.
    """
    w = tuple(s - o for s, o in zip(site.as_tuple(), q0))
    basis = np.stack(
        [np.asarray(u, dtype=float), np.asarray(v, dtype=float), np.ones(4)], axis=1
    )
    solution, _, _, _ = np.linalg.lstsq(basis, np.asarray(w, dtype=float), rcond=None)
    i = int(round(float(solution[0])))
    j = int(round(float(solution[1])))
    m = int(round(float(solution[2])))
    if all(w[n] == i * u[n] + j * v[n] + m for n in range(4)):
        return (i, j)
    return None


def _cell_edges(values: Sequence[int]) -> "np.ndarray":
    """Return pcolormesh cell edges for a sorted list of integer indices."""
    lo = float(values[0]) - 0.5
    hi = float(values[-1]) + 0.5
    return np.linspace(lo, hi, len(values) + 1)


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
) -> "QuadMesh":
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
    :func:`_plane_coordinates`.  Sites off the plane are ignored; plane
    cells outside the field's ball are rendered as masked (neutral)
    tiles.

    The heatmap axes are the integer plane indices ``i`` (steps along
    ``u``) and ``j`` (steps along ``v``).

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

    Returns
    - QuadMesh: The heatmap mesh placed on ``ax``.

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
    if np.linalg.matrix_rank(np.asarray([u, v], dtype=float)) != 2:
        raise ValueError("plane vectors u and v must be linearly independent")
    q0t = (int(q0[0]), int(q0[1]), int(q0[2]), int(q0[3]))

    cells: List[Tuple[int, int, Quadray]] = []
    for site in _as_quadray_rows(sites):
        coords = _plane_coordinates(site, q0t, u, v)
        if coords is None:
            continue
        cells.append((coords[0], coords[1], site))
    if not cells:
        raise ValueError("no lattice site from `sites` lies on the requested plane")

    i_vals = sorted({cell[0] for cell in cells})
    j_vals = sorted({cell[1] for cell in cells})
    i_pos = {val: n for n, val in enumerate(i_vals)}
    j_pos = {val: n for n, val in enumerate(j_vals)}
    grid = np.full((len(i_vals), len(j_vals)), np.nan, dtype=float)
    for i, j, site in cells:
        grid[i_pos[i], j_pos[j]] = field.predict(site)

    vmin = float(np.nanmin(grid))
    vmax = float(np.nanmax(grid))
    vmax = vmax if vmax > vmin else vmin + 1.0
    cmap_obj = mpl.colormaps[cmap].copy()
    cmap_obj.set_bad("0.88")
    mesh = ax.pcolormesh(
        _cell_edges(i_vals),
        _cell_edges(j_vals),
        np.ma.masked_invalid(grid),
        cmap=cmap_obj,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(f"i (steps along u = {list(u)})")
    ax.set_ylabel(f"j (steps along v = {list(v)})")
    if title is None:
        title = f"IVM field slice through q0 = {list(q0t)}"
    ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label="field value")
    return mesh


def dynamics_strip(
    axs: Sequence["Axes"],
    trajectory: Trajectory,
    t_indices: Sequence[int],
    *,
    embedding: Iterable[Iterable[float]] = DEFAULT_EMBEDDING,
    cmap: str = "coolwarm",
    titles: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Render evolution snapshots of a trajectory as a strip of 3D panels.

    Each selected snapshot index ``t`` is scattered over the embedded
    lattice sites of ``trajectory.lattice`` on one shared symmetric
    color scale ``[-vmax, vmax]`` with ``vmax = max |u_t|`` over the
    selected snapshots, so panels are directly comparable.

    Parameters
    - axs: One 3D axes per snapshot, owned by the caller.
    - trajectory: Simulation record from :func:`ivm_dynamics.simulate`.
    - t_indices: Snapshot indices into ``trajectory.fields``.
    - embedding: 3x4 embedding matrix (defaults to ``DEFAULT_EMBEDDING``).
    - cmap: Diverging colormap name.
    - titles: Optional per-panel titles; defaults to
      ``"{kind}, t = {t}"``.

    Returns
    - List: One scatter handle per panel, in panel order.

    Raises
    - ValueError: If ``t_indices`` is empty, lengths of ``axs`` and
      ``t_indices`` differ, or an index is outside the trajectory.
    """
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
    handles: List[Any] = []
    for n, (ax, t) in enumerate(zip(axs, t_indices)):
        handle = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=fields[t],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
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
        handles.append(handle)
    return handles


def gallery(paths_out_dir: str, seed: int = 12) -> List[str]:
    """Compose the three lattice-gallery figures deterministically.

    Panels:

    1. ``vis_gallery_shell.png`` — frequency shells 1 and 2 of the
       omnidirectional close packing (:func:`shell_scatter` with
       tetrahedral axis hints; sites from
       :func:`omni_numbering.generate_shell`).
    2. ``vis_gallery_field.png`` — a
       :class:`ivm_field.IVMField` learned from noisy observations of a
       linear synthetic truth over the radius-3 ball, sliced through the
       default lattice plane (:func:`field_slice`, plane
       :data:`DEFAULT_PLANE`).
    3. ``vis_gallery_dynamics.png`` — a heat-diffusion
       :func:`ivm_dynamics.simulate` strip over the radius-3 lattice at
       ``t = 0, 10, 20`` (:func:`dynamics_strip`).

    All randomness is drawn from one seeded generator (field observation
    mask and noise) plus the seeded dynamics run, so the figures are
    reproducible byte-for-byte for a fixed ``seed``.

    Parameters
    - paths_out_dir: Directory the three PNGs are written to (created
      when missing).
    - seed: Seed for the field observations and the dynamics run.

    Returns
    - List[str]: The three written paths, in :data:`GALLERY_FILES` order.
    """
    if not os.path.isdir(paths_out_dir):
        os.makedirs(paths_out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Figure 1: frequency shells 1 and 2 with tetrahedral axis hints.
    fig = plt.figure(figsize=(11.0, 4.6))
    for n, k in enumerate((1, 2), start=1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        shell_scatter(ax, generate_shell(k), k)
    fig.suptitle("IVM frequency shells of the omnidirectional close packing")
    fig.tight_layout()
    shell_path = os.path.join(paths_out_dir, GALLERY_FILES[0])
    fig.savefig(shell_path, dpi=160)
    plt.close(fig)

    # Figure 2: learned scalar field sliced along the default lattice plane.
    radius = 3
    field = IVMField.lattice_ball(radius)
    sites = field.sites
    coords = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in sites], dtype=float)
    truth = np.array([float(p[0]) + 0.5 * float(p[1]) for p in coords])
    mask = rng.random(len(sites)) < 0.25
    obs_sites = [q for q, keep in zip(sites, mask) if keep]
    noise = rng.normal(0.0, 0.1, size=int(mask.sum()))
    obs_values = [
        truth[n] + float(residual)
        for n, residual in zip(np.flatnonzero(mask), noise)
    ]
    field.learn(obs_sites, obs_values, lam=0.01, kernel_width=0.5)
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(1, 1, 1)
    field_slice(ax, field, sites, DEFAULT_PLANE)
    fig.tight_layout()
    field_path = os.path.join(paths_out_dir, GALLERY_FILES[1])
    fig.savefig(field_path, dpi=160)
    plt.close(fig)

    # Figure 3: heat-diffusion evolution strip on the radius-3 lattice.
    trajectory = simulate(20, DynamicsParams(kind="heat", alpha=0.45, seed=seed))
    fig = plt.figure(figsize=(12.0, 4.2))
    axs = [fig.add_subplot(1, 3, n, projection="3d") for n in (1, 2, 3)]
    dynamics_strip(axs, trajectory, (0, 10, 20))
    fig.suptitle("Heat diffusion on the IVM lattice (radius-3 ball)")
    fig.tight_layout()
    dynamics_path = os.path.join(paths_out_dir, GALLERY_FILES[2])
    fig.savefig(dynamics_path, dpi=160)
    plt.close(fig)

    return [shell_path, field_path, dynamics_path]
