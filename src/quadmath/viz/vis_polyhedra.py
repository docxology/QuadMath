"""Closed convex meshes and measured manuscript figures in scale-one XYZ units.

These small integer examples are not a general mesh-analysis API; symergetics
provides that surface. Supporting planes avoid handwritten, incomplete face
lists. Labels derive from actual native tetrahedra, not a volume lookup table.
"""
from __future__ import annotations

import math
import os
from fractions import Fraction
from itertools import combinations, permutations, product
from typing import Optional

import numpy as np

from quadmath.core.examples import example_cuboctahedron_vertices_xyz
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, integer_tetra_volume, to_xyz
from quadmath.lattice.conversions import xyz_to_quadray_canonical
from quadmath.paths import get_figure_dir
from quadmath.viz._common import figure_scope, resolve_output_path, save_figure, set_axes_equal

__all__ = ["render_polyhedra_constructions"]


def _supporting_faces(vertices: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Find outward cyclic supporting faces of these bounded integer examples.

    Strictly interior samples are ignored. Coplanar face samples must be its
    extreme vertices; this helper is not an arbitrary point-cloud hull service.
    """
    planes: dict[tuple[int, ...], tuple[int, ...]] = {}
    for a, b, c in combinations(range(len(vertices)), 3):
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        if not np.any(normal):
            continue
        side = (vertices - vertices[a]) @ normal
        if np.min(side) < 0 < np.max(side):
            continue
        if np.min(side) >= 0:
            normal = -normal
        divisor = math.gcd(*(int(value) for value in normal))
        normal = normal // divisor
        offset = int(vertices[a] @ normal)
        key = (*map(int, normal), offset)
        if key in planes:
            continue
        indices = np.flatnonzero(vertices @ normal == offset).tolist()
        center = np.mean(vertices[indices], axis=0)
        axes = [axis for axis in range(3) if axis != int(np.argmax(np.abs(normal)))]
        i, j = axes
        indices.sort(key=lambda index: math.atan2(
            vertices[index, j] - center[j], vertices[index, i] - center[i],
        ))
        p, q, r = vertices[indices[:3]]
        if np.cross(q - p, r - p) @ normal < 0:
            indices.reverse()
        planes[key] = tuple(indices)
    return tuple(planes[key] for key in sorted(planes))


def _solid_meshes() -> list[tuple[str, np.ndarray, tuple[tuple[int, ...], ...]]]:
    """Six centered closed solids, with no independent display rescaling."""
    tetra = np.asarray([
        to_xyz(Quadray(*(int(axis == index) for axis in range(4))), DEFAULT_EMBEDDING)
        for index in range(4)
    ], dtype=int)
    cube = np.asarray(list(product((-1, 1), repeat=3)), dtype=int)
    octa = np.concatenate([2 * np.eye(3, dtype=int), -2 * np.eye(3, dtype=int)])
    truncated = np.asarray(sorted({
        point for s, t in product((-1, 1), repeat=2)
        for point in permutations((0, 2 * s, 4 * t))
    }), dtype=int)
    vertices = [
        ("Tetrahedron", tetra),
        ("Cube", cube),
        ("Octahedron", octa),
        ("Rhombic dodecahedron", np.concatenate([octa, cube])),
        ("Cuboctahedron", np.asarray(example_cuboctahedron_vertices_xyz(), dtype=int)),
        ("Truncated octahedron", truncated),
    ]
    return [(name, points, _supporting_faces(points)) for name, points in vertices]


def _solid_measures(
    vertices: np.ndarray, faces: tuple[tuple[int, ...], ...],
) -> tuple[Fraction, tuple[int, ...]]:
    """Measure these convex origin-containing meshes in native tetra units."""
    quadrays = [xyz_to_quadray_canonical(tuple(map(int, point))) for point in vertices]
    origin = Quadray(0, 0, 0, 0)
    volume = sum((
        integer_tetra_volume(origin, quadrays[face[0]], quadrays[face[i]], quadrays[face[i + 1]])
        for face in faces for i in range(1, len(face) - 1)
    ), Fraction())
    lengths = {
        int((vertices[a] - vertices[b]) @ (vertices[a] - vertices[b]))
        for face in faces for a, b in zip(face, (*face[1:], face[0]))
    }
    return volume, tuple(sorted(lengths))


def render_polyhedra_constructions(out_path: Optional[str] = None) -> tuple[str, str]:
    """Save measured, equal-scale solid panels and their high-resolution copy.

    Parameters
    - out_path: Destination; bare names follow the shared figure-directory
      policy. Defaults to ``polyhedra_quadray_constructions.png``. The second
      image adds ``_hq`` to the stem. Explicit parent directories must exist.

    Returns
    - tuple[str, str]: Main and high-resolution paths. All panels share XYZ
      limits, unit aspect and camera; the cube/RD do not claim the tetra edge.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    meshes = _solid_meshes()
    bound = 1.1 * max(int(np.max(np.abs(vertices))) for _, vertices, _ in meshes)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:cyan")
    with figure_scope(figsize=(12.4, 8.2), layout="constrained") as fig:
        for panel, ((name, vertices, faces), color) in enumerate(zip(meshes, colors), 1):
            ax = fig.add_subplot(2, 3, panel, projection="3d")
            ax.add_collection3d(Poly3DCollection(
                [vertices[list(face)] for face in faces], facecolor=color,
                edgecolor="0.25", linewidth=0.65, alpha=0.5,
            ))
            volume, edge_squares = _solid_measures(vertices, faces)
            ax.set_title(f"{name}\nIVM volume {volume}; edge² {', '.join(map(str, edge_squares))}")
            ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
                   xlabel="X", ylabel="Y", zlabel="Z")
            set_axes_equal(ax)
            ax.set_proj_type("ortho")
            ax.view_init(elev=22, azim=-35)
        fig.suptitle("Closed polyhedra • shared scale-one XYZ units • native volumes measured from faces")
        path = resolve_output_path(out_path or "polyhedra_quadray_constructions.png", get_figure_dir)
        stem, extension = os.path.splitext(path)
        high_resolution = f"{stem}_hq{extension}"
        save_figure(fig, path, dpi=300)
        save_figure(fig, high_resolution, dpi=600)
        return path, high_resolution
