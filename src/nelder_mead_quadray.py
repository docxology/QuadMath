from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

from quadray import Quadray, integer_tetra_volume


@dataclass
class SimplexState:
    vertices: List[Quadray]
    values: List[float]
    volume: int
    history: List[List[Quadray]]
    best_values: List[float]
    worst_values: List[float]
    spreads: List[float]
    volumes: List[int]


def order_simplex(vertices: List[Quadray], f: Callable[[Quadray], float]) -> Tuple[List[Quadray], List[float]]:
    """Sort vertices by objective value ascending and return paired lists."""
    vals = [f(v) for v in vertices]
    pairs = sorted(zip(vals, vertices), key=lambda t: t[0])
    values_sorted, verts_sorted = zip(*pairs)
    return list(verts_sorted), list(values_sorted)


def centroid_excluding(vertices: List[Quadray], exclude_idx: int) -> Quadray:
    """Integer centroid of three vertices, excluding the specified index.

    Uses floor division to stay on the integer quadray lattice.
    """
    acc = Quadray(0, 0, 0, 0)
    for i, v in enumerate(vertices):
        if i != exclude_idx:
            acc = acc.add(v)
    return Quadray(acc.a // 3, acc.b // 3, acc.c // 3, acc.d // 3)


def project_to_lattice(q: Quadray) -> Quadray:
    """Project a quadray to the canonical lattice representative via normalize."""
    return q.normalize()


def compute_volume(vertices: List[Quadray]) -> int:
    """Integer IVM tetra-volume from the first four vertices."""
    return integer_tetra_volume(vertices[0], vertices[1], vertices[2], vertices[3])


def nelder_mead_quadray(
    f: Callable[[Quadray], float],
    initial_vertices: List[Quadray],
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
    on_step: Optional[Callable[[List[Quadray]], None]] = None,
) -> SimplexState:
    """Nelder–Mead on the integer quadray lattice.

    The algorithm mirrors the continuous simplex method but snaps proposed points
    back to the lattice using `normalize()`. The history of simplex vertices is
    returned for downstream visualization.

    Parameters mirror the classical Nelder–Mead coefficients and stop criteria.
    """
    assert len(initial_vertices) == 4, "Need 4 vertices for 4D (tetrahedron)"
    vertices, values = order_simplex(initial_vertices, f)
    history: List[List[Quadray]] = [list(vertices)]
    # Per-iteration diagnostics
    best_values: List[float] = [values[0]]
    worst_values: List[float] = [values[-1]]
    spreads: List[float] = [values[-1] - values[0]]
    volumes: List[int] = [compute_volume(vertices)]

    for _ in range(max_iter):
        vertices, values = order_simplex(vertices, f)
        vol = compute_volume(vertices)
        spread = max(values) - min(values)
        # Record diagnostics at the start of the iteration (after ordering)
        best_values.append(values[0])
        worst_values.append(values[-1])
        spreads.append(spread)
        volumes.append(vol)
        if vol == 0 and spread < tol:
            return SimplexState(vertices, values, vol, history, best_values, worst_values, spreads, volumes)

        centroid = centroid_excluding(vertices, 3)
        worst = vertices[3]

        vr = Quadray(
            centroid.a + int(alpha * (centroid.a - worst.a)),
            centroid.b + int(alpha * (centroid.b - worst.b)),
            centroid.c + int(alpha * (centroid.c - worst.c)),
            centroid.d + int(alpha * (centroid.d - worst.d)),
        )
        vr = project_to_lattice(vr)
        fr = f(vr)

        if fr < values[0]:
            ve = Quadray(
                centroid.a + int(gamma * (vr.a - centroid.a)),
                centroid.b + int(gamma * (vr.b - centroid.b)),
                centroid.c + int(gamma * (vr.c - centroid.c)),
                centroid.d + int(gamma * (vr.d - centroid.d)),
            )
            ve = project_to_lattice(ve)
            vertices[3] = ve if f(ve) < fr else vr
        elif fr < values[2]:
            vertices[3] = vr
        else:
            vc = Quadray(
                centroid.a + int(rho * (worst.a - centroid.a)),
                centroid.b + int(rho * (worst.b - centroid.b)),
                centroid.c + int(rho * (worst.c - centroid.c)),
                centroid.d + int(rho * (worst.d - centroid.d)),
            )
            vc = project_to_lattice(vc)
            if f(vc) < values[3]:
                vertices[3] = vc
            else:
                best = vertices[0]
                for i in range(1, 4):
                    vs = Quadray(
                        best.a + int(sigma * (vertices[i].a - best.a)),
                        best.b + int(sigma * (vertices[i].b - best.b)),
                        best.c + int(sigma * (vertices[i].c - best.c)),
                        best.d + int(sigma * (vertices[i].d - best.d)),
                    )
                    vertices[i] = project_to_lattice(vs)

        history.append(list(vertices))
        if on_step:
            on_step(list(vertices))

    vertices, values = order_simplex(vertices, f)
    final_vol = compute_volume(vertices)
    # Ensure last diagnostics reflect final ordered state
    if volumes[-1] != final_vol or best_values[-1] != values[0] or worst_values[-1] != values[-1]:
        volumes.append(final_vol)
        best_values.append(values[0])
        worst_values.append(values[-1])
        spreads.append(values[-1] - values[0])
        history.append(list(vertices))
    return SimplexState(vertices, values, final_vol, history, best_values, worst_values, spreads, volumes)
