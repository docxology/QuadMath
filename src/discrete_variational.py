from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Iterable, List, Sequence, Tuple

from quadray import Quadray


def neighbor_moves_ivm() -> List[Quadray]:
    """Return the 12 canonical IVM neighbor moves as Quadray deltas.

    The move set consists of all distinct permutations of (2, 1, 1, 0).
    Applying a move is followed by projective normalization to remain in
    the canonical non-negative representative with at least one zero.
    """
    base = (2, 1, 1, 0)
    uniq: List[Tuple[int, int, int, int]] = sorted(set(permutations(base)))
    return [Quadray(*u) for u in uniq]


def apply_move(q: Quadray, delta: Quadray) -> Quadray:
    """Apply a lattice move and normalize to the canonical representative."""
    return q.add(delta).normalize()


@dataclass
class DiscretePath:
    """Optimization trajectory on the integer quadray lattice."""

    path: List[Quadray]
    values: List[float]


def discrete_ivm_descent(
    objective: Callable[[Quadray], float],
    start: Quadray,
    *,
    moves: OptionalMoves = None,
    max_iter: int = 200,
    on_step: Callable[[Quadray, float], None] | None = None,
) -> DiscretePath:
    """Greedy discrete descent over the quadray integer lattice.

    At each iteration evaluate the objective at all neighbor moves from the
    current point (including staying put) and move to the lowest objective.
    Terminates when no improving neighbor exists or when ``max_iter`` is
    reached.

    Parameters
    - objective: Callable mapping ``Quadray`` to scalar loss.
    - start: Initial lattice point (will be normalized).
    - moves: Optional iterable of ``Quadray`` deltas to consider per step.
      Defaults to the 12 IVM neighbor moves.
    - max_iter: Hard cap on the number of move evaluations.
    - on_step: Optional callback receiving ``(q, value)`` after each accepted step.

    Returns
    - DiscretePath: Sequence of visited points (including start) and values.
    """
    q = start.normalize()
    value = float(objective(q))
    path: List[Quadray] = [q]
    values: List[float] = [value]

    candidate_moves: Sequence[Quadray] = list(moves) if moves is not None else neighbor_moves_ivm()

    for _ in range(max_iter):
        best_q = q
        best_v = value

        # Include the option to remain at the current point
        for delta in candidate_moves:
            q_next = apply_move(q, delta)
            v_next = float(objective(q_next))
            if v_next < best_v:
                best_q, best_v = q_next, v_next

        # If no improvement, terminate
        if best_v >= value:
            break

        q, value = best_q, best_v
        path.append(q)
        values.append(value)
        if on_step is not None:
            on_step(q, value)

    return DiscretePath(path=path, values=values)


# Lightweight protocol for optional typing of moves parameter
class OptionalMoves(Iterable[Quadray]):
    pass


__all__ = [
    "neighbor_moves_ivm",
    "apply_move",
    "DiscretePath",
    "discrete_ivm_descent",
]


