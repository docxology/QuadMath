import numpy as np

from quadray import Quadray
from discrete_variational import neighbor_moves_ivm, apply_move, discrete_ivm_descent


def test_neighbor_moves_ivm_has_12_unique():
    moves = neighbor_moves_ivm()
    assert len(moves) == 12
    # uniqueness by tuple
    tuples = {m.as_tuple() for m in moves}
    assert len(tuples) == 12


def test_apply_move_normalizes():
    q = Quadray(3, 3, 3, 3)
    delta = Quadray(2, 1, 1, 0)
    q2 = apply_move(q, delta)
    # After adding, normalization should subtract min to produce at least one zero
    assert min(q2.as_tuple()) == 0


def test_discrete_ivm_descent_monotone():
    # Quadratic bowl in abc-only coordinates; d influences via normalization
    def f(q: Quadray) -> float:
        a, b, c, d = q.as_tuple()
        return float((a - 1) ** 2 + (b - 2) ** 2 + (c - 0) ** 2 + 0.1 * d)

    start = Quadray(10, 0, 0, 0)
    path = discrete_ivm_descent(f, start, max_iter=50)
    assert len(path.path) >= 1
    # values strictly decrease along accepted steps
    diffs = np.diff(np.array(path.values))
    assert np.all(diffs < 1e-12)


def test_discrete_ivm_descent_on_step_callback():
    # Verify callback is invoked and receives matching points/values
    seq = []

    def f(q: Quadray) -> float:
        return float((q.a - 1) ** 2 + (q.b - 1) ** 2 + (q.c) ** 2)

    def cb(q: Quadray, v: float) -> None:
        seq.append((q.as_tuple(), v))

    _ = discrete_ivm_descent(f, Quadray(4, 0, 0, 0), max_iter=10, on_step=cb)
    # Ensure callback fired at least once and values correspond to f(q)
    assert len(seq) >= 1
    for qt, val in seq:
        q = Quadray(*qt)
        assert abs(f(q) - val) < 1e-9


def test_discrete_ivm_descent_custom_moves_breaks_when_no_improvement():
    # Objective increases with any positive move from origin
    def f(q: Quadray) -> float:
        a, b, c, d = q.as_tuple()
        return float(a + b + c + d)

    start = Quadray(0, 0, 0, 0)
    # Restrict moves to a single positive neighbor, guaranteeing no improvement
    moves = [Quadray(2, 1, 1, 0)]
    path = discrete_ivm_descent(f, start, moves=moves, max_iter=5)
    # Should immediately stop (no improving neighbor)
    assert len(path.path) == 1 and len(path.values) == 1


