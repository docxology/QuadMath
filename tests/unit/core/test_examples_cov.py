from fractions import Fraction

from quadmath.core.examples import example_optimize


def test_example_optimize_converges():
    state = example_optimize()
    # Nelder-Mead on the convex bowl must fully converge: zero simplex volume
    assert state.volume == Fraction(0)
    # And record the optimization trajectory
    assert len(state.history) >= 1
