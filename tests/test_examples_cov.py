from examples import example_optimize


def test_example_optimize_state_volume_and_history():
    state = example_optimize()
    # Touch fields to cover dataclass structure
    assert hasattr(state, "volume") and isinstance(state.history, list)

