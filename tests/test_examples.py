from examples import (
    example_ivm_neighbors,
    example_volume,
    example_optimize,
    example_partition_tetra_volume,
    example_cuboctahedron_neighbors,
    example_cuboctahedron_vertices_xyz,
)


def test_examples_ivm_neighbors_and_volume():
    pts = example_ivm_neighbors()
    assert len(pts) == 12
    assert example_volume() == 1


def test_example_optimize_runs():
    state = example_optimize()
    assert hasattr(state, "vertices") and len(state.vertices) == 4


def test_example_partition_tetra_volume_basic():
    mu = (2, 1, 1, 0)
    s = (1, 2, 1, 0)
    a = (1, 1, 2, 0)
    psi = (2, 2, 1, 1)
    V = example_partition_tetra_volume(mu, s, a, psi)
    assert isinstance(V, int)
    assert V >= 0


def test_cuboctahedron_neighbors_and_xyz():
    neis = example_cuboctahedron_neighbors()
    assert len(neis) == 12
    xyz = example_cuboctahedron_vertices_xyz()
    assert len(xyz) == 12
    # All neighbors lie at the same radius under the default embedding
    import math

    r = None
    for x, y, z in xyz:
        cur = math.sqrt(x * x + y * y + z * z)
        if r is None:
            r = cur
        else:
            assert abs(cur - r) < 1e-9
