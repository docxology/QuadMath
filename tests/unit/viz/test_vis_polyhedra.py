"""Mathematical closure, supporting planes and native units of rendered solids."""
from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from quadmath.viz.vis_polyhedra import (
    _solid_measures, _solid_meshes, _supporting_faces, render_polyhedra_constructions,
)


@pytest.mark.parametrize(
    "mesh,counts,volume,edge_squared,face_degrees",
    [
        (0, (4, 6, 4), 1, 8, {3: 4}),
        (1, (8, 12, 6), 3, 4, {4: 6}),
        (2, (6, 12, 8), 4, 8, {3: 8}),
        (3, (14, 24, 12), 6, 3, {4: 12}),
        (4, (12, 24, 14), 20, 8, {3: 8, 4: 6}),
        (5, (24, 36, 14), 96, 8, {4: 6, 6: 8}),
    ],
)
def test_rendered_solids_are_closed_supporting_meshes_in_declared_units(
    mesh, counts, volume, edge_squared, face_degrees,
):
    _, vertices, faces = _solid_meshes()[mesh]
    edges = Counter(
        tuple(sorted((a, b)))
        for face in faces for a, b in zip(face, (*face[1:], face[0]))
    )
    assert (len(vertices), len(edges), len(faces)) == counts
    assert all(incidence == 2 for incidence in edges.values())
    assert len(vertices) - len(edges) + len(faces) == 2
    assert Counter(map(len, faces)) == face_degrees
    for face in faces:
        p, q, r = vertices[list(face[:3])]
        normal = np.cross(q - p, r - p)
        assert np.any(normal)
        assert np.all((vertices[list(face)] - p) @ normal == 0)
        assert np.all((vertices - p) @ normal <= 0)
    measured_volume, measured_edges = _solid_measures(vertices, faces)
    assert measured_volume == Fraction(volume)
    assert measured_edges == (edge_squared,)


def test_supporting_mesh_does_not_expose_strictly_interior_samples():
    cube = np.asarray([*product((-1, 1), repeat=3), (0, 0, 0)], dtype=int)
    faces = _supporting_faces(cube)
    assert all(8 not in face for face in faces)
    assert Counter(map(len, faces)) == {4: 6}
    assert _solid_measures(cube, faces) == (Fraction(3), (4,))


def test_solid_renderer_releases_its_figure_after_success(tmp_path):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    render_polyhedra_constructions(str(tmp_path / "solids.svg"))
    assert plt.get_fignums() == before


def test_solid_renderer_releases_its_figure_when_output_root_is_blocked(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("QUADMATH_OUTPUT_DIR", str(blocked))
    before = plt.get_fignums()
    with pytest.raises(FileExistsError):
        render_polyhedra_constructions()
    assert plt.get_fignums() == before
