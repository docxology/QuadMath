"""Consumer-visible geometry and resource regressions for the public abstract."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import quadmath.viz.abstracts as abstracts
from quadmath.core.quadray import DEFAULT_EMBEDDING, IVM_NEIGHBOR_MOVES
from quadmath.viz._common import save_figure


def test_rendered_basis_endpoints_and_thirteen_sphere_contacts(tmp_path, monkeypatch):
    """Inspect the actual artists passed to the real PNG/SVG writers."""
    figures = []

    def observe_real_save(fig, path, **kwargs):
        if not figures:
            figures.append(fig)
        save_figure(fig, path, **kwargs)

    monkeypatch.setattr(abstracts, "save_figure", observe_real_save)
    caller_figure = plt.figure()
    before = plt.get_fignums()
    try:
        abstracts.render_quadray_abstract(str(tmp_path / "abstract.png"))
        assert plt.get_fignums() == before
        basis, packing = figures[0].axes
        embedding = np.asarray(DEFAULT_EMBEDDING)
        arrows = {artist.get_gid(): artist for artist in basis.collections}
        for name, endpoint in zip("ABCD", embedding.T):
            # Matplotlib stores the first quiver segment tip-first. This is the
            # rendered shaft, not just the input vector handed to the renderer.
            shaft = arrows[f"quadray-basis-{name}"]._segments3d[0]
            np.testing.assert_allclose(shaft, [endpoint, np.zeros(3)])

        contact_lines = [line for line in packing.lines
                         if (line.get_gid() or "").startswith("fcc-central-contact-")]
        segments = np.asarray([line.get_data_3d() for line in contact_lines])
        assert segments.shape == (12, 3, 2)
        np.testing.assert_array_equal(segments[:, :, 0], np.zeros((12, 3)))
        neighbors = segments[:, :, 1]
        np.testing.assert_array_equal(neighbors, np.asarray(IVM_NEIGHBOR_MOVES) @ embedding.T)
        np.testing.assert_array_equal(np.sum(neighbors ** 2, axis=1), np.full(12, 8))
        assert len(np.unique(neighbors, axis=0)) == 12
        centers = np.vstack((np.zeros(3), neighbors))
        spheres = [artist for artist in packing.collections
                   if (artist.get_gid() or "").startswith("fcc-sphere-")]
        assert len(spheres) == 13
        for center, surface in zip(centers, spheres):
            # Poly3DCollection's homogeneous mesh vertices are the actual sphere
            # surface samples; every point must have the declared radius sqrt(2).
            np.testing.assert_allclose(np.sum((surface._vec[:3].T - center) ** 2, axis=1), 2)
        distances_squared = np.sum((centers[:, None] - centers[None, :]) ** 2, axis=2)
        assert np.min(distances_squared[np.triu_indices(13, 1)]) == 8
        for axis in (basis, packing):
            spans = np.asarray([np.ptp(axis.get_xlim3d()), np.ptp(axis.get_ylim3d()),
                                np.ptp(axis.get_zlim3d())])
            np.testing.assert_allclose(axis.get_box_aspect() / spans,
                                       np.full(3, axis.get_box_aspect()[0] / spans[0]))
    finally:
        plt.close(caller_figure)


@pytest.mark.parametrize("failed_format", ["png", "svg"])
def test_real_save_error_closes_only_owned_figure_and_removes_partial(tmp_path, failed_format):
    """An unwritable target exercises actual writer errors without save mocks."""
    path = tmp_path / "abstract.png"
    if failed_format == "png":
        path = tmp_path / "absent" / "abstract.png"
    else:
        (tmp_path / "abstract.svg").mkdir()
    caller_figure = plt.figure()
    before = plt.get_fignums()
    try:
        with pytest.raises(OSError):
            abstracts.render_quadray_abstract(str(path))
        assert plt.get_fignums() == before
        assert list(tmp_path.rglob("*.partial*")) == []
    finally:
        plt.close(caller_figure)


def test_non_png_target_cannot_overwrite_svg_companion(tmp_path):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        abstracts.render_quadray_abstract(str(tmp_path / "abstract.svg"))
    assert plt.get_fignums() == before
