"""Lattice geometry, field semantics and caller-resource regressions."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quadmath.lattice.ivm_dynamics import DynamicsParams, simulate
from quadmath.lattice.ivm_field import IVMField, quadray_shell_norm
from quadmath.lattice.omni_numbering import generate_shell
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, to_xyz
from quadmath.viz.vis_lattice import (
    DEFAULT_PLANE,
    boundary_comparison,
    dynamics_comparison,
    dynamics_strip,
    field_slice,
    gallery,
    shell_scatter,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close only figures owned by this test, not ambient caller figures."""
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


# ---------------------------------------------------------------------------
# shell_scatter
# ---------------------------------------------------------------------------


def test_shell_scatter_places_artists_and_axis_hints():
    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    handle = shell_scatter(ax, generate_shell(2), 2)
    assert handle.axes is ax
    assert len(ax.collections) == 1
    assert len(ax.get_lines()) == 4  # one dashed ray per tetrahedral direction
    assert len(ax.texts) == 4  # labels A, B, C, D
    assert ax.get_title() == "IVM graph shell 2 (42 sites)"
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y" and ax.get_zlabel() == "z"


def test_shell_scatter_accepts_quadray_rows_without_hints():
    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    sites = [Quadray(2, 1, 1, 0), Quadray(1, 2, 1, 0), Quadray(1, 1, 2, 0)]
    handle = shell_scatter(
        ax, sites, 1, axis_hints=False, color="tab:orange", title="custom title"
    )
    assert handle.axes is ax
    assert len(ax.collections) == 1
    assert len(ax.get_lines()) == 0
    assert len(ax.texts) == 0
    assert ax.get_title() == "custom title"


def test_shell_scatter_creates_no_figure():
    before = plt.get_fignums()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    shell_scatter(ax, generate_shell(1), 1)
    assert plt.get_fignums() == before + [fig.number]


def test_shell_scatter_rejects_empty_sites():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(ValueError, match="sites must be non-empty"):
        shell_scatter(ax, [], 1)


def test_shell_scatter_rejects_negative_shell():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(ValueError, match="shell index must be non-negative"):
        shell_scatter(ax, generate_shell(1), -1)


# ---------------------------------------------------------------------------
# field_slice
# ---------------------------------------------------------------------------


def _shell_norm_field(radius: int) -> IVMField:
    """Scalar field whose value at each site is half the shell norm (= shell k)."""
    field = IVMField.lattice_ball(radius)
    field.values = np.array(
        [quadray_shell_norm(q) / 2.0 for q in field.sites], dtype=float
    )
    return field


def test_field_slice_values_match_shell_norms_on_plane():
    field = _shell_norm_field(2)
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111)
    mesh = field_slice(ax, field, field.sites, DEFAULT_PLANE)
    assert mesh.axes is ax
    data = np.ma.filled(np.ma.asarray(mesh.get_array(), dtype=float), np.nan)
    assert data.shape == (5, 5)
    # Exact lattice facts on the default plane within the radius-2 ball:
    # the origin alone (shell 0), the six in-plane shell-1 neighbors, and
    # twelve shell-2 sites; the remaining 6 cells lie outside the ball.
    assert np.sum(data == 0.0) == 1
    assert np.sum(data == 1.0) == 6
    assert np.sum(data == 2.0) == 12
    assert np.isnan(data).sum() == 6
    assert data[2, 2] == 0.0  # the origin sits at plane indices (0, 0)
    assert ax.get_xlabel().startswith("i (steps along u = [2, 1, 1, 0])")
    assert ax.get_ylabel().startswith("j (steps along v = [1, 2, 1, 0])")
    assert len(fig.axes) == 2  # main axes + colorbar
    assert ax.get_title() == "IVM field slice through q0 = [0, 0, 0, 0]"


def _plane_index_field(radius: int, plane) -> IVMField:
    """Field whose value at a site is ``10 * i + j`` for its plane indices.

    Indices are solved in xyz space (origin q0 = 0), independently of the
    integer solve used by ``field_slice``.
    """
    basis = np.array(
        [to_xyz(Quadray(*plane[0]), DEFAULT_EMBEDDING), to_xyz(Quadray(*plane[1]), DEFAULT_EMBEDDING)]
    ).T
    field = IVMField.lattice_ball(radius)
    values = []
    for site in field.sites:
        i, j = np.linalg.lstsq(basis, np.array(to_xyz(site, DEFAULT_EMBEDDING)), rcond=None)[0]
        values.append(10.0 * round(i) + round(j))
    field.values = np.array(values, dtype=float)
    return field


def _cell_centers_and_values(mesh):
    """Cell centers (x, y) and values of a pcolormesh, as (ny, nx) arrays."""
    corners = mesh.get_coordinates()
    centers = 0.5 * (corners[:-1, :-1] + corners[1:, 1:])
    values = np.ma.filled(np.ma.asarray(mesh.get_array(), dtype=float).reshape(centers.shape[:2]), np.nan)
    return centers, values


@pytest.mark.parametrize(
    "plane",
    [DEFAULT_PLANE, ((2, 1, 1, 0), (4, 0, 0, 0))],
    ids=["square-default-plane", "non-square-plane"],
)
def test_field_slice_cells_land_at_their_plane_coordinates(plane):
    field = _plane_index_field(2, plane)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mesh = field_slice(ax, field, field.sites, plane, colorbar=False)
    centers, values = _cell_centers_and_values(mesh)
    filled = np.argwhere(np.isfinite(values))
    assert len(filled) > 0
    for row, col in filled:
        x, y = centers[row, col]
        assert values[row, col] == 10 * round(x) + round(y)


def test_field_slice_non_square_plane_renders():
    field = _plane_index_field(2, ((2, 1, 1, 0), (4, 0, 0, 0)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mesh = field_slice(ax, field, field.sites, ((2, 1, 1, 0), (4, 0, 0, 0)), colorbar=False)
    centers, _ = _cell_centers_and_values(mesh)
    x_positions = np.unique(np.round(centers[..., 0]))
    y_positions = np.unique(np.round(centers[..., 1]))
    assert len(x_positions) != len(y_positions)


def test_field_slice_colorbar_off_and_custom_title():
    field = _shell_norm_field(2)
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111)
    mesh = field_slice(
        ax, field, field.sites, DEFAULT_PLANE, colorbar=False, title="plane cut"
    )
    assert mesh.axes is ax
    assert len(fig.axes) == 1
    assert ax.get_title() == "plane cut"


def test_field_slice_shifted_origin_relocates_grid():
    field = _shell_norm_field(2)
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111)
    # Plane origin at the in-plane shell-1 site u = (2, 1, 1, 0): the lattice
    # origin then sits at plane index (-1, 0), so the whole grid shifts by
    # one step along u. Same 19 in-ball cells as the q0 = 0 slice, relabeled.
    mesh = field_slice(
        ax, field, field.sites, DEFAULT_PLANE, q0=(2, 1, 1, 0), colorbar=False
    )
    data = np.ma.filled(np.ma.asarray(mesh.get_array(), dtype=float), np.nan)
    assert data.shape == (5, 5)
    assert data[2, 2] == 0.0  # lattice origin at plane indices (-1, 0)
    assert np.sum(data == 0.0) == 1
    assert np.sum(data == 1.0) == 6
    assert np.sum(data == 2.0) == 12
    assert np.isnan(data).sum() == 6


def test_field_slice_rejects_degenerate_plane_vectors():
    field = _shell_norm_field(2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="must have 4 components"):
        field_slice(ax, field, field.sites, ((2, 1, 1), (1, 2, 1, 0)))
    with pytest.raises(ValueError, match="must have 4 components"):
        field_slice(ax, field, field.sites, ((2, 1, 1, 0), (1, 2, 1)))
    with pytest.raises(ValueError, match="linearly independent"):
        field_slice(ax, field, field.sites, ((2, 1, 1, 0), (2, 1, 1, 0)))


def test_field_slice_rejects_plane_without_sites():
    field = _shell_norm_field(2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # For every radius-2 site the third coordinate equation of the plane
    # system with q0 = (1, 2, 3, 4) is inconsistent, so the plane is empty.
    with pytest.raises(ValueError, match="no lattice site"):
        field_slice(ax, field, field.sites, DEFAULT_PLANE, q0=(1, 2, 3, 4))


def test_field_slice_rejects_empty_candidates():
    field = _shell_norm_field(2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="sites must be non-empty"):
        field_slice(ax, field, [], DEFAULT_PLANE)


def test_field_slice_sparse_indices_preserve_samples_and_mask_gaps():
    field = _shell_norm_field(3)
    fig, ax = plt.subplots()
    mesh = field_slice(
        ax, field, [Quadray(0, 0, 0, 0), Quadray(6, 3, 3, 0)],
        DEFAULT_PLANE, colorbar=False,
    )
    centers, values = _cell_centers_and_values(mesh)
    np.testing.assert_array_equal(mesh.get_coordinates()[0, :, 0], [-0.5, 0.5, 2.5, 3.5])
    np.testing.assert_array_equal(centers[0, [0, 2], 0], [0, 3])
    np.testing.assert_array_equal(values[0, [0, 2]], [0, 3])
    assert np.isnan(values[0, 1])


def test_field_slice_gauge_and_large_in_plane_origins_keep_unit_cells():
    field = _shell_norm_field(2)
    n = 2**80 + 1
    gauge = (n,) * 4
    u, v = DEFAULT_PLANE
    gauged_plane = tuple(tuple(c + n for c in vec) for vec in (u, v))
    fig, ax = plt.subplots()
    gauged = field_slice(ax, field, field.sites, gauged_plane, q0=gauge, colorbar=False)
    assert np.ma.count(gauged.get_array()) == 19
    assert gauged.plane_index_origin == (0, 0)
    fig, ax = plt.subplots()
    shifted = field_slice(
        ax, field, field.sites, DEFAULT_PLANE,
        q0=tuple(n * c for c in u), colorbar=False,
    )
    centers, values = _cell_centers_and_values(shifted)
    assert shifted.plane_index_origin == (-n - 2, 0)
    assert values[2, 2] == 0
    assert centers[2, 2, 0] == 2
    assert str(-n - 2) in ax.get_xlabel()
    assert np.all(np.diff(shifted.get_coordinates()[0, :, 0]) == 1)


@pytest.mark.parametrize("v", [(3, 2, 2, 1), (1, 1, 1, 1)])
def test_field_slice_rejects_quotient_degenerate_vectors(v):
    field = _shell_norm_field(1)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="independent modulo"):
        field_slice(ax, field, field.sites, (DEFAULT_PLANE[0], v))


def test_field_slice_two_samples_survive_ill_conditioned_basis():
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    field = _shell_norm_field(1)
    u, v = DEFAULT_PLANE
    n = 2**60 + 1
    skew = tuple(n * a + b for a, b in zip(u, v))
    fig, ax = plt.subplots()
    # v has indices (-n, 1), while origin has (0, 0). Recentring a
    # single sample would hide the unresolvable half-integer edge at n.
    shared_norm = Normalize(-1, 2)
    mesh = field_slice(
        ax, field, [Quadray(0, 0, 0, 0), Quadray(*v)], (u, skew),
        colorbar=False, norm=shared_norm,
    )
    assert isinstance(mesh, PolyCollection)
    assert mesh.plane_index_origin == (-n, 0)
    assert mesh.plane_indices == ((0, 0), (-n, 1))
    assert mesh.plane_indices[0][0] - mesh.plane_indices[1][0] == n
    np.testing.assert_array_equal(mesh.get_offsets(), [[float(n), 0], [0, 1]])
    np.testing.assert_array_equal(mesh.get_array(), [0, 1])
    assert mesh.norm is shared_norm
    # One unit-square path is reused only at the two observed cells; there
    # is no filled Cartesian product or interpolated band between them.
    assert len(mesh.get_paths()) == 1
    np.testing.assert_array_equal(
        mesh.get_paths()[0].vertices[:4],
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
    )
    fig.canvas.draw()
    np.testing.assert_allclose(
        mesh.get_facecolors(), mesh.cmap(shared_norm([0, 1])),
    )


def test_field_slice_requires_integer_membership_and_four_component_origin():
    field = _shell_norm_field(1)
    u, v = DEFAULT_PLANE
    fig, ax = plt.subplots()
    doubled = (tuple(2 * c for c in u), v)
    mesh = field_slice(
        ax, field, [Quadray(0, 0, 0, 0), Quadray(*u)], doubled, colorbar=False,
    )
    # Geometrically coplanar is insufficient: integer-index divisibility matters.
    np.testing.assert_array_equal(mesh.get_array().compressed(), [0])
    assert mesh.plane_indices == ((0, 0),)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="q0 must have 4 components"):
        field_slice(ax, field, field.sites, DEFAULT_PLANE, q0=(0, 0, 0))


def _sparse_field(indices):
    u, v = DEFAULT_PLANE
    sites = tuple(
        Quadray(*(i * a + j * b for a, b in zip(u, v))).normalize()
        for i, j in indices
    )
    return IVMField(
        radius=0, sites=sites, site_index={q: n for n, q in enumerate(sites)},
        values=np.arange(len(sites), dtype=float),
    )


@pytest.mark.parametrize("separation", [10**9, 2**60 + 1])
def test_field_slice_huge_empty_band_uses_only_observed_unit_cells(separation):
    from matplotlib.collections import PolyCollection

    indices = [(0, 0), (separation, 0)]
    field = _sparse_field(indices)
    fig, ax = plt.subplots()
    mesh = field_slice(ax, field, field.sites, DEFAULT_PLANE, colorbar=False)
    assert isinstance(mesh, PolyCollection)
    assert mesh.plane_indices == tuple(indices)
    assert mesh.plane_index_origin == (0, 0)
    np.testing.assert_array_equal(mesh.get_array(), [0, 1])
    np.testing.assert_array_equal(mesh.get_offsets(), [[0, 0], [float(separation), 0]])
    assert len(mesh.get_paths()) == 1
    np.testing.assert_array_equal(np.ptp(mesh.get_paths()[0].vertices, axis=0), [1, 1])
    assert ax.get_xlim()[0] <= -0.5
    assert ax.get_xlim()[1] >= float(separation)
    assert ax.get_ylim()[0] <= -0.5 and ax.get_ylim()[1] >= 0.5
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_facecolors(), mesh.cmap(mesh.norm([0, 1])))


def test_field_slice_sparse_product_uses_only_occupied_polygons():
    from matplotlib.collections import PolyCollection

    indices = [(3 * i, 3 * i) for i in range(100)]
    field = _sparse_field(indices)
    fig, ax = plt.subplots()
    artist = field_slice(ax, field, field.sites, DEFAULT_PLANE, colorbar=False)
    assert isinstance(artist, PolyCollection)
    assert len(artist.get_paths()) == 1
    np.testing.assert_array_equal(artist.get_offsets(), indices)
    assert artist.plane_indices == tuple(indices)
    path = artist.get_paths()[0]
    np.testing.assert_array_equal(path.vertices[:4].mean(axis=0), [0, 0])
    np.testing.assert_array_equal(np.ptp(path.vertices, axis=0), [1, 1])
    # The unit path transforms as a displacement, not another translated
    # point: zooming/panning must not add the axes origin twice.
    ax.set_xlim(100, 110)
    ax.set_ylim(100, 110)
    fig.canvas.draw()
    expected_delta = (
        ax.transData.transform(path.vertices) - ax.transData.transform((0, 0))
    )
    np.testing.assert_allclose(artist.get_transform().transform(path.vertices), expected_delta)
    np.testing.assert_allclose(
        artist.get_offset_transform().transform(artist.get_offsets()),
        ax.transData.transform(indices),
    )
    np.testing.assert_array_equal(artist.get_array(), np.arange(100))


def test_zero_field_slice_and_dynamics_use_colormap_midpoint_and_metric_aspect():
    field = IVMField.lattice_ball(1)
    fig, ax = plt.subplots()
    mesh = field_slice(ax, field, field.sites, DEFAULT_PLANE, colorbar=False)
    assert mesh.norm(0) == 0.5
    trajectory = simulate(2, DynamicsParams(), u0=np.zeros(13))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    artist = dynamics_strip([ax], trajectory, [2])[0]
    assert artist.norm(0) == 0.5
    spans = np.array([hi - lo for lo, hi in [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]])
    aspect = ax.get_box_aspect()
    np.testing.assert_allclose(aspect / spans, np.repeat(aspect[0] / spans[0], 3))


def test_shell_scatter_equal_metric_aspect_fits_axis_hints():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    shell_scatter(ax, generate_shell(1), 1)
    limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    spans = np.array([hi - lo for lo, hi in limits])
    aspect = ax.get_box_aspect()
    np.testing.assert_allclose(aspect / spans, np.repeat(aspect[0] / spans[0], 3))
    for line in ax.lines:
        for values, (lo, hi) in zip(line.get_data_3d(), limits):
            assert lo <= min(values) <= max(values) <= hi


# ---------------------------------------------------------------------------
# dynamics_strip
# ---------------------------------------------------------------------------


def test_dynamics_strip_default_titles_and_handles():
    trajectory = simulate(8, DynamicsParams(kind="heat", alpha=0.5, seed=12))
    fig = plt.figure(figsize=(9.0, 3.4))
    axs = [fig.add_subplot(1, 3, n, projection="3d") for n in (1, 2, 3)]
    handles = dynamics_strip(axs, trajectory, (0, 4, 8))
    assert len(handles) == 3
    for handle, ax in zip(handles, axs):
        assert handle.axes is ax
        assert len(ax.collections) == 1
    assert axs[0].get_title() == "heat, t = 0"
    assert axs[1].get_title() == "heat, t = 4"
    assert axs[2].get_title() == "heat, t = 8"


def test_dynamics_strip_custom_titles():
    trajectory = simulate(4, DynamicsParams(kind="majority", alpha=0.4, seed=12))
    fig = plt.figure(figsize=(9.0, 3.4))
    axs = [fig.add_subplot(1, 2, n, projection="3d") for n in (1, 2)]
    dynamics_strip(axs, trajectory, (0, 4), titles=("start", "end"))
    assert axs[0].get_title() == "start"
    assert axs[1].get_title() == "end"


def test_dynamics_strip_rejects_empty_indices():
    trajectory = simulate(2, DynamicsParams(kind="heat", alpha=0.5, seed=12))
    with pytest.raises(ValueError, match="t_indices must be non-empty"):
        dynamics_strip([], trajectory, [])


def test_dynamics_strip_rejects_length_mismatch():
    trajectory = simulate(2, DynamicsParams(kind="heat", alpha=0.5, seed=12))
    fig = plt.figure(figsize=(6.0, 3.0))
    axs = [fig.add_subplot(1, 2, n, projection="3d") for n in (1, 2)]
    with pytest.raises(ValueError, match="same length"):
        dynamics_strip(axs, trajectory, (0, 1, 2))


def test_dynamics_strip_rejects_out_of_range_index():
    trajectory = simulate(2, DynamicsParams(kind="heat", alpha=0.5, seed=12))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(ValueError, match="outside 0..2"):
        dynamics_strip([ax], trajectory, (4,))


def test_dynamics_strip_accepts_generator_inputs():
    trajectory = simulate(4, DynamicsParams(kind="heat", alpha=0.5, seed=12))
    fig = plt.figure(figsize=(6.0, 3.0))
    axs = [fig.add_subplot(1, 2, n, projection="3d") for n in (1, 2)]
    handles = dynamics_strip((ax for ax in axs), trajectory, (t for t in (0, 4)))
    assert len(handles) == 2
    assert axs[0].get_title() == "heat, t = 0"
    assert axs[1].get_title() == "heat, t = 4"


def test_shell_scatter_accepts_generator_of_sites():
    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    shell_scatter(ax, (site for site in generate_shell(2)), 2)
    assert ax.get_title() == "IVM graph shell 2 (42 sites)"
    assert len(ax.get_lines()) == 4

def test_boundary_comparison_plots_actual_boundary_errors_and_harmonic_residuals():
    hard = IVMField.lattice_ball(2)
    soft = IVMField.lattice_ball(2)
    boundary = [q for q, neighbors in zip(hard.sites, hard.adjacency) if len(neighbors) < 12]
    coords = np.array([to_xyz(q, DEFAULT_EMBEDDING) for q in boundary])
    targets = coords[:, 0] + 0.2 * coords[:, 1] ** 2
    hard.harmonic_extension(boundary, targets)
    soft.learn(boundary, targets, lam=0.4, kernel_width=1.5)
    fig, axs = plt.subplots(2, 2)
    result = boundary_comparison(list(axs.flat), hard, soft, boundary, targets)
    np.testing.assert_allclose(result["boundary_errors"]["hard"], 0, atol=1e-12)
    np.testing.assert_allclose(result["interior_residuals"]["hard"], 0, atol=1e-12)
    assert np.max(np.abs(result["boundary_errors"]["soft"])) > 0.01
    assert np.max(np.abs(result["interior_residuals"]["soft"])) > 0.01
    assert result["artists"][0].norm is result["artists"][1].norm
    np.testing.assert_allclose(axs[1, 1].lines[0].get_ydata(), np.abs(result["interior_residuals"]["hard"]))
    np.testing.assert_allclose(axs[1, 0].collections[0].get_offsets()[:, 0], targets)
    assert targets.min() <= hard.values.min() <= hard.values.max() <= targets.max()
    with pytest.raises(ValueError, match="four axes"):
        boundary_comparison([], hard, soft, boundary, targets)
    with pytest.raises(ValueError, match="identical sites"):
        boundary_comparison(list(axs.flat), hard, IVMField.lattice_ball(1), boundary, targets)
    with pytest.raises(ValueError, match="same nonzero length"):
        boundary_comparison(list(axs.flat), hard, soft, [], [])


def test_dynamics_comparison_reveals_boundary_stationary_mode_and_invariants():
    conservative = simulate(120, DynamicsParams(kind="conservative", alpha=0.45), u0=np.ones(13))
    normalized = simulate(
        120, DynamicsParams(kind="heat", alpha=0.45),
        lattice=conservative.lattice, u0=conservative.fields[0],
    )
    fig = plt.figure()
    axs = [fig.add_subplot(2, 3, n, projection="3d" if n < 4 else None) for n in range(1, 7)]
    result = dynamics_comparison(axs, conservative, normalized)
    np.testing.assert_allclose(conservative.fields[-1], 1, atol=1e-13)
    stationary = result["stationary_fields"]["symmetric normalized"]
    mode = np.sqrt(conservative.lattice.degrees)
    np.testing.assert_allclose(stationary, mode * (mode @ np.ones(13)) / (mode @ mode))
    assert stationary.max() - stationary.min() > 0.1
    np.testing.assert_allclose(normalized.fields[-1], stationary, atol=1e-12)
    np.testing.assert_allclose(axs[3].lines[0].get_ydata(), 0, atol=1e-12)
    assert abs(axs[3].lines[1].get_ydata()[-1]) > 0.1
    np.testing.assert_allclose(axs[3].lines[2].get_ydata(), 0, atol=1e-12)
    np.testing.assert_array_equal(axs[4].lines[0].get_ydata(), conservative.sos)
    np.testing.assert_array_equal(result["artists"][0].get_array(), conservative.lattice.degrees)
    assert result["artists"][1].norm is result["artists"][2].norm
    with pytest.raises(ValueError, match="six axes"):
        dynamics_comparison([], conservative, normalized)
    with pytest.raises(ValueError, match="expected conservative and heat"):
        dynamics_comparison(axs, normalized, conservative)
    with pytest.raises(ValueError, match="share graph"):
        dynamics_comparison(axs, conservative, simulate(1, DynamicsParams()))


@pytest.mark.parametrize("alpha", [0.0, 0.35])
def test_dynamics_comparison_componentwise_equilibrium_including_isolated_sites(alpha):
    from quadmath.lattice.ivm_dynamics import make_lattice

    lattice = make_lattice(1)
    initial = np.arange(lattice.size, dtype=float) + 1
    conservative = simulate(3, DynamicsParams(kind="conservative", alpha=alpha), lattice=lattice, u0=initial)
    normalized = simulate(3, DynamicsParams(kind="heat", alpha=alpha), lattice=lattice, u0=initial)
    fig = plt.figure()
    axs = [fig.add_subplot(2, 3, n, projection="3d" if n < 4 else None) for n in range(1, 7)]
    result = dynamics_comparison(axs, conservative, normalized)
    np.testing.assert_array_equal(result["stationary_fields"]["conservative"], initial)
    expected = initial if alpha == 0 else np.zeros(lattice.size)
    np.testing.assert_array_equal(result["stationary_fields"]["symmetric normalized"], expected)



def test_gallery_preserves_caller_figure_and_closes_its_own(tmp_path):
    fig, ax = plt.subplots()
    original, = ax.plot([0., 1.], [3., 4.])
    before = plt.get_fignums()
    gallery(str(tmp_path / "gallery"), seed=137)
    assert plt.get_fignums() == before and plt.figure(fig.number) is fig
    np.testing.assert_array_equal(original.get_ydata(), [3., 4.])
