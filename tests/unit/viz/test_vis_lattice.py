"""Tests for the lattice visualization gallery (quadmath/viz/vis_lattice.py).

All assertions are deterministic and pixel-free: artist placement on
caller-provided axes, exact field values at known lattice cells (shell
norms are integers), and byte-identical PNG re-renders for a fixed seed.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import lattice_gallery
import quadmath.paths as paths_module
from quadmath.lattice.ivm_dynamics import DynamicsParams, simulate
from quadmath.lattice.ivm_field import IVMField, quadray_shell_norm
from quadmath.lattice.omni_numbering import generate_shell
from quadmath.core.quadray import Quadray
from quadmath.viz.vis_lattice import (
    DEFAULT_PLANE,
    GALLERY_FILES,
    dynamics_strip,
    field_slice,
    gallery,
    shell_scatter,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure a test created (headless Agg via conftest)."""
    yield
    plt.close("all")


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
    assert ax.get_title() == "IVM shell 2 (42 sites)"
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
    fig = plt.figure()
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


# ---------------------------------------------------------------------------
# gallery: composition, determinism, reproducibility
# ---------------------------------------------------------------------------


def test_gallery_writes_reproducible_figures(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    paths1 = gallery(str(run1), seed=12)
    paths2 = gallery(str(run2), seed=12)
    assert [os.path.basename(p) for p in paths1] == list(GALLERY_FILES)
    for path in paths1:
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
    for first, second in zip(paths1, paths2):
        with open(first, "rb") as fh_a, open(second, "rb") as fh_b:
            assert fh_a.read() == fh_b.read()


def test_gallery_existing_dir_and_seed_sensitivity(tmp_path):
    run_dir = tmp_path / "shared"
    paths_a = gallery(str(run_dir), seed=12)  # creates the directory
    paths_b = gallery(str(run_dir), seed=12)  # reuses the existing directory
    paths_c = gallery(str(tmp_path / "seed13"), seed=13)
    assert [os.path.basename(p) for p in paths_b] == list(GALLERY_FILES)
    # The shell figure uses no randomness: identical across seeds. The field
    # and dynamics figures depend on the seed: different PNG bytes.
    with open(paths_a[0], "rb") as fh_a, open(paths_c[0], "rb") as fh_c:
        assert fh_a.read() == fh_c.read()
    for index in (1, 2):
        with open(paths_a[index], "rb") as fh_a, open(paths_c[index], "rb") as fh_c:
            assert fh_a.read() != fh_c.read()


# ---------------------------------------------------------------------------
# thin script contract (quadmath/scripts/lattice_gallery.py)
# ---------------------------------------------------------------------------


def test_lattice_gallery_script_prints_paths(tmp_path, monkeypatch, capsys):
    figures_dir = tmp_path / "figures"
    monkeypatch.setattr(paths_module, "get_figure_dir", lambda: str(figures_dir))
    lattice_gallery.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert [os.path.basename(line) for line in lines] == list(GALLERY_FILES)
    for line in lines:
        assert os.path.isfile(line)
        assert os.path.getsize(line) > 0
