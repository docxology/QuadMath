import numpy as np
import pytest

from quadmath.viz.animations import (
    GRID_SIZE,
    Frame,
    _project_to_grid,
    _slerp,
    diffusion_frames,
    frames_to_gif,
    lattice_frames,
    simplex_frames,
)


def _unit_quat(axis_angle_deg: float) -> np.ndarray:
    """Unit quaternion rotating about z by the given angle."""
    half = np.deg2rad(axis_angle_deg) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


# ---------------------------------------------------------------- Frame ----

def test_frame_accepts_float_and_uint8_arrays():
    float_frame = Frame(np.full((4, 4), 0.5), "float")
    assert float_frame.title == "float"
    assert float_frame.array.dtype == np.float64
    uint8_frame = Frame(np.array([[0, 128], [255, 64]], dtype=np.uint8))
    assert uint8_frame.title == ""


def test_frame_rejects_non_2d_array():
    with pytest.raises(ValueError):
        Frame(np.zeros(4))


def test_frame_rejects_non_ndarray():
    with pytest.raises(ValueError):
        Frame([[0.0, 1.0], [1.0, 0.0]])


def test_frame_rejects_bad_dtype():
    with pytest.raises(ValueError):
        Frame(np.array([[1, 2], [3, 4]], dtype=np.int64))


def test_frame_rejects_float_out_of_range():
    with pytest.raises(ValueError):
        Frame(np.array([[1.5]]))
    with pytest.raises(ValueError):
        Frame(np.array([[-0.1]]))


# --------------------------------------------------------------- slerp ----

def test_slerp_endpoints_are_exact():
    qa = np.array([1.0, 0.0, 0.0, 0.0])
    qb = _unit_quat(90.0)
    assert np.allclose(_slerp(qa, qb, 0.0), qa, atol=1e-12)
    assert np.allclose(_slerp(qa, qb, 1.0), qb, atol=1e-12)


def test_slerp_midpoint_is_unit_and_between():
    qa = _unit_quat(0.0)
    qb = _unit_quat(90.0)
    mid = _slerp(qa, qb, 0.5)
    assert abs(float(np.linalg.norm(mid)) - 1.0) < 1e-9
    # Half of a 90-degree rotation: the angle of mid is ~45 degrees.
    angle = 2.0 * np.degrees(np.arctan2(np.linalg.norm(mid[1:]), mid[0]))
    assert abs(angle - 45.0) < 1e-6


def test_slerp_takes_shortest_arc_for_antiparallel_inputs():
    qa = _unit_quat(0.0)
    mid = _slerp(qa, -qa, 0.5)
    # dot(qa, -qa) = -1 flips the target back to qa, so every t is qa.
    assert np.allclose(mid, qa, atol=1e-12)


def test_slerp_near_parallel_uses_normalized_lerp():
    qa = _unit_quat(0.0)
    assert np.allclose(_slerp(qa, qa, 0.5), qa, atol=1e-12)


def test_slerp_rejects_bad_inputs():
    qa = np.array([1.0, 0.0, 0.0, 0.0])
    qb = _unit_quat(30.0)
    with pytest.raises(ValueError):
        _slerp(qa[:3], qb, 0.5)
    with pytest.raises(ValueError):
        _slerp(qa * 2.0, qb, 0.5)
    with pytest.raises(ValueError):
        _slerp(qa, qb, -0.1)
    with pytest.raises(ValueError):
        _slerp(qa, qb, 1.1)


# ------------------------------------------------------ simplex_frames ----

def test_simplex_frames_count_shape_and_range():
    frames = simplex_frames(_unit_quat(0.0), _unit_quat(90.0), n=4)
    assert len(frames) == 4
    for i, frame in enumerate(frames):
        assert frame.array.shape == (GRID_SIZE, GRID_SIZE)
        assert frame.array.dtype.kind == "f"
        assert float(frame.array.min()) >= 0.0
        assert float(frame.array.max()) <= 1.0
        assert frame.title.startswith("simplex")
        assert f"t={i / 3:.3f}" in frame.title


def test_simplex_frames_are_deterministic():
    a = simplex_frames(_unit_quat(0.0), _unit_quat(60.0), n=3)
    b = simplex_frames(_unit_quat(0.0), _unit_quat(60.0), n=3)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.array, fb.array)


def test_simplex_frames_rotates_the_scene():
    # 45-degree rotation about z is not a symmetry of the radius-1 ball.
    frames = simplex_frames(_unit_quat(0.0), _unit_quat(45.0), n=2)
    assert int(np.count_nonzero(frames[0].array)) > 0
    assert not np.array_equal(frames[0].array, frames[1].array)


def test_simplex_frames_rejects_n_below_two():
    with pytest.raises(ValueError):
        simplex_frames(_unit_quat(0.0), _unit_quat(30.0), n=1)


def test_simplex_frames_rejects_non_unit_quaternion():
    with pytest.raises(ValueError):
        simplex_frames(np.array([2.0, 0.0, 0.0, 0.0]), _unit_quat(30.0), n=3)
    with pytest.raises(ValueError):
        simplex_frames(_unit_quat(0.0), _unit_quat(30.0) * 0.5, n=3)


# ------------------------------------------------------ lattice_frames ----

def test_lattice_frames_count_shape_and_pulse():
    frames = lattice_frames(shells=2, n=4)
    assert len(frames) == 4
    for frame in frames:
        assert frame.array.shape == (GRID_SIZE, GRID_SIZE)
        assert float(frame.array.min()) >= 0.0
        assert float(frame.array.max()) <= 1.0
    # Center pixel stays lit throughout; pulse changes the extent.
    center = frames[0].array[GRID_SIZE // 2, GRID_SIZE // 2]
    assert center == pytest.approx(1.0)
    assert not np.array_equal(frames[0].array, frames[1].array)


def test_lattice_frames_are_deterministic():
    a = lattice_frames(shells=2, n=4)
    b = lattice_frames(shells=2, n=4)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.array, fb.array)


def test_lattice_frames_rejects_bad_arguments():
    with pytest.raises(ValueError):
        lattice_frames(shells=0, n=4)
    with pytest.raises(ValueError):
        lattice_frames(shells=2, n=1)


# ---------------------------------------------------- diffusion_frames ----

def test_diffusion_frames_count_shape_and_range():
    frames = diffusion_frames(n_steps=4, seed=0)
    assert len(frames) == 4
    for frame in frames:
        assert frame.array.shape == (GRID_SIZE, GRID_SIZE)
        assert float(frame.array.min()) >= 0.0
        assert float(frame.array.max()) <= 1.0
    # Frame 0 is the one-hot source: exactly one lit pixel.
    assert int(np.count_nonzero(frames[0].array)) == 1


def test_diffusion_frames_spread_monotonically():
    frames = diffusion_frames(n_steps=6, seed=0)
    counts = [int(np.count_nonzero(f.array)) for f in frames]
    assert counts[0] == 1
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]


def test_diffusion_frames_are_deterministic():
    a = diffusion_frames(n_steps=5, seed=3)
    b = diffusion_frames(n_steps=5, seed=3)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.array, fb.array)


def test_diffusion_frames_seed_keeps_one_hot_source():
    for seed in (0, 1, 7):
        frame = diffusion_frames(n_steps=1, seed=seed)[0].array
        assert int(np.count_nonzero(frame)) == 1


def test_diffusion_frames_rejects_bad_arguments():
    with pytest.raises(ValueError):
        diffusion_frames(n_steps=0)


# --------------------------------------------------------- frames_to_gif ----

def test_frames_to_gif_is_byte_identical(tmp_path):
    frames = lattice_frames(shells=2, n=4)
    first = frames_to_gif(frames, str(tmp_path / "a.gif"), fps=8, scale=1)
    second = frames_to_gif(frames, str(tmp_path / "b.gif"), fps=8, scale=1)
    assert first == str(tmp_path / "a.gif")
    with open(first, "rb") as fa:
        with open(second, "rb") as fb:
            assert fa.read() == fb.read()


def test_frames_to_gif_dimensions_and_looping(tmp_path):
    from PIL import Image

    frames = lattice_frames(shells=2, n=3)
    out = frames_to_gif(frames, str(tmp_path / "lat.gif"), fps=8, scale=1)
    with Image.open(out) as img:
        assert img.size == (GRID_SIZE, GRID_SIZE)
        assert getattr(img, "n_frames", 1) == len(frames)
        assert img.info.get("loop") == 0


def test_frames_to_gif_uint8_and_upscale(tmp_path):
    from PIL import Image

    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[3, 3] = 255
    out = frames_to_gif([Frame(arr, "tiny")], str(tmp_path / "u8.gif"), fps=4, scale=8)
    with Image.open(out) as img:
        assert img.size == (64, 64)


def test_frames_to_gif_rejects_bad_arguments(tmp_path):
    frames = lattice_frames(shells=1, n=2)
    target = str(tmp_path / "x.gif")
    with pytest.raises(ValueError):
        frames_to_gif([], target)
    with pytest.raises(ValueError):
        frames_to_gif(frames, target, fps=0)
    with pytest.raises(ValueError):
        frames_to_gif(frames, target, scale=0)


# ------------------------------------------------------ _project_to_grid ----

def test_project_to_grid_clamps_out_of_window_points():
    xyz = np.array([[-100.0, -100.0, 0.0], [0.0, 0.0, 0.0]])
    values = np.array([0.8, 0.4])
    grid = _project_to_grid(xyz, values, size=8)
    assert grid.shape == (8, 8)
    # Out-of-window point clamps to the bottom-left border pixel.
    assert grid[7, 0] == 0.8
    # Origin maps to the center pixel of an even grid (rounded).
    assert int(np.count_nonzero(grid)) >= 1
    assert grid.max() == 0.8