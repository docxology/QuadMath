"""Deterministic frame-based animations over the IVM lattice.

Every renderer here produces a list of :class:`Frame` objects: small 2D
numpy arrays (float in ``[0, 1]`` or ``uint8``) plus a title.  No matplotlib
and no wall-clock or RNG inputs (except the explicit ``seed`` of
:func:`diffusion_frames`), so identical arguments always yield byte-identical
frames, and :func:`frames_to_gif` therefore writes byte-identical GIF files.

Fixed-camera convention (shared by all renderers):
the lattice is embedded in R^3 via :data:`quadmath.core.quadray.DEFAULT_EMBEDDING`
and viewed with an orthographic camera looking down the ``+z`` axis: each
point ``(x, y, z)`` is projected to ``(x, y)`` and mapped onto a square
``GRID_SIZE`` x ``GRID_SIZE`` grid whose horizontal extent is
``[-_HALF_EXTENT, _HALF_EXTENT]`` and whose vertical axis is flipped so that
``+y`` points up (row 0 is the top).  ``_HALF_EXTENT = 8.0`` covers a
radius-3 IVM ball (max embedded norm 6) even while pulsing, with margin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, to_xyz
from quadmath.lattice.ivm_field import IVM_NEIGHBOR_STEPS, ball_sites, quadray_shell_norm

__all__ = [
    "Frame",
    "diffusion_frames",
    "frames_to_gif",
    "lattice_frames",
    "simplex_frames",
]

#: Side length of the fixed square output grid (pixels per frame axis).
GRID_SIZE = 48

#: Half-width of the fixed orthographic camera window in embedded R^3 units.
_HALF_EXTENT = 8.0

#: Numerical tolerance for the unit-norm check on quaternion inputs.
_UNIT_TOL = 1e-6

#: Diffusion update strength per explicit step (alpha < 0.5 keeps it stable).
_DIFFUSION_ALPHA = 0.25

#: Pulse amplitude of :func:`lattice_frames`: radius oscillates in
#: ``[1 - _PULSE_AMP, 1 + _PULSE_AMP]`` relative to the shell radius.
_PULSE_AMP = 0.25


@dataclass(frozen=True)
class Frame:
    """A single animation frame.

    Attributes
    - array: 2D numpy array, either ``uint8`` (raw gray levels) or a float
      dtype with all values in ``[0, 1]`` (brightness).
    - title: Human-readable frame label (e.g. an interpolation parameter).

    Raises
    - ValueError: If ``array`` is not a 2D numpy array, is neither ``uint8``
      nor a float dtype, or is float with values outside ``[0, 1]``.
    """

    array: np.ndarray
    title: str = ""

    def __post_init__(self) -> None:
        arr = self.array
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError("Frame.array must be a 2D numpy array")
        if not (np.issubdtype(arr.dtype, np.floating) or arr.dtype == np.uint8):
            raise ValueError(
                f"Frame.array dtype must be a float dtype or uint8, got {arr.dtype}"
            )
        if np.issubdtype(arr.dtype, np.floating):
            if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
                raise ValueError("float Frame.array values must lie in [0, 1]")


def _as_unit_quat(q: Sequence[float], name: str) -> np.ndarray:
    """Validate and return ``q`` as a float 4-vector of unit norm.

    Parameters
    - q: Sequence of four numbers ``(w, x, y, z)``.
    - name: Label used in error messages.

    Returns
    - np.ndarray: Shape ``(4,)`` float copy of ``q`` (not re-normalized).

    Raises
    - ValueError: If ``q`` does not have exactly four components or its norm
      differs from 1 by more than 1e-6.
    """
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be a 4-vector (w, x, y, z), got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > _UNIT_TOL:
        raise ValueError(f"{name} must be a unit quaternion, got norm {norm!r}")
    return arr


def _slerp(qa: Sequence[float], qb: Sequence[float], t: float) -> np.ndarray:
    """Spherical-linear interpolation between two unit quaternions.

    Standard formula (Shoemaker): with ``d = <qa, qb>`` and, after flipping
    ``qb`` by its sign when ``d < 0`` (shortest arc), the angle
    ``theta = acos(d)``,

        q(t) = sin((1 - t) theta) / sin(theta) * qa
             + sin(t theta)       / sin(theta) * qb

    For nearly-parallel inputs (``d > 1 - 1e-9``) a normalized linear
    interpolation is used instead to avoid division by ``sin(theta) -> 0``.
    The result is normalized, so its norm is 1 up to floating-point error.

    Parameters
    - qa, qb: Unit quaternions as 4-vectors ``(w, x, y, z)``.
    - t: Interpolation parameter in ``[0, 1]``.

    Returns
    - np.ndarray: Shape ``(4,)`` unit quaternion on the shortest arc from
      ``qa`` to ``qb``.  ``t = 0`` returns ``qa`` and ``t = 1`` returns
      ``qb`` exactly; when the inputs are anti-parallel (``d < 0``) the
      returned endpoint is ``-qb``, which represents the same rotation.

    Raises
    - ValueError: If either input is not a unit 4-vector or ``t`` is outside
      ``[0, 1]``.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"t must lie in [0, 1], got {t!r}")
    a = _as_unit_quat(qa, "qa")
    b = _as_unit_quat(qb, "qb")
    d = float(a @ b)
    if d < 0.0:
        b = -b
        d = -d
    if d > 1.0 - 1e-9:
        q = a + t * (b - a)
    else:
        theta = math.acos(min(1.0, d))
        sin_theta = math.sin(theta)
        q = (math.sin((1.0 - t) * theta) / sin_theta) * a + (
            math.sin(t * theta) / sin_theta
        ) * b
    return q / np.linalg.norm(q)


def _rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the 3-vector ``v`` by the unit quaternion ``q``.

    Uses the sandwich product ``q (0, v) conj(q)`` expanded as
    ``v' = v + 2 * cross(qv, cross(qv, v) + w * qv)`` with ``q = (w, qv)``.
    """
    w = float(q[0])
    qv = q[1:]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _project_to_grid(
    xyz: np.ndarray, values: np.ndarray, size: int = GRID_SIZE
) -> np.ndarray:
    """Render points under the fixed orthographic xy camera onto a grid.

    Parameters
    - xyz: Array of shape ``(n, 3)`` embedded points.
    - values: Brightness in ``[0, 1]`` per point.
    - size: Square grid side length.

    Returns
    - np.ndarray: Shape ``(size, size)`` float array in ``[0, 1]``; where
      several points map to the same pixel the maximum brightness wins, and
      points outside the camera window clamp onto the border pixel.
    """
    grid = np.zeros((size, size), dtype=float)
    for (x, y, _), val in zip(xyz, values):
        col = int(np.clip(round((x / _HALF_EXTENT + 1.0) * (size - 1) / 2.0), 0, size - 1))
        row = int(np.clip(round((1.0 - y / _HALF_EXTENT) * (size - 1) / 2.0), 0, size - 1))
        grid[row, col] = max(grid[row, col], val)
    return grid


def _ball_render_values(sites: List[Quadray], shells: int) -> np.ndarray:
    """Brightness per ball site: radial shell, brightest at the center.

    Site with shell norm ``2k`` gets ``0.25 + 0.75 * (1 - k / shells)``, so
    the origin is 1.0 and the outermost shell is 0.25 (never invisible).
    """
    return np.array(
        [0.25 + 0.75 * (1.0 - quadray_shell_norm(q) / (2.0 * shells)) for q in sites],
        dtype=float,
    )


def simplex_frames(q0: Sequence[float], q1: Sequence[float], n: int = 16) -> List[Frame]:
    """Render a quaternion-slerp rotation of the IVM radius-1 ball.

    The two unit quaternions ``q0`` and ``q1`` are interpolated on the
    shortest arc with :func:`_slerp`; at each ``t = i / (n - 1)`` the 13
    lattice sites of :func:`quadmath.lattice.ivm_field.ball_sites` radius 1
    are rotated by ``q(t)``, projected with the fixed orthographic xy camera
    onto a 48x48 grid, and shaded by radial shell (see ``_ball_render_values``
    with ``shells = 1``).  Deterministic: no RNG, no wall clock.

    Parameters
    - q0, q1: Unit quaternions (4-vectors ``(w, x, y, z)``) at ``t = 0`` and
      ``t = 1``.
    - n: Number of frames; frame ``i`` uses ``t = i / (n - 1)``.

    Returns
    - List[Frame]: ``n`` frames of shape ``(48, 48)`` float arrays in
      ``[0, 1]``.

    Raises
    - ValueError: If ``n < 2`` or either quaternion is not a unit 4-vector.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    _as_unit_quat(q0, "q0")
    _as_unit_quat(q1, "q1")
    sites = ball_sites(1)
    embedding = np.array(DEFAULT_EMBEDDING, dtype=float)
    base_xyz = np.array([to_xyz(q, embedding) for q in sites], dtype=float)
    values = _ball_render_values(sites, shells=1)
    frames: List[Frame] = []
    for i in range(n):
        t = i / (n - 1)
        q = _slerp(q0, q1, t)
        rotated = np.array([_rotate_vector(q, p) for p in base_xyz])
        grid = _project_to_grid(rotated, values)
        frames.append(Frame(grid, f"simplex slerp t={t:.3f}"))
    return frames


def lattice_frames(shells: int = 3, n: int = 12) -> List[Frame]:
    """Render a pulsing IVM lattice ball.

    Each frame shows the ball :func:`quadmath.lattice.ivm_field.ball_sites`
    of radius ``shells`` with its embedded coordinates scaled by a
    deterministic pulse ``1 + 0.25 * sin(2 * pi * i / n)`` for frame index
    ``i``, so the ball grows and contracts over one full cycle.  Brightness
    per site is the fixed radial-shell shading of ``_ball_render_values``;
    there is no RNG and no wall clock, so the render is fully deterministic.

    Parameters
    - shells: Ball radius in shell units (must be at least 1).
    - n: Number of frames spanning one pulse period.

    Returns
    - List[Frame]: ``n`` frames of shape ``(48, 48)`` float arrays in ``[0, 1]``.

    Raises
    - ValueError: If ``shells < 1`` or ``n < 2``.
    """
    if shells < 1:
        raise ValueError(f"shells must be at least 1, got {shells}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    sites = ball_sites(shells)
    embedding = np.array(DEFAULT_EMBEDDING, dtype=float)
    base_xyz = np.array([to_xyz(q, embedding) for q in sites], dtype=float)
    values = _ball_render_values(sites, shells=shells)
    frames: List[Frame] = []
    for i in range(n):
        scale = 1.0 + _PULSE_AMP * math.sin(2.0 * math.pi * i / n)
        grid = _project_to_grid(base_xyz * scale, values)
        frames.append(Frame(grid, f"ivm ball pulse scale={scale:.3f}"))
    return frames


def diffusion_frames(n_steps: int = 12, seed: int = 0) -> List[Frame]:
    """Render explicit heat diffusion on the IVM radius-3 ball adjacency.

    The lattice ball :func:`quadmath.lattice.ivm_field.ball_sites` (radius 3,
    147 sites) is turned into a graph using :data:`quadmath.lattice.ivm_field.IVM_NEIGHBOR_STEPS`:
    two sites are adjacent iff their difference normalizes to one of the 12
    IVM neighbor steps.  A ``np.random.default_rng(seed)`` draw picks the
    one-hot heat source site; frame ``i`` shows the field after ``i``
    explicit update steps

        u <- u + alpha * (mean of graph neighbors - u),  alpha = 0.25,

    with values clipped at 0.  Each frame is normalized deterministically by
    dividing by its (always positive) maximum heat, giving float values in
    ``[0, 1]``.  Because each update only spreads heat to graph neighbors,
    the set of heated sites grows monotonically.

    Parameters
    - n_steps: Number of frames rendered (frame 0 is the one-hot source).
    - seed: Seed selecting the heat source site deterministically.

    Returns
    - List[Frame]: ``n_steps`` frames of shape ``(48, 48)`` float arrays in
      ``[0, 1]``.

    Raises
    - ValueError: If ``n_steps < 1``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    sites = ball_sites(3)
    index = {q.normalize().as_tuple(): i for i, q in enumerate(sites)}
    neighbors: List[List[int]] = []
    for q in sites:
        found = set()
        for step in IVM_NEIGHBOR_STEPS:
            j = index.get(q.add(step).normalize().as_tuple())
            if j is not None:
                found.add(j)
        neighbors.append(sorted(found))

    rng = np.random.default_rng(seed)
    source = int(rng.integers(len(sites)))
    heat = np.zeros(len(sites), dtype=float)
    heat[source] = 1.0

    embedding = np.array(DEFAULT_EMBEDDING, dtype=float)
    xyz = np.array([to_xyz(q, embedding) for q in sites], dtype=float)

    frames: List[Frame] = []
    for step_i in range(n_steps):
        if step_i > 0:
            neighbor_mean = np.array(
                [float(np.mean([heat[j] for j in nb])) for nb in neighbors]
            )
            heat = np.clip(heat + _DIFFUSION_ALPHA * (neighbor_mean - heat), 0.0, None)
        grid = _project_to_grid(xyz, heat / float(heat.max()))
        frames.append(Frame(grid, f"diffusion step {step_i}"))
    return frames


def frames_to_gif(
    frames: Sequence[Frame], out_path: str, fps: int = 8, scale: int = 8
) -> str:
    """Assemble frames into an animated GIF at ``out_path``.

    Float frames are quantized to ``uint8`` gray levels by ``round(255 * v)``
    (clipped to ``[0, 255]``); ``uint8`` frames are used as-is.  Each frame
    is upscaled by the integer factor ``scale`` with nearest-neighbor
    resampling, and the sequence is saved with per-frame duration
    ``1000 // fps`` milliseconds and infinite looping.  PIL is imported
    lazily inside this function.  The PIL GIF encoder is deterministic and
    no timestamps are embedded, so identical ``(frames, fps, scale)`` inputs
    produce byte-identical files.

    Parameters
    - frames: Non-empty sequence of :class:`Frame`.
    - out_path: Destination file path (parent directory must exist).
    - fps: Frames per second; duration is ``1000 // fps`` ms.
    - scale: Integer upscale factor applied to every frame.

    Returns
    - str: ``out_path``, unchanged.

    Raises
    - ValueError: If ``frames`` is empty, ``fps <= 0``, or ``scale <= 0``.
    """
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    from PIL import Image  # noqa: WPS433  (lazy import; Pillow is optional at import time)

    nearest = getattr(Image, "Resampling", Image).NEAREST
    duration_ms = 1000 // fps
    images = []
    for frame in frames:
        arr = frame.array
        if arr.dtype == np.uint8:
            data = arr
        else:
            data = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        img = Image.fromarray(data, mode="L")
        if scale != 1:
            img = img.resize((data.shape[1] * scale, data.shape[0] * scale), nearest)
        images.append(img)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return out_path