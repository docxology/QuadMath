"""Shared helpers for the viz modules: 3D axes scaling and atomic figure output."""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@contextmanager
def figure_scope(*args, **kwargs) -> Iterator[Figure]:
    """Yield a new figure and close it on exit, including when a save fails."""
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


def set_axes_equal(ax: Axes) -> None:
    """Scale a 3D axes box to the data spans so equal lengths match on every axis."""
    spans = [hi - lo for lo, hi in (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())]
    ax.set_box_aspect(spans)  # type: ignore[attr-defined]


def embedding_array(embedding: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a 3x4 embedding as a float array.

    Rows are materialized first, so a one-shot iterable of rows is safe to
    consume more than once downstream.
    """
    return np.asarray([[float(c) for c in row] for row in embedding], dtype=float)


@contextmanager
def atomic_target(path: str) -> Iterator[str]:
    """Yield a temporary sibling of ``path``; move it onto ``path`` only on success.

    The temporary name keeps the extension of ``path`` so writers that infer
    the format from it (matplotlib, numpy, ffmpeg) behave as for the final name.
    """
    directory, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    tmp = os.path.join(directory, f"{stem}.{uuid.uuid4().hex[:12]}.partial{ext}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_figure(fig: Figure, path: str, **kwargs) -> None:
    """Write ``fig`` to ``path`` atomically; keyword arguments go to ``savefig``."""
    with atomic_target(path) as tmp:
        fig.savefig(tmp, **kwargs)


def encoded_angle(quat: Sequence[float]) -> float:
    """Return the rotation magnitude a unit quaternion ``(w, x, y, z)`` encodes.

    The magnitude is ``2*atan2(||(x, y, z)||, w)``; passing it to
    :func:`quadmath.core.quadray.qrotate` with the quaternion satisfies that
    function's encoded-angle validation.
    """
    return 2.0 * float(np.arctan2(np.sqrt(quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2), quat[0]))
