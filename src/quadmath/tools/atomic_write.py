"""Atomic text writes for generated data under quadmath/output/."""
from __future__ import annotations

import contextlib
import os
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

import numpy as np

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


@contextmanager
def atomic_open(path: str | os.PathLike[str], *, newline: str | None = None) -> Iterator[IO[str]]:
    """Yield a text handle whose contents replace ``path`` only if the block succeeds.

    The temp file sits beside the target so ``os.replace`` stays on one
    filesystem. On any exception the temp file is removed and the previous
    target, if any, is left untouched.
    """
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    temp = os.path.join(directory, f".{os.path.basename(target)}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    handle = open(temp, "x", encoding="utf-8", newline=newline)
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str, *, newline: str | None = None) -> None:
    """Write ``text`` to ``path`` atomically (see :func:`atomic_open`)."""
    with atomic_open(path, newline=newline) as handle:
        handle.write(text)


def atomic_savez(path: str | os.PathLike[str], /, **arrays: Any) -> None:
    """Write ``arrays`` as an ``.npz`` archive at ``path`` atomically and byte-reproducibly.

    ``np.savez`` stamps each entry with the wall clock; here every entry carries
    a fixed ZIP timestamp so identical arrays always produce identical bytes.
    Object arrays are refused (``allow_pickle=False``).
    """
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    temp = os.path.join(directory, f".{os.path.basename(target)}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "xb") as handle:
            with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for key, value in arrays.items():
                    info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_EPOCH)
                    with archive.open(info, mode="w", force_zip64=True) as entry:
                        np.lib.format.write_array(entry, np.asanyarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise
