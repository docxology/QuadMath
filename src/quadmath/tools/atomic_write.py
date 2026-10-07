"""Atomic text writes for generated data under quadmath/output/."""
from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO


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
