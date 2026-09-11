"""Timing harness and benchmark scenarios for the QuadMath lattice modules.

Provides a small, deterministic-by-construction timing harness
(:func:`time_callable`) plus four benchmark scenarios over the landed
lattice modules — coordinate conversions, shell enumeration, nearest-site
search, and field fitting — each returning typed :class:`BenchRow` records.
Timing values themselves are wall-clock measurements and necessarily vary
between runs; the workloads (sample construction, seeds, operation
composition) are fully deterministic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from ivm_field import IVMField, quadray_shell_norm, shell_sites
from lattice_search import nearest
from omni_numbering import MAX_SHELL, generate_shell, shell_count, sites_through_shell
from quadray import DEFAULT_EMBEDDING, Quadray, quadray_from_xyz, to_xyz

#: Workload sizes used by :func:`run_all`.  Read at call time so tests can
#: substitute smaller values without re-importing the module.
BENCH_DEFAULTS: Dict[str, int] = {
    "conversions_n": 200,
    "conversions_trials": 5,
    "shell_k_max": 4,
    "shell_trials": 5,
    "search_n_sites": 200,
    "search_queries": 50,
    "search_trials": 5,
    "field_n_sites": 64,
    "field_trials": 3,
}

#: Seed for the conversion benchmark's deterministic integer XYZ samples.
_CONVERSION_SEED: int = 7

#: Seed for the lattice-search benchmark's deterministic query centers.
_SEARCH_SEED: int = 11

#: Seed for the field-fit benchmark's deterministic observation sampling.
_FIELD_SEED: int = 13

#: Lattice ball radius used to host the field-fit observations.
_FIELD_RADIUS: int = 3

#: Query radius and neighbor count for the lattice-search workload.
_SEARCH_RADIUS: float = 2.0
_SEARCH_K: int = 8


@dataclass(frozen=True)
class BenchRow:
    """One timed benchmark result.

    ``total_s`` is the summed wall time over the counted trials;
    ``mean_s``/``median_s``/``p95_s`` are the mean, median, and 95th
    percentile of the per-trial samples; ``ops_per_s = n * trials /
    total_s`` is the throughput of the whole workload across all counted
    trials, where ``n`` is the per-trial workload size (number of samples,
    sites, or queries depending on the scenario).
    """

    name: str
    n: int
    trials: int
    total_s: float
    mean_s: float
    median_s: float
    p95_s: float
    ops_per_s: float

    def as_dict(self) -> dict:
        """Return the row as a plain ``dict`` suitable for serialization."""
        return {
            "name": self.name,
            "n": self.n,
            "trials": self.trials,
            "total_s": self.total_s,
            "mean_s": self.mean_s,
            "median_s": self.median_s,
            "p95_s": self.p95_s,
            "ops_per_s": self.ops_per_s,
        }


def time_callable(fn: Callable[[], object], *, trials: int = 5, warmup: int = 1) -> List[float]:
    """Time ``fn`` with ``time.perf_counter`` and return per-trial wall seconds.

    ``warmup`` calls run first to settle caches and are not counted.

    Parameters
    - fn: Zero-argument callable invoked ``warmup + trials`` times.
    - trials: Number of counted trials (positive integer).
    - warmup: Number of uncounted warmup calls (non-negative integer).

    Returns
    - list[float]: Exactly ``trials`` positive wall-clock seconds.

    Raises
    - ValueError: If ``trials`` is less than 1 or ``warmup`` is negative.
    """
    if trials < 1:
        raise ValueError(f"trials must be a positive integer, got {trials}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def _row(name: str, n: int, samples: Sequence[float]) -> BenchRow:
    """Assemble a :class:`BenchRow` from per-trial wall-second samples."""
    total = float(sum(samples))
    trials = len(samples)
    return BenchRow(
        name=name,
        n=n,
        trials=trials,
        total_s=total,
        mean_s=total / trials,
        median_s=float(np.median(samples)),
        p95_s=float(np.percentile(samples, 95)),
        ops_per_s=n * trials / total,
    )


def bench_conversions(n: int = 200, trials: int = 5) -> List[BenchRow]:
    """Benchmark quadray/XYZ conversions over ``n`` deterministic samples.

    Samples are integer XYZ coordinates in ``[-8, 8]`` drawn from
    ``np.random.default_rng(7)``; the quadray inputs to :func:`to_xyz` are
    the lattice round-trips of those samples via :func:`quadray_from_xyz`.

    Parameters
    - n: Number of deterministic samples per trial (positive integer).
    - trials: Counted timing trials per row.

    Returns
    - list[BenchRow]: One row each for ``quadray.to_xyz`` and
      ``quadray.quadray_from_xyz``, with ``n`` ops per trial.

    Raises
    - ValueError: If ``n`` is less than 1 or ``trials`` is invalid.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    rng = np.random.default_rng(_CONVERSION_SEED)
    coords = rng.integers(-8, 9, size=(n, 3))
    samples_xyz = [(float(x), float(y), float(z)) for x, y, z in coords]
    samples_q = [quadray_from_xyz(x, y, z, DEFAULT_EMBEDDING) for x, y, z in samples_xyz]
    to_xyz_row = _row(
        "quadray.to_xyz",
        n,
        time_callable(lambda: [to_xyz(q, DEFAULT_EMBEDDING) for q in samples_q], trials=trials),
    )
    from_xyz_row = _row(
        "quadray.quadray_from_xyz",
        n,
        time_callable(
            lambda: [quadray_from_xyz(x, y, z, DEFAULT_EMBEDDING) for x, y, z in samples_xyz],
            trials=trials,
        ),
    )
    return [to_xyz_row, from_xyz_row]


def bench_shell_enumeration(k_max: int = 4, trials: int = 5) -> List[BenchRow]:
    """Benchmark shell enumeration through shell ``k_max``.

    Rows cover the omnidirectional close-packing enumeration
    (``omni_numbering.sites_through_shell`` and
    ``omni_numbering.generate_shell``) and the IVM-field shell enumeration
    (``ivm_field.shell_sites``).  ``n`` is the number of sites the row's
    call returns, so ``ops_per_s`` is sites per second.

    Parameters
    - k_max: Deepest shell to enumerate (integer in ``[0, MAX_SHELL]``).
    - trials: Counted timing trials per row.

    Returns
    - list[BenchRow]: One row per enumerated API, sized by its site count.

    Raises
    - ValueError: If ``k_max`` is not an integer in ``[0, MAX_SHELL]``
      or ``trials`` is invalid.
    """
    if isinstance(k_max, bool) or not isinstance(k_max, int):
        raise ValueError(f"k_max must be an integer, got {type(k_max).__name__}")
    if k_max < 0 or k_max > MAX_SHELL:
        raise ValueError(f"k_max must be in [0, MAX_SHELL={MAX_SHELL}], got {k_max}")
    probe = sites_through_shell(k_max)
    rows = [
        _row(
            "omni_numbering.sites_through_shell",
            int(probe.shape[0]),
            time_callable(lambda: sites_through_shell(k_max), trials=trials),
        ),
        _row(
            "omni_numbering.generate_shell",
            int(shell_count(k_max)),
            time_callable(lambda: generate_shell(k_max), trials=trials),
        ),
        _row(
            "ivm_field.shell_sites",
            len(_shell_probe(k_max)),
            time_callable(lambda: _shell_probe(k_max), trials=trials),
        ),
    ]
    return rows


def _shell_probe(k_max: int) -> List["Quadray"]:
    """Return the shell-``k_max`` sites via the IVM-field enumerator."""
    return shell_sites(k_max)


def bench_lattice_search(n_sites: int = 200, queries: int = 50, trials: int = 5) -> List[BenchRow]:
    """Benchmark nearest-site queries through the ``lattice_search`` ball index.

    ``n_sites`` deterministic query centers are sampled (fixed seed) from
    the canonical site enumeration through shell 4; each timed trial runs
    ``queries`` :func:`lattice_search.nearest` calls cycling through those
    centers at a fixed radius and neighbor count, so ``n`` (and hence
    ``ops_per_s``) counts nearest-site queries, not sites.

    Parameters
    - n_sites: Number of deterministic query centers (positive integer,
      at most the sites through shell 4).
    - queries: Number of nearest-site queries per trial (positive integer;
      centers are cycled if ``queries`` exceeds ``n_sites``).
    - trials: Counted timing trials.

    Returns
    - list[BenchRow]: One row for ``lattice_search.nearest``.

    Raises
    - ValueError: If ``n_sites`` or ``queries`` is less than 1, more than
      the available query-center pool, or ``trials`` is invalid.
    """
    if n_sites < 1 or queries < 1:
        raise ValueError(f"n_sites and queries must be positive, got n_sites={n_sites}, queries={queries}")
    pool = sites_through_shell(4)
    if n_sites > pool.shape[0]:
        raise ValueError(
            f"n_sites={n_sites} exceeds the {pool.shape[0]} query centers through shell 4"
        )
    rng = np.random.default_rng(_SEARCH_SEED)
    picked = rng.choice(pool.shape[0], size=n_sites, replace=False)
    centers = pool[picked]
    workload = [(int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in centers]

    def run_queries() -> List[object]:
        results = []
        for i in range(queries):
            center = workload[i % n_sites]
            results.append(nearest(center, _SEARCH_RADIUS, _SEARCH_K))
        return results

    return [
        _row(
            "lattice_search.nearest",
            queries,
            time_callable(run_queries, trials=trials),
        )
    ]


def bench_field_fit(n_sites: int = 64, trials: int = 3) -> List[BenchRow]:
    """Benchmark ``IVMField.learn`` on a synthetic field with a fixed seed.

    The field is built once over the IVM ball of radius
    ``_FIELD_RADIUS``; ``n_sites`` observation sites are sampled with
    ``np.random.default_rng(_FIELD_SEED)`` and carry the deterministic
    shell-norm field.  ``learn`` recomputes ``values`` from scratch, so
    repeated trials re-fit identically.

    Parameters
    - n_sites: Number of observation sites (positive integer, at most the
      ball's site count).
    - trials: Counted timing trials.

    Returns
    - list[BenchRow]: One row for ``ivm_field.IVMField.learn`` with ``n``
      observation sites per trial.

    Raises
    - ValueError: If ``n_sites`` is less than 1 or exceeds the ball size,
      or ``trials`` is invalid.
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be positive, got {n_sites}")
    field = IVMField.lattice_ball(_FIELD_RADIUS)
    if n_sites > len(field.sites):
        raise ValueError(
            f"n_sites={n_sites} exceeds the {len(field.sites)} sites in the "
            f"radius-{_FIELD_RADIUS} ball"
        )
    rng = np.random.default_rng(_FIELD_SEED)
    obs_idx = rng.choice(len(field.sites), size=n_sites, replace=False)
    obs_sites = [field.sites[int(i)] for i in obs_idx]
    obs_values = [float(quadray_shell_norm(q)) for q in obs_sites]

    return [
        _row(
            "ivm_field.IVMField.learn",
            n_sites,
            time_callable(lambda: field.learn(obs_sites, obs_values), trials=trials),
        )
    ]


def summary_table(rows: List[BenchRow]) -> str:
    """Render aligned fixed-width ASCII rows as a table.

    Columns: ``name``, ``n``, ``trials``, ``mean_s``, ``median_s``,
    ``p95_s``, ``ops_per_s``.  The header is followed by a dash separator
    and one line per row; an empty ``rows`` list yields header and
    separator only.

    Parameters
    - rows: Benchmark rows to render.

    Returns
    - str: Newline-joined fixed-width table.
    """
    headers = ("name", "n", "trials", "mean_s", "median_s", "p95_s", "ops_per_s")
    aligns = ("<", ">", ">", ">", ">", ">", ">")
    cells = [
        [
            r.name,
            str(r.n),
            str(r.trials),
            f"{r.mean_s:.6f}",
            f"{r.median_s:.6f}",
            f"{r.p95_s:.6f}",
            f"{r.ops_per_s:.2f}",
        ]
        for r in rows
    ]
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in cells])
        for i in range(len(headers))
    ]

    def line(row: Sequence[str]) -> str:
        parts = []
        for i in range(len(headers)):
            spec = "{0:" + aligns[i] + str(widths[i]) + "}"
            parts.append(spec.format(row[i]))
        return "  ".join(parts)

    lines = [line(headers), "  ".join("-" * w for w in widths)]
    lines.extend(line(row) for row in cells)
    return "\n".join(lines)


def run_all() -> List[BenchRow]:
    """Run every benchmark with the module-level :data:`BENCH_DEFAULTS`.

    Returns
    - list[BenchRow]: The concatenation of the conversion, shell
      enumeration, lattice-search, and field-fit rows, in that order.
    """
    return (
        bench_conversions(
            BENCH_DEFAULTS["conversions_n"],
            BENCH_DEFAULTS["conversions_trials"],
        )
        + bench_shell_enumeration(
            BENCH_DEFAULTS["shell_k_max"],
            BENCH_DEFAULTS["shell_trials"],
        )
        + bench_lattice_search(
            BENCH_DEFAULTS["search_n_sites"],
            BENCH_DEFAULTS["search_queries"],
            BENCH_DEFAULTS["search_trials"],
        )
        + bench_field_fit(
            BENCH_DEFAULTS["field_n_sites"],
            BENCH_DEFAULTS["field_trials"],
        )
    )