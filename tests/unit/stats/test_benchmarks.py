"""Tests for the timing harness and benchmark scenarios (quadmath/stats/benchmarks.py).

All assertions are structural and timing-free: row counts, name sets,
positive-float invariants, dict round-trips, and table layout.  Wall-clock
timings are measurements and are never asserted numerically.
"""
from __future__ import annotations

import pytest

import quadmath.stats.benchmarks as benchmarks
from quadmath.stats.benchmarks import (
    BENCH_DEFAULTS,
    BenchRow,
    bench_conversions,
    bench_field_fit,
    bench_lattice_search,
    bench_shell_enumeration,
    run_all,
    summary_table,
    time_callable,
)
from quadmath.lattice.omni_numbering import MAX_SHELL


def _assert_positive_row(row: BenchRow) -> None:
    """Assert the numeric invariants every BenchRow must satisfy."""
    assert isinstance(row.name, str) and row.name
    assert row.n > 0 and row.trials > 0
    for attr in ("total_s", "mean_s", "median_s", "p95_s", "ops_per_s"):
        value = getattr(row, attr)
        assert isinstance(value, float) and value > 0.0
    assert row.mean_s == pytest.approx(row.total_s / row.trials)
    assert row.ops_per_s == pytest.approx(row.n * row.trials / row.total_s)
    assert row.p95_s >= row.median_s


# ---------------------------------------------------------------------------
# time_callable
# ---------------------------------------------------------------------------


def test_time_callable_returns_trials_samples_after_warmup():
    calls = []

    def fn() -> int:
        calls.append(1)
        return len(calls)

    samples = time_callable(fn, trials=3, warmup=2)
    assert len(samples) == 3
    assert all(isinstance(s, float) and s > 0.0 for s in samples)
    assert len(calls) == 5  # warmup calls run first, trials after


def test_time_callable_zero_warmup_runs_no_warmup_calls():
    calls = []
    samples = time_callable(lambda: calls.append(1), trials=2, warmup=0)
    assert len(samples) == 2
    assert all(s > 0.0 for s in samples)
    assert len(calls) == 2


@pytest.mark.parametrize("kwargs", [{"trials": 0}, {"trials": -2}, {"warmup": -1}])
def test_time_callable_rejects_invalid_params(kwargs):
    with pytest.raises(ValueError):
        time_callable(lambda: None, **kwargs)


# ---------------------------------------------------------------------------
# BenchRow
# ---------------------------------------------------------------------------


def test_bench_row_as_dict_round_trip():
    row = BenchRow(
        name="op",
        n=10,
        trials=3,
        total_s=1.5,
        mean_s=0.5,
        median_s=0.4,
        p95_s=0.9,
        ops_per_s=20.0,
    )
    assert row.as_dict() == {
        "name": "op",
        "n": 10,
        "trials": 3,
        "total_s": 1.5,
        "mean_s": 0.5,
        "median_s": 0.4,
        "p95_s": 0.9,
        "ops_per_s": 20.0,
    }


# ---------------------------------------------------------------------------
# bench_conversions
# ---------------------------------------------------------------------------


def test_bench_conversions_rows_are_structurally_valid():
    rows = bench_conversions(n=32, trials=3)
    assert [r.name for r in rows] == [
        "quadray.to_xyz",
        "quadray.quadray_from_xyz",
    ]
    for row in rows:
        assert (row.n, row.trials) == (32, 3)
        _assert_positive_row(row)


def test_bench_conversions_rejects_bad_n():
    with pytest.raises(ValueError, match="n must be a positive integer"):
        bench_conversions(n=0, trials=2)


# ---------------------------------------------------------------------------
# bench_shell_enumeration
# ---------------------------------------------------------------------------


def test_bench_shell_enumeration_rows():
    rows = bench_shell_enumeration(k_max=2, trials=3)
    assert [r.name for r in rows] == [
        "omni_numbering.sites_through_shell",
        "omni_numbering.generate_shell",
        "ivm_field.shell_sites",
    ]
    # Shell cardinalities are lattice invariants: shell 2 has 42 sites and
    # the through-shell-2 ball has 1 + 12 + 42 = 55 sites.
    assert rows[0].n == 55
    assert rows[1].n == rows[2].n == 42
    for row in rows:
        assert row.trials == 3
        _assert_positive_row(row)


def test_bench_shell_enumeration_accepts_shell_zero():
    rows = bench_shell_enumeration(k_max=0, trials=2)
    assert all(row.n == 1 for row in rows)


def test_bench_shell_enumeration_rejects_bad_k_max():
    for bad in (-1, MAX_SHELL + 1, True, 1.5):
        with pytest.raises(ValueError):
            bench_shell_enumeration(k_max=bad, trials=2)


# ---------------------------------------------------------------------------
# bench_lattice_search
# ---------------------------------------------------------------------------


def test_bench_lattice_search_rows():
    rows = bench_lattice_search(n_sites=32, queries=8, trials=3)
    assert [r.name for r in rows] == ["lattice_search.nearest"]
    assert (rows[0].n, rows[0].trials) == (8, 3)
    _assert_positive_row(rows[0])


def test_bench_lattice_search_cycles_centers_when_queries_exceed_pool():
    rows = bench_lattice_search(n_sites=4, queries=9, trials=1)
    assert rows[0].n == 9
    _assert_positive_row(rows[0])


def test_bench_lattice_search_rejects_bad_params():
    with pytest.raises(ValueError, match="n_sites and queries must be positive"):
        bench_lattice_search(n_sites=0, queries=8, trials=2)
    with pytest.raises(ValueError, match="n_sites and queries must be positive"):
        bench_lattice_search(n_sites=32, queries=0, trials=2)
    with pytest.raises(ValueError, match="exceeds the"):
        bench_lattice_search(n_sites=10_000, queries=8, trials=2)


# ---------------------------------------------------------------------------
# bench_field_fit
# ---------------------------------------------------------------------------


def test_bench_field_fit_rows():
    rows = bench_field_fit(n_sites=16, trials=2)
    assert [r.name for r in rows] == ["ivm_field.IVMField.learn"]
    assert (rows[0].n, rows[0].trials) == (16, 2)
    _assert_positive_row(rows[0])


def test_bench_field_fit_rejects_bad_n_sites():
    with pytest.raises(ValueError, match="n_sites must be positive"):
        bench_field_fit(n_sites=0, trials=2)
    with pytest.raises(ValueError, match="exceeds the"):
        bench_field_fit(n_sites=10_000, trials=2)


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------


def test_summary_table_contains_header_and_all_names():
    rows = bench_conversions(n=32, trials=2) + bench_lattice_search(
        n_sites=16, queries=4, trials=2
    )
    table = summary_table(rows)
    lines = table.splitlines()
    assert len(lines) == len(rows) + 2  # header + separator + one line per row
    for token in ("name", "n", "trials", "mean_s", "median_s", "p95_s", "ops_per_s"):
        assert token in lines[0]
    for row in rows:
        assert row.name in table


def test_summary_table_empty_rows_renders_header_and_separator():
    lines = summary_table([]).splitlines()
    assert len(lines) == 2
    assert "name" in lines[0]
    assert set(lines[1]) <= {"-", " "}


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def test_run_all_covers_four_benches_with_small_sizes(monkeypatch):
    small = {
        "conversions_n": 32,
        "conversions_trials": 2,
        "shell_k_max": 2,
        "shell_trials": 2,
        "search_n_sites": 16,
        "search_queries": 4,
        "search_trials": 2,
        "field_n_sites": 16,
        "field_trials": 2,
    }
    for key, value in small.items():
        monkeypatch.setitem(benchmarks.BENCH_DEFAULTS, key, value)
    assert set(benchmarks.BENCH_DEFAULTS) == set(BENCH_DEFAULTS)  # untouched keys survive
    rows = run_all()
    assert [r.name for r in rows] == [
        "quadray.to_xyz",
        "quadray.quadray_from_xyz",
        "omni_numbering.sites_through_shell",
        "omni_numbering.generate_shell",
        "ivm_field.shell_sites",
        "lattice_search.nearest",
        "ivm_field.IVMField.learn",
    ]
    assert all(r.trials == 2 for r in rows)
    for row in rows:
        _assert_positive_row(row)


def test_bench_defaults_expose_module_constants():
    assert BENCH_DEFAULTS["conversions_n"] == 200
    assert BENCH_DEFAULTS["shell_k_max"] == 4
    assert BENCH_DEFAULTS["search_n_sites"] == 200
    assert BENCH_DEFAULTS["field_n_sites"] == 64
    assert set(BENCH_DEFAULTS) == {
        "conversions_n",
        "conversions_trials",
        "shell_k_max",
        "shell_trials",
        "search_n_sites",
        "search_queries",
        "search_trials",
        "field_n_sites",
        "field_trials",
    }