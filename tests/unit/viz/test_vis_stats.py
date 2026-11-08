"""Real statistical geometry, numerical boundaries and figure ownership."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from quadmath.viz.vis_stats import (
    gallery,
    plot_ci_bars,
    plot_ecdf,
    plot_latency_hist,
    plot_scaling_loglog,
)


@pytest.fixture(autouse=True)
def close_owned_figures():
    before = set(plt.get_fignums())
    yield
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_histogram_counts_and_sample_mean():
    ax = plot_latency_hist(np.array([1., 2., 2., 3., 4.]), bins=4)
    np.testing.assert_array_equal([patch.get_height() for patch in ax.patches], [1, 2, 1, 1])
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [2.4, 2.4])


@pytest.mark.parametrize("sample, expected", [
    ([0., 0.], 0.),
    ([1e308, 1.1e308, 1.2e308], 1.1e308),
])
def test_histogram_mean_remains_representable(sample, expected):
    ax = plot_latency_hist(np.array(sample), bins=3)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [expected, expected], rtol=2e-15)


def test_histogram_composes_without_replacing_caller_data():
    fig, ax = plt.subplots()
    original, = ax.plot([0., 1.], [3., 4.])
    before = plt.get_fignums()
    returned = plot_latency_hist(np.array([0.5, 1.5]), ax=ax)
    assert returned is ax and plt.figure(fig.number) is fig
    assert plt.get_fignums() == before
    np.testing.assert_array_equal(original.get_ydata(), [3., 4.])
    np.testing.assert_allclose(ax.lines[-1].get_xdata(), [1., 1.])


@pytest.mark.parametrize("sample", [
    [], [np.nan], [np.inf], [[1., 2.]], [1. + 2.j], np.array(1.), ["not numeric"],
])
def test_invalid_observations_do_not_allocate_a_figure(sample):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_ecdf(np.asarray(sample))
    assert plt.get_fignums() == before


@pytest.mark.parametrize("borrowed", [False, True])
def test_real_drawing_error_closes_only_owned_figures(borrowed):
    fig, caller = plt.subplots()
    original, = caller.plot([0., 1.], [3., 4.])
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_latency_hist(np.array([1., 2.]), ax=caller if borrowed else None, bins=0)
    assert plt.get_fignums() == before and plt.figure(fig.number) is fig
    np.testing.assert_array_equal(original.get_ydata(), [3., 4.])


def test_scaling_predictions_remain_finite_and_ordered_at_extreme_sizes():
    sizes = np.array([1e202, 1e200, 1e201])
    times = np.array([10000., 1., 100.])
    ax = plot_scaling_loglog(sizes, times)
    scatter, fit = ax.lines
    np.testing.assert_array_equal(scatter.get_xdata(), sizes)
    np.testing.assert_array_equal(scatter.get_ydata(), times)
    x = np.asarray(fit.get_xdata())
    assert np.all(np.diff(x) > 0)
    np.testing.assert_allclose(fit.get_ydata(), (x / 1e200) ** 2, rtol=1e-10)


def test_constant_timing_fit_is_drawn_without_invented_variation():
    ax = plot_scaling_loglog(np.array([1., 10., 100.]), np.array([5., 5., 5.]))
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [5., 5., 5.], rtol=1e-12)


@pytest.mark.parametrize("sizes, times", [
    ([1., 2.], [1.]),
    ([0., 1.], [1., 2.]),
    ([1., 2.], [-1., 2.]),
    ([1.], [2.]),
    ([4., 4., 4.], [1., 2., 3.]),
])
def test_unidentified_or_invalid_scaling_data_cannot_leave_a_chart(sizes, times):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_scaling_loglog(np.array(sizes), np.array(times))
    assert plt.get_fignums() == before


@pytest.mark.parametrize("times", [[1e308, 1e308, 1e-308], [1e-308, 1e-308, 1e308]])
def test_unrepresentable_fitted_curve_fails_instead_of_disappearing(times):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_scaling_loglog(np.array([1., 2., 3.]), np.array(times))
    assert plt.get_fignums() == before


def test_interval_endpoints_and_estimates_are_independent_geometry():
    means = np.array([-1., 3., 0.5, 7.])
    lows = np.array([0., 0., 0.25, 5.])
    highs = np.array([1., 1., 0.75, 5.])
    ax = plot_ci_bars(["below", "above", "inside", "zero-width"], means, lows, highs)
    segments = np.asarray(ax.collections[0].get_segments())
    np.testing.assert_array_equal(segments[:, :, 0], np.repeat(np.arange(4)[:, None], 2, axis=1))
    np.testing.assert_array_equal(segments[:, :, 1], np.column_stack((lows, highs)))
    points = np.vstack([np.column_stack(line.get_data()) for line in ax.lines])
    for x, mean in enumerate(means):
        assert np.any(np.all(points == [x, mean], axis=1))


@pytest.mark.parametrize("labels, lows, highs", [
    (["a"], [0., 1.], [2., 3.]),
    (["a", "b"], [0.], [2., 3.]),
    (["a", "b"], [0., 1.], [2.]),
])
def test_misaligned_intervals_do_not_allocate_a_figure(labels, lows, highs):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_ci_bars(labels, np.array([1., 2.]), np.array(lows), np.array(highs))
    assert plt.get_fignums() == before


def test_reversed_interval_bounds_do_not_mutate_caller_axes():
    fig, ax = plt.subplots()
    original, = ax.plot([0., 1.], [3., 4.])
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_ci_bars(["reversed"], np.array([0.]), np.array([2.]), np.array([1.]), ax=ax)
    assert plt.get_fignums() == before and plt.figure(fig.number) is fig
    assert list(ax.lines) == [original]
    np.testing.assert_array_equal(original.get_ydata(), [3., 4.])


def test_ecdf_ties_accumulate_their_exact_empirical_mass():
    sample = np.array([3., 1., 2., 2.])
    ax = plot_ecdf(sample)
    x, y = map(np.asarray, ax.lines[0].get_data())
    assert y[0] == 0. and y[-1] == 1. and np.all(np.diff(y) >= 0)
    for value, fraction in [(1., 0.25), (2., 0.75), (3., 1.)]:
        assert y[x <= value][-1] == fraction
    np.testing.assert_array_equal(sample, [3., 1., 2., 2.])


def test_singleton_ecdf_has_a_visible_zero_to_one_jump():
    ax = plot_ecdf(np.array([5.]))
    vertices = ax.lines[0].get_path().vertices
    np.testing.assert_array_equal(vertices[:, 0], np.full(vertices.shape[0], 5.))
    assert vertices[:, 1].min() == 0. and vertices[:, 1].max() == 1.


def test_gallery_preserves_caller_figure_and_closes_its_own(tmp_path):
    fig, ax = plt.subplots()
    original, = ax.plot([0., 1.], [3., 4.])
    before = plt.get_fignums()
    gallery(str(tmp_path / "gallery"), seed=137)
    assert plt.get_fignums() == before and plt.figure(fig.number) is fig
    np.testing.assert_array_equal(original.get_ydata(), [3., 4.])
