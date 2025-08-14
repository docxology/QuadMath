import numpy as np

from information import fisher_information_matrix, natural_gradient_step, free_energy


def test_fisher_information_matrix_shapes():
    grads = np.array([[1.0, 0.0], [0.0, 2.0]])
    F = fisher_information_matrix(grads)
    assert F.shape == (2, 2)


def test_fisher_information_matrix_bad_ndim():
    try:
        fisher_information_matrix(np.array([1.0, 2.0]))
        assert False
    except ValueError:
        assert True


def test_natural_gradient_step_runs():
    g = np.array([1.0, -2.0])
    F = np.eye(2)
    step = natural_gradient_step(g, F, step_size=0.1, ridge=0.0)
    assert step.shape == (2,)


def test_free_energy_basic():
    logp = np.log(np.array([0.6, 0.4]))
    q = np.array([0.5, 0.5])
    p = np.array([0.5, 0.5])
    F = free_energy(logp, q, p)
    assert np.isfinite(F)


def test_free_energy_shape_mismatch():
    try:
        free_energy(np.array([0.0, 0.0]), np.array([1.0]), np.array([1.0]))
        assert False
    except ValueError:
        assert True


def test_natural_gradient_shape_mismatch():
    g = np.array([1.0, -2.0, 3.0])
    F = np.eye(2)
    try:
        natural_gradient_step(g, F)
        assert False
    except ValueError:
        assert True
