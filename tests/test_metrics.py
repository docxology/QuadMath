import numpy as np

from metrics import shannon_entropy, information_length


def test_shannon_entropy_and_information_length():
    p = np.array([0.25, 0.75])
    H = shannon_entropy(p)
    assert H > 0.0
    grads = np.array([[0.0, 0.0], [1.0, 0.0]])
    L = information_length(grads)
    assert L >= 0.0
import numpy as np

from metrics import shannon_entropy, information_length, fim_eigenspectrum


def test_shannon_entropy_uniform():
    p = np.array([1.0, 1.0, 1.0, 1.0])
    H = shannon_entropy(p)
    assert np.isclose(H, np.log(4.0))


def test_shannon_entropy_unnormalized():
    p = np.array([10.0, 10.0])
    H = shannon_entropy(p)
    assert np.isclose(H, np.log(2.0))


def test_information_length_monotone():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    L = information_length(path)
    assert L >= 0.0


def test_information_length_short_path():
    L = information_length(np.array([[0.0, 0.0]]))
    assert L == 0.0


def test_fim_eigenspectrum():
    F = np.array([[2.0, 0.0], [0.0, 1.0]])
    w, V = fim_eigenspectrum(F)
    assert np.allclose(np.sort(w), np.array([1.0, 2.0]))


def test_fim_eigenspectrum_non_square():
    try:
        fim_eigenspectrum(np.array([[1.0, 2.0, 3.0]]))
        assert False
    except ValueError:
        assert True
