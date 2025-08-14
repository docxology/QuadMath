import numpy as np

from metrics import fim_eigenspectrum


def test_fim_eigenspectrum_ordering():
    F = np.array([[2.0, 0.0], [0.0, 1.0]])
    w, V = fim_eigenspectrum(F)
    assert np.allclose(w, np.array([2.0, 1.0]))
    assert V.shape == (2, 2)

