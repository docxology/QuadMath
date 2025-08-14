import numpy as np

from conversions import urner_embedding, quadray_to_xyz
from quadray import Quadray


def test_quadray_to_xyz_runs():
    M = urner_embedding(scale=1.0)
    q = Quadray(2, 1, 1, 0)
    x, y, z = quadray_to_xyz(q, M)
    assert isinstance(x, float) and isinstance(y, float) and isinstance(z, float)

