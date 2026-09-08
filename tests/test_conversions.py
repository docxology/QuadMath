import numpy as np

from conversions import urner_embedding, quadray_to_xyz
from quadray import Quadray


def test_quadray_to_xyz_known_image():
    M = urner_embedding(scale=1.0)
    q = Quadray(2, 1, 1, 0)
    x, y, z = quadray_to_xyz(q, M)
    # Exact image under the Urner embedding: rows are +/-1, input integral
    assert (x, y, z) == (0.0, 2.0, 2.0)

