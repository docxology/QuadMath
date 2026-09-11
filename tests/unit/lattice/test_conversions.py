"""Tests for the XYZ <-> IVM conversions layer (quadmath/lattice/conversions.py)."""
from fractions import Fraction
from typing import Sequence, cast

import numpy as np
import pytest

from quadmath.lattice.ivm_field import shell_sites
from quadmath.lattice.lattice_search import squared_distance
from quadmath.lattice.omni_numbering import sites_through_shell

from quadmath.lattice.conversions import (
    embedding_basis,
    quadray_roundtrip,
    quadray_to_xyz,
    urner_embedding,
    xyz_to_quadray_canonical,
)
from quadmath.core.quadray import DEFAULT_EMBEDDING, Quadray, to_xyz


def test_quadray_to_xyz_known_image():
    M = urner_embedding(scale=1.0)
    q = Quadray(2, 1, 1, 0)
    x, y, z = quadray_to_xyz(q, M)
    # Exact image under the Urner embedding: rows are +/-1, input integral
    assert (x, y, z) == (0.0, 2.0, 2.0)


def test_quadray_to_xyz_delegates_and_defaults():
    # Delegation: explicit urner_embedding() must equal to_xyz under
    # DEFAULT_EMBEDDING exactly, over the origin and every shell-1 site.
    q0 = Quadray(0, 0, 0, 0)
    for q in [q0] + shell_sites(1):
        assert quadray_to_xyz(q, urner_embedding()) == to_xyz(q, DEFAULT_EMBEDDING)
    # M=None default path: same triple, numerically identical.
    assert quadray_to_xyz(q0) == (0.0, 0.0, 0.0)
    q = shell_sites(1)[0]
    assert quadray_to_xyz(q) == to_xyz(q, DEFAULT_EMBEDDING)


@pytest.mark.parametrize("shape_arg", [(2, 4), (3, 3)])
def test_quadray_to_xyz_rejects_wrong_shape(shape_arg):
    with pytest.raises(ValueError, match="shape"):
        quadray_to_xyz(Quadray(0, 0, 0, 0), np.zeros(shape_arg))


def test_urner_embedding_scale_and_gram():
    base = urner_embedding()
    scaled = urner_embedding(scale=2.0)
    assert np.array_equal(scaled, 2.0 * base)
    assert all(np.isclose(row.sum(), 0.0) for row in base)
    assert np.allclose(base @ base.T, 4.0 * np.eye(3))


def test_roundtrip_all_sites_shells_0_to_4():
    # 1 + 12 + 42 + 92 + 162 = 309 quadrays: exact identity under default M.
    sites = [q for k in range(5) for q in shell_sites(k)]
    assert len(sites) == 1 + 12 + 42 + 92 + 162
    for q in sites:
        assert quadray_roundtrip(q) == q


def test_roundtrip_raises_unnormalized():
    with pytest.raises(AssertionError, match="roundtrip is not the identity"):
        quadray_roundtrip(Quadray(2, 2, 2, 1))


def test_roundtrip_exact_across_scales():
    # pinv(cM).(cM.q) = q - (sum(q)/4)*(1,1,1,1) is scale-independent, so
    # the same-rows roundtrip is exact at any Urner scale.
    for scale in (0.5, 2.0):
        M = urner_embedding(scale)
        for q in shell_sites(1):
            assert quadray_roundtrip(q, M) == q
    M_half = urner_embedding(0.5)
    for q in shell_sites(2):
        assert quadray_roundtrip(q, M_half) == q
    # Void-direction quadray: sum 1, preimage (0.75,-0.25,-0.25,-0.25)
    # rounds to q exactly.
    assert quadray_roundtrip(Quadray(1, 0, 0, 0), M_half) == Quadray(1, 0, 0, 0)


def test_xyz_to_quadray_canonical_recovers_shells_default():
    for k in range(5):
        for q in shell_sites(k):
            assert xyz_to_quadray_canonical(to_xyz(q, DEFAULT_EMBEDDING)) == q


def test_xyz_to_quadray_canonical_half_scale_fcc():
    M_half = urner_embedding(0.5)
    # Explicit half-integer example: the half-scale image is the FCC lattice.
    assert to_xyz(Quadray(1, 0, 0, 0), M_half) == (0.5, 0.5, 0.5)
    q = xyz_to_quadray_canonical((0.5, 0.5, 0.5), M_half)
    assert q == Quadray(1, 0, 0, 0)
    for k in range(5):
        for site in shell_sites(k):
            assert xyz_to_quadray_canonical(to_xyz(site, M_half), M_half) == site


def test_xyz_to_quadray_canonical_fiber_tiebreak():
    # Unnormalized fiber representative (3,2,2,1) shares the image of
    # (2,1,1,0): the min-0 canonical representative must be returned.
    q_unnorm = Quadray(3, 2, 2, 1)
    q_canon = Quadray(2, 1, 1, 0)
    assert to_xyz(q_unnorm, DEFAULT_EMBEDDING) == to_xyz(q_canon, DEFAULT_EMBEDDING)
    assert xyz_to_quadray_canonical(to_xyz(q_unnorm, DEFAULT_EMBEDDING)) == q_canon


def test_xyz_to_quadray_canonical_exact_input_types():
    # Fraction vector and np.int64 array both hit the exact-converter branches.
    xyz_frac = (Fraction(1), Fraction(1), Fraction(1))
    assert xyz_to_quadray_canonical(xyz_frac) == Quadray(1, 0, 0, 0)
    xyz_int64 = np.array([1, 1, 1], dtype=np.int64)
    assert xyz_to_quadray_canonical(xyz_int64) == Quadray(1, 0, 0, 0)
    # np.float64 is a float subclass: accepted, exact binary rational.
    assert xyz_to_quadray_canonical(np.array([1.0, 1.0, 1.0], dtype=np.float64)) == Quadray(1, 0, 0, 0)


def test_xyz_to_quadray_canonical_rejects_non_image():
    # (0.5, 0.5, 0.5) is off the scale-1 lattice: integral preimage fails.
    with pytest.raises(ValueError, match="not in the embedding image"):
        xyz_to_quadray_canonical((0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="not in the embedding image"):
        xyz_to_quadray_canonical((0.1, 0.0, 0.0))


def test_xyz_to_quadray_canonical_rejects_bad_length():
    with pytest.raises(ValueError, match="length-3 sequence"):
        xyz_to_quadray_canonical(cast(Sequence, 3.0))  # not a sequence at all
    with pytest.raises(ValueError, match="length 3"):
        xyz_to_quadray_canonical((1.0, 2.0))
    with pytest.raises(ValueError, match="length 3"):
        xyz_to_quadray_canonical((1.0, 2.0, 3.0, 4.0))


def test_xyz_to_quadray_canonical_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        xyz_to_quadray_canonical((0.0, 0.0, 0.0), np.zeros((3, 3)))


def test_xyz_to_quadray_canonical_rejects_nonzero_row_sums():
    bad = [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]]  # row sums 1, 2, 2
    with pytest.raises(ValueError, match="does not sum to exactly 0"):
        xyz_to_quadray_canonical((1.0, 1.0, 1.0), bad)


def test_xyz_to_quadray_canonical_rejects_rank_deficient():
    # All row sums are 0 but rank < 3 (rows 0,1 are parallel, row 2 is zero).
    bad = [[1, -1, -1, 1], [2, -2, -2, 2], [0, 0, 0, 0]]
    with pytest.raises(ValueError, match="rank < 3"):
        xyz_to_quadray_canonical((0.0, 0.0, 0.0), bad)


def test_xyz_to_quadray_canonical_rejects_bad_entry_types():
    with pytest.raises(TypeError, match="must be int, float, Fraction"):
        xyz_to_quadray_canonical(("a", 0.0, 0.0))
    with pytest.raises(TypeError, match="must be int, float, Fraction"):
        xyz_to_quadray_canonical((None, 0.0, 0.0))
    with pytest.raises(TypeError, match="must be int, float, Fraction"):
        xyz_to_quadray_canonical((1.0, 1.0, 1.0), [[1, "x", 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])


def test_embedding_basis_columns_and_gram():
    B = embedding_basis()
    assert B.shape == (4, 3)
    G = B @ B.T
    assert np.array_equal(G, 4.0 * np.eye(4) - np.ones((4, 4)))
    assert np.array_equal(G, urner_embedding(1.0).T @ urner_embedding(1.0))
    B_half = embedding_basis(urner_embedding(0.5))
    assert np.array_equal(B_half @ B_half.T, 0.25 * (4.0 * np.eye(4) - np.ones((4, 4))))


def test_embedding_basis_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        embedding_basis(np.zeros((3, 3)))


def test_lattice_distance_identity_via_conversions():
    # lattice_search.squared_distance is d^2 = 4*sum(delta^2) - (sum delta)^2
    # over integer quadray deltas; this must equal the embedded float
    # pipeline |conversions.xyz difference|^2 under DEFAULT_EMBEDDING
    # (exact for these small magnitudes).  Exactness of the identity is what
    # embedding_basis's Gram matrix G = 4*I4 - J4 encodes.
    origin = Quadray(0, 0, 0, 0)
    centers = [origin] + shell_sites(1)
    targets = sites_through_shell(2)
    assert targets.shape[0] == 55
    for center in centers:
        d2_lat = squared_distance(np.array(center.as_tuple()), targets)
        p = np.array(to_xyz(center, DEFAULT_EMBEDDING))
        d2_float = np.sum(
            (np.array([to_xyz(Quadray(*row), DEFAULT_EMBEDDING) for row in targets]) - p) ** 2,
            axis=1,
        )
        deltas = targets - np.array(center.as_tuple())
        d2_closed = 4.0 * np.sum(deltas * deltas, axis=1) - np.sum(deltas, axis=1) ** 2
        assert np.array_equal(d2_lat, d2_closed)
        assert np.allclose(d2_lat, d2_float, rtol=0, atol=1e-12)
