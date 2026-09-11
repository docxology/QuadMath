"""Executable transcription of SPEC.md — the QuadMath lattice-core specification.

Every normative claim S1..S30 in the repo-root ``SPEC.md`` is exercised by exactly
one named test below; the set of test function names equals the set of names in
SPEC.md's claim index (§7.5) in both directions.  All examples are deterministic
real numerical values (no randomness, no mocks); results that are exact in the
code (``fractions.Fraction``, integer identities) are asserted exactly.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Optional, Sequence, Tuple, cast

import numpy as np
import pytest

from quadmath.lattice.conversions import (
    embedding_basis,
    quadray_roundtrip,
    quadray_to_xyz,
    urner_embedding,
    xyz_to_quadray_canonical,
)
from quadmath.lattice.ivm_field import (
    IVM_NEIGHBOR_STEPS,
    ball_sites,
    is_ivm_site,
    quadray_shell_norm,
    shell_cardinalities,
    shell_sites,
)
from quadmath.lattice.lattice_search import nearest, squared_distance, within_radius
from quadmath.lattice.omni_numbering import (
    MAX_SHELL,
    NEIGHBOR_MOVES,
    cumulative_count,
    generate_shell,
    shell_count,
    site_at_index,
    site_index,
    sites_through_shell,
)
from quadmath.core.quadray import (
    DEFAULT_EMBEDDING,
    Quadray,
    ace_tetravolume_5x5,
    angle,
    distance,
    dot,
    integer_tetra_volume,
    magnitude,
    quadray_from_xyz,
    to_xyz,
)


def _brute_force_nearest(
    xyz: Tuple[float, float, float], box: int = 2
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Exhaustive XYZ-nearest normalized quadray class over ``[-box, box]^4``.

    Returns ``(d2, (a, b, c, d))`` minimizing ``(d2, tuple)`` lexicographically —
    the deterministic tie-break used throughout SPEC.md's worked examples.
    """
    best: Optional[Tuple[float, Tuple[int, int, int, int]]] = None
    for a in range(-box, box + 1):
        for b in range(-box, box + 1):
            for c in range(-box, box + 1):
                for d in range(-box, box + 1):
                    raw = np.array([a, b, c, d])
                    centered = raw - raw.min()
                    canonical = (int(centered[0]), int(centered[1]), int(centered[2]), int(centered[3]))
                    image = to_xyz(Quadray(*canonical), DEFAULT_EMBEDDING)
                    d2 = sum((image[i] - xyz[i]) ** 2 for i in range(3))
                    key = (d2, canonical)
                    if best is None or key < best:
                        best = key
    assert best is not None
    return best


# --- §2: Quadray coordinates, projective classes, and tetra-volume ----------


def test_normalize_selects_unique_canonical_representative():
    """S1: normalize subtracts (k,k,k,k); unique min-0 canonical representative."""
    assert Quadray(3, 2, 2, 1).normalize() == Quadray(2, 1, 1, 0)
    assert Quadray(1, 0, 0, -1).normalize() == Quadray(2, 1, 1, 0)
    q = Quadray(2, 1, 1, 0)
    assert q.normalize().normalize() == q.normalize()
    for t in (-2, -1, 1, 2):
        shifted = Quadray(q.a + t, q.b + t, q.c + t, q.d + t)
        assert shifted.normalize() == q
        assert to_xyz(shifted, DEFAULT_EMBEDDING) == to_xyz(q, DEFAULT_EMBEDDING)


def test_forward_map_matrix_product_values():
    """S2: to_xyz evaluates the matrix product M q with the pinned values."""
    q = Quadray(2, 1, 1, 0)
    assert to_xyz(q, DEFAULT_EMBEDDING) == (0.0, 2.0, 2.0)
    assert to_xyz(Quadray(1, 1, 1, 0), DEFAULT_EMBEDDING) == (-1.0, 1.0, 1.0)
    m = np.asarray(DEFAULT_EMBEDDING)
    product = m @ np.asarray(q.as_tuple(), dtype=np.float64)
    assert np.array_equal(np.asarray(to_xyz(q, DEFAULT_EMBEDDING)), product)
    images = sorted(to_xyz(site, DEFAULT_EMBEDDING) for site in shell_sites(1))
    assert len(set(images)) == 12
    assert all(sorted(abs(v) for v in image) == [0.0, 2.0, 2.0] for image in images)


def test_forward_map_constant_on_projective_fiber():
    """S3: to_xyz(q + t*(1,1,1,1)) == to_xyz(q) for representative t values."""
    for base in (Quadray(2, 1, 1, 0), Quadray(1, 1, 0, 0)):
        image = to_xyz(base, DEFAULT_EMBEDDING)
        for t in range(-3, 4):
            shifted = Quadray(base.a + t, base.b + t, base.c + t, base.d + t)
            assert to_xyz(shifted, DEFAULT_EMBEDDING) == image


def test_tetra_volume_unit_primitive_and_ace_agreement():
    """S4: |det|/4 exact; unit tetra 1, primitive 1/4; the two functions agree."""
    unit = (Quadray(0, 0, 0, 0), Quadray(2, 1, 1, 0), Quadray(1, 2, 1, 0), Quadray(1, 1, 2, 0))
    primitive = (Quadray(0, 0, 0, 0), Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0))
    general = (Quadray(0, 0, 0, 0), Quadray(4, 2, 2, 0), Quadray(2, 4, 2, 0), Quadray(2, 2, 4, 0))
    assert integer_tetra_volume(*unit) == Fraction(1) == ace_tetravolume_5x5(*unit)
    assert integer_tetra_volume(*primitive) == Fraction(1, 4) == ace_tetravolume_5x5(*primitive)
    assert integer_tetra_volume(*general) == ace_tetravolume_5x5(*general) == Fraction(8)
    assert isinstance(integer_tetra_volume(*general), Fraction)


def test_embedded_geometry_exact_values():
    """S5: magnitude/dot/distance/angle pinned under DEFAULT_EMBEDDING."""
    q1, q2 = Quadray(2, 1, 1, 0), Quadray(1, 2, 1, 0)
    origin = Quadray(0, 0, 0, 0)
    assert magnitude(q1, DEFAULT_EMBEDDING) == pytest.approx(math.sqrt(8.0), abs=1e-12)
    assert distance(origin, q1, DEFAULT_EMBEDDING) == pytest.approx(math.sqrt(8.0), abs=1e-12)
    assert dot(q1, q2, DEFAULT_EMBEDDING) == 4.0
    assert angle(q2, origin, q1, DEFAULT_EMBEDDING) == pytest.approx(math.pi / 3.0, abs=1e-12)


# --- §3: The embedding and its projective fiber ------------------------------


def test_urner_embedding_matrix_rows_and_row_sums():
    """S6: the pinned Urner matrix; zero row sums; DEFAULT_EMBEDDING agreement."""
    expected = np.array(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    assert np.array_equal(urner_embedding(1.0), expected)
    assert np.array_equal(urner_embedding(0.5), 0.5 * expected)
    assert np.array_equal(np.asarray(DEFAULT_EMBEDDING), expected)
    assert all(float(urner_embedding(1.0)[j].sum()) == 0.0 for j in range(3))


def test_quadray_to_xyz_delegates_and_shape_validates():
    """S7: delegation to quadray.to_xyz; explicit M must have shape (3, 4)."""
    for site in [Quadray(0, 0, 0, 0)] + shell_sites(1):
        assert quadray_to_xyz(site) == to_xyz(site, DEFAULT_EMBEDDING)
    with pytest.raises(ValueError, match="shape"):
        quadray_to_xyz(Quadray(0, 0, 0, 0), np.zeros((2, 4)))


# --- §4: Gram identities, exact distances, and the IVM lattice ---------------


def test_row_gram_identity_scales_quadratically():
    """S8: M M^T == 4 c^2 I_3 at Urner scales 1 and 1/2."""
    for c in (1.0, 0.5):
        m = urner_embedding(c)
        assert np.array_equal(m @ m.T, 4.0 * c * c * np.eye(3))


def test_embedding_basis_column_gram_identity():
    """S9: B B^T == 4 I_4 - J_4 at scale 1; scales by c^2; B rows are columns."""
    b = embedding_basis()
    assert b.shape == (4, 3)
    assert np.array_equal(b, urner_embedding(1.0).T)
    gram = b @ b.T
    assert np.array_equal(gram, 4.0 * np.eye(4) - np.ones((4, 4)))
    assert np.array_equal(np.diag(gram), np.full(4, 3.0))
    assert all(gram[i][j] == -1.0 for i in range(4) for j in range(4) if i != j)
    b_half = embedding_basis(urner_embedding(0.5))
    assert np.array_equal(b_half @ b_half.T, 0.25 * (4.0 * np.eye(4) - np.ones((4, 4))))


def test_distance_identity_matches_float_pipeline():
    """S10: d^2 = 4*sum(delta^2) - (sum delta)^2; exact; matches float pipeline."""
    origin = np.zeros((1, 4))
    assert squared_distance(np.array([2.0, 1.0, 1.0, 0.0]), origin)[0] == 8.0
    assert squared_distance(np.array([4.0, 2.0, 2.0, 0.0]), origin)[0] == 32.0
    assert distance(
        Quadray(0, 0, 0, 0), Quadray(2, 1, 1, 0), DEFAULT_EMBEDDING
    ) == pytest.approx(math.sqrt(8.0), abs=1e-12)
    centers = [Quadray(0, 0, 0, 0)] + shell_sites(1)
    targets = sites_through_shell(2)
    assert targets.shape[0] == 55
    for center in centers:
        center_arr = np.asarray(center.as_tuple(), dtype=np.float64)
        d2_lat = squared_distance(center_arr, targets)
        base = np.asarray(to_xyz(center, DEFAULT_EMBEDDING))
        embedded = np.asarray(
            [to_xyz(Quadray(*row), DEFAULT_EMBEDDING) for row in targets]
        )
        d2_float = np.sum((embedded - base) ** 2, axis=1)
        deltas = targets - center_arr
        d2_closed = 4.0 * np.sum(deltas * deltas, axis=1) - np.sum(deltas, axis=1) ** 2
        assert np.array_equal(d2_lat, d2_closed)
        assert np.allclose(d2_lat, d2_float, rtol=0.0, atol=1e-12)


def test_shell_truncation_bound_8g():
    """S11: d^2(0, s) >= 8g on shell g; minima 8, 16, 40, 64 (tight at g = 1, 2)."""
    sites = sites_through_shell(4)
    expected_minima = {1: 8.0, 2: 16.0, 3: 40.0, 4: 64.0}
    for g in range(1, 5):
        shell = sites[cumulative_count(g - 1) : cumulative_count(g)]
        assert shell.shape[0] == [12, 42, 92, 162][g - 1]
        d2 = squared_distance(np.zeros(4), shell)
        assert d2.min() >= 8.0 * g
        assert d2.min() == expected_minima[g]


def test_ivm_membership_residue_classes():
    """S12: site iff normalized sum ≡ 0 (mod 4); class property; voids rejected."""
    sites = ball_sites(4)
    assert len(sites) == 309
    assert all(is_ivm_site(q) for q in sites)
    assert is_ivm_site(Quadray(3, 2, 2, 1)) == is_ivm_site(Quadray(2, 1, 1, 0)) is True
    assert not is_ivm_site(Quadray(1, 0, 0, 0))
    assert not is_ivm_site(Quadray(1, 1, 0, 0))
    assert not is_ivm_site(Quadray(1, 1, 1, 0))
    assert not is_ivm_site(Quadray(2, 2, 2, 1))


def test_shell_norm_values_and_void_rejection():
    """S13: N(q) = sum|q_i - s/4| = 2k on sites; class property; voids raise."""
    assert quadray_shell_norm(Quadray(0, 0, 0, 0)) == 0
    assert quadray_shell_norm(Quadray(2, 1, 1, 0)) == 2
    assert quadray_shell_norm(Quadray(3, 2, 2, 1)) == 2
    assert quadray_shell_norm(Quadray(0, 0, 1, 3)) == 4
    assert quadray_shell_norm(Quadray(4, 2, 2, 0)) == 4
    for void in (Quadray(1, 0, 0, 0), Quadray(1, 1, 0, 0), Quadray(1, 1, 1, 0)):
        with pytest.raises(ValueError, match="not an IVM lattice site"):
            quadray_shell_norm(void)


def test_shell_enumeration_cuboctahedral_cardinalities():
    """S14: lexicographic shell enumeration; cardinalities 10k^2+2; ball union."""
    assert shell_cardinalities(4) == [1, 12, 42, 92, 162]
    for k in range(5):
        sites_k = shell_sites(k)
        assert len(sites_k) == (1 if k == 0 else 10 * k * k + 2)
        assert sites_k == sorted(sites_k, key=lambda q: q.as_tuple())
        for q in sites_k:
            assert min(q.as_tuple()) == 0
            assert sum(q.as_tuple()) % 4 == 0
            assert quadray_shell_norm(q) == 2 * k
    assert [q.as_tuple() for q in shell_sites(2)[:3]] == [
        (0, 0, 1, 3),
        (0, 0, 2, 2),
        (0, 0, 3, 1),
    ]
    ball = ball_sites(4)
    assert len(ball) == 309
    assert ball == [q for k in range(5) for q in shell_sites(k)]


def test_twelve_neighbor_moves_cuboctahedron():
    """S15: 12 normalized permutations of (2,1,1,0) = shell 1; d^2 = 8 each."""
    expected = tuple(sorted(set(itertools.permutations((2, 1, 1, 0)))))
    assert tuple(q.as_tuple() for q in IVM_NEIGHBOR_STEPS) == expected
    assert tuple(q.as_tuple() for q in shell_sites(1)) == expected
    assert NEIGHBOR_MOVES.shape == (12, 4)
    assert NEIGHBOR_MOVES.dtype == np.int64
    assert all(tuple(row) in expected for row in NEIGHBOR_MOVES.tolist())
    origin = np.zeros((1, 4))
    for q in IVM_NEIGHBOR_STEPS:
        image = to_xyz(q, DEFAULT_EMBEDDING)
        assert sorted(abs(v) for v in image) == [0.0, 2.0, 2.0]
        row = np.asarray(q.as_tuple(), dtype=np.float64)
        assert squared_distance(row, origin)[0] == 8.0


def test_shell_count_and_cumulative_closed_form():
    """S16: shell_count = 10k^2+2; cumulative closed form matches through 6."""
    partial = 0
    for k in range(7):
        expected = 1 if k == 0 else 10 * k * k + 2
        assert shell_count(k) == expected
        partial += expected
        closed = 1 + 2 * k + 10 * k * (k + 1) * (2 * k + 1) // 6
        assert cumulative_count(k) == closed == partial
    assert [cumulative_count(k) for k in range(7)] == [1, 13, 55, 147, 309, 561, 923]
    for k in range(5):
        assert generate_shell(k).shape[0] == shell_count(k)


def test_omni_numbering_bidirectional_bijection():
    """S17: shell-major lexicographic order; site_index/site_at_index inverses."""
    assert MAX_SHELL == 32
    sites = sites_through_shell(3)
    assert sites.shape[0] == cumulative_count(3) == 147
    assert tuple(int(v) for v in sites[0]) == (0, 0, 0, 0)
    assert tuple(int(v) for v in sites[1]) == (0, 1, 1, 2)
    for index in range(sites.shape[0]):
        q = site_at_index(index, 3)
        assert site_index(q, 3) == index
    assert site_index(Quadray(1, 0, 0, 0), 3) == -1
    assert site_index((5, 5, 5, 5), 3) == 0
    for k in range(1, 4):
        omni_rows = [tuple(int(v) for v in row) for row in generate_shell(k)]
        assert omni_rows == [q.as_tuple() for q in shell_sites(k)]


def test_nearest_within_radius_exact_and_deterministic():
    """S18: exact d^2 results consistent with brute force; deterministic order."""
    center = np.array([0.5, 1.0, -0.5, 2.0])
    sites, d2 = within_radius(center, 3.0)
    assert np.all(d2 <= 9.0)
    assert np.all(np.diff(d2) >= 0)
    for value in np.unique(d2):
        tie = [tuple(int(v) for v in row) for row in sites[d2 == value].tolist()]
        assert tie == sorted(tie)
    ns, nd = nearest(center, 3.0, 5)
    assert ns.shape[0] == 5
    assert np.all(np.diff(nd) >= 0)
    assert np.array_equal(ns, sites[:5])
    assert np.array_equal(nd, d2[:5])
    o_sites, o_d2 = within_radius(Quadray(0, 0, 0, 0), 2.0 * math.sqrt(2.0))
    assert o_sites.shape[0] == 13
    assert o_d2[0] == 0.0
    assert np.all(o_d2[1:] == 8.0)
    assert [tuple(int(v) for v in row) for row in o_sites[1:]] == [
        q.as_tuple() for q in IVM_NEIGHBOR_STEPS
    ]


def test_search_fails_closed_beyond_max_shell():
    """S19: queries needing shells beyond MAX_SHELL raise ValueError."""
    with pytest.raises(ValueError, match="MAX_SHELL"):
        nearest(Quadray(0, 0, 0, 0), 20.0, 1)
    with pytest.raises(ValueError, match="MAX_SHELL"):
        within_radius(Quadray(0, 0, 0, 0), 20.0)


# --- §5: Exact canonical inversion -------------------------------------------


def test_canonical_inversion_exact_recovery_shells_0_to_4():
    """S20: exact recovery of all 309 canonical sites; also at scale 1/2."""
    sites = [q for k in range(5) for q in shell_sites(k)]
    assert len(sites) == 309
    for q in sites:
        assert xyz_to_quadray_canonical(to_xyz(q, DEFAULT_EMBEDDING)) == q
    m_half = urner_embedding(0.5)
    for q in sites:
        assert xyz_to_quadray_canonical(to_xyz(q, m_half), m_half) == q


def test_canonical_inversion_exact_entry_types():
    """S21: exact entry types accepted; image of (1,1,1,0) is (-1, 1, 1)."""
    target = Quadray(1, 1, 1, 0)
    assert xyz_to_quadray_canonical((-1, 1, 1)) == target
    assert xyz_to_quadray_canonical((Fraction(-1), Fraction(1), Fraction(1))) == target
    assert xyz_to_quadray_canonical((np.int64(-1), np.int64(1), np.int64(1))) == target
    assert xyz_to_quadray_canonical(np.array([-1.0, 1.0, 1.0], dtype=np.float64)) == target


def test_canonical_inversion_fiber_tiebreak_min_zero():
    """S22: fiber {y + t*1}; unique min-0 representative returned."""
    q_canon = Quadray(2, 1, 1, 0)
    q_unnorm = Quadray(3, 2, 2, 1)
    assert to_xyz(q_unnorm, DEFAULT_EMBEDDING) == to_xyz(q_canon, DEFAULT_EMBEDDING)
    assert to_xyz(q_canon, DEFAULT_EMBEDDING) == (0.0, 2.0, 2.0)
    assert xyz_to_quadray_canonical(to_xyz(q_unnorm, DEFAULT_EMBEDDING)) == q_canon


def test_canonical_inversion_rank_lemma():
    """S23: det[C0 C1 C2] = 4 c^3; zero-row-sum rank deficiency raises."""
    a = ((1, -1, -1), (1, 1, -1), (1, -1, 1))
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    assert det == 4
    for c in (0.5, 2.0):
        assert np.linalg.det(urner_embedding(c)[:, :3]) == pytest.approx(4.0 * c**3, abs=1e-12)
    rank_deficient = [[1, -1, -1, 1], [2, -2, -2, 2], [0, 0, 0, 0]]
    with pytest.raises(ValueError, match="rank < 3"):
        xyz_to_quadray_canonical((0.0, 0.0, 0.0), rank_deficient)


def test_canonical_inversion_rejects_non_image():
    """S24: off-lattice points raise ValueError; never snapped."""
    for xyz in ((0.5, 0.5, 0.5), (0.1, 0.0, 0.0)):
        with pytest.raises(ValueError, match="not in the embedding image"):
            xyz_to_quadray_canonical(xyz)


def test_canonical_inversion_fail_closed_taxonomy():
    """S25: ValueError for structural mismatches; TypeError for foreign entries."""
    with pytest.raises(ValueError, match="length-3 sequence"):
        xyz_to_quadray_canonical(cast(Sequence, None))
    with pytest.raises(ValueError, match="length 3, got 2"):
        xyz_to_quadray_canonical((1.0, 2.0))
    with pytest.raises(ValueError, match="length 3, got 4"):
        xyz_to_quadray_canonical((1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="shape"):
        xyz_to_quadray_canonical((0.0, 0.0, 0.0), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="does not sum to exactly 0"):
        xyz_to_quadray_canonical((1.0, 1.0, 1.0), [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]])
    with pytest.raises(TypeError, match="got str"):
        xyz_to_quadray_canonical("abc")
    with pytest.raises(TypeError, match="int, float, Fraction"):
        xyz_to_quadray_canonical((np.float32(1.0), 1.0, 1.0))


# --- §6: The float inverse and the round-trip contract -----------------------


def test_roundtrip_identity_shells_0_to_4():
    """S26: quadray_roundtrip is the identity on all 309 canonical sites."""
    sites = [q for k in range(5) for q in shell_sites(k)]
    assert len(sites) == 309
    for q in sites:
        assert quadray_roundtrip(q) == q


def test_roundtrip_scale_robustness():
    """S27: exact at Urner scales 1/2 and 2; pinv(cM)(cM q) = q - (sum q/4)*1."""
    sites = [q for k in range(5) for q in shell_sites(k)]
    for c in (0.5, 2.0):
        m = urner_embedding(c)
        for q in sites:
            assert quadray_roundtrip(q, m) == q
    for c in (0.5, 2.0):
        m = urner_embedding(c)
        pinv = np.linalg.pinv(m)
        for q in (Quadray(2, 1, 1, 0), Quadray(3, 0, 1, 0), Quadray(1, 1, 1, 0)):
            projection = pinv @ np.asarray(quadray_to_xyz(q, m))
            expected = np.asarray(q.as_tuple(), dtype=np.float64) - sum(q.as_tuple()) / 4.0
            assert np.allclose(projection, expected, rtol=0.0, atol=1e-12)


def test_roundtrip_documented_failure_unnormalized():
    """S28: unnormalized input raises AssertionError; (2,2,2,1) -> (1,1,1,0)."""
    with pytest.raises(AssertionError, match="roundtrip is not the identity"):
        quadray_roundtrip(Quadray(2, 2, 2, 1))


def test_quadray_from_xyz_coset_preserving_roundtrip():
    """S29: round-half-up preserves the (1,1,1,1)-coset at scales 1 and 1/2."""
    classes = (
        Quadray(1, 1, 0, 0),  # sum ≡ 2 (mod 4): exact half-integer ties
        Quadray(1, 0, 0, 0),  # sum ≡ 1 (mod 4)
        Quadray(1, 1, 1, 0),  # sum ≡ 3 (mod 4)
        Quadray(0, 1, 1, 2),  # IVM site, sum ≡ 0 (mod 4)
        Quadray(2, 2, 2, 1),  # unnormalized representative of (1,1,1,0)
    )
    for c in (1.0, 0.5):
        m = urner_embedding(c)
        for q in classes:
            x, y, z = to_xyz(q, m)
            assert quadray_from_xyz(x, y, z, m) == q.normalize()


def test_quadray_from_xyz_not_always_xyz_nearest():
    """S30: component-wise quadray-nearest is not always XYZ-nearest."""
    xyz = (-0.25, 0.75, 0.75)
    assert quadray_from_xyz(xyz[0], xyz[1], xyz[2], DEFAULT_EMBEDDING) == Quadray(0, 0, 0, 0)
    best_d2, best = _brute_force_nearest(xyz)
    assert best == (1, 1, 1, 0)
    assert best_d2 == pytest.approx(11.0 / 16.0, abs=1e-12)
    assert 11.0 / 16.0 < 19.0 / 16.0
