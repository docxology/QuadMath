"""Tests for quadmath/validate/validate.py.

Pass cases use unit quaternions built from the Quadray lattice constructors
(the basis points used by the core examples); failure cases feed deliberately
corrupted values (scaled non-unit quaternions, mutated copies carrying
non-finite components) so both branches of every check execute.  All
expectations are exact or float-deterministic; no randomness, no mocks.
"""

import dataclasses
import math

import pytest

from quadmath.core.quadray import Quadray
from quadmath.validate import (
    DEFAULT_CHECKS,
    NOTES,
    SKIPPED_CHECKS,
    ValidationReport,
    check_associativity,
    check_conjugate_inverse,
    check_double_cover,
    check_normalization,
    check_slerp_midpoint,
    run_validation,
)

# Unit quaternions on the quadray lattice: (a, b, c, d) read as (w, x, y, z)
E = Quadray(1, 0, 0, 0)  # identity, w = 1
I = Quadray(0, 1, 0, 0)  # i
J = Quadray(0, 0, 1, 0)  # j
K = Quadray(0, 0, 0, 1)  # k


def _scaled():
    """Deliberately corrupted value: non-unit quaternion 2*e (norm 2)."""
    return Quadray(2, 0, 0, 0)


def _nan_mutated():
    """Deliberately corrupted value: mutated copy with a NaN component.

    Quadray does not enforce its int field types, so the mutation is
    constructible; non-finite components must surface as check failures.
    """
    return Quadray(1, float("nan"), 0, 0)


def test_validation_report_is_frozen():
    report = ValidationReport("check_example", True, "ok")
    assert (report.name, report.passed, report.detail) == ("check_example", True, "ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.passed = False


def test_check_normalization_passes_for_unit_lattice_quaternions():
    for q in (E, I, J, K):
        report = check_normalization(q)
        assert isinstance(report, ValidationReport)
        assert report.passed is True
        assert report.name == "check_normalization"
        assert report.detail


def test_check_normalization_failure_detail_includes_norm():
    report = check_normalization(_scaled())
    assert report.passed is False
    assert report.name == "check_normalization"
    assert "2.0" in report.detail  # the measured norm 2.0 must be reported


def test_check_normalization_tolerance_is_parameterized():
    assert check_normalization(_scaled(), tol=1.5).passed is True
    assert check_normalization(_scaled()).passed is False


def test_check_conjugate_inverse_passes_for_unit_quaternions():
    for q in (E, I, J, K):
        report = check_conjugate_inverse(q)
        assert report.passed is True
        assert report.name == "check_conjugate_inverse"
        assert report.detail


def test_check_conjugate_inverse_fails_for_scaled_quaternion():
    report = check_conjugate_inverse(_scaled())
    assert report.passed is False
    assert "4.0" in report.detail  # q * conj(q) = 4 * identity for 2*e


def test_check_double_cover_holds_for_basis_and_scaled_pairs():
    # i*j = k exactly, so R(i)R(j) == R(k) == R(i*j) under the documented
    # active right-hand convention
    assert check_double_cover(I, J).passed is True
    assert check_double_cover(E, K).passed is True
    # Double cover: positive rescalings represent the same rotations, so a
    # scaled pair still satisfies the homomorphism on unit representatives
    assert check_double_cover(_scaled(), Quadray(0, 0, 3, 0)).passed is True


def test_check_double_cover_failure_report_for_nan_mutation():
    report = check_double_cover(_nan_mutated(), E)
    assert report.passed is False
    assert report.name == "check_double_cover"
    assert report.detail


def test_check_double_cover_rejects_zero_norm():
    with pytest.raises(ValueError):
        check_double_cover(Quadray(0, 0, 0, 0), E)


def test_check_slerp_midpoint_passes_on_shared_hemisphere_pairs():
    # dot(e, i) = 0 and dot(i, j) = 0: the shortest arc and the written arc
    # coincide; identical quaternions degenerate to the trivial midpoint
    for q0, q1 in ((E, I), (I, J), (E, E)):
        report = check_slerp_midpoint(q0, q1)
        assert report.passed is True
        assert report.name == "check_slerp_midpoint"
        assert report.detail


def test_check_slerp_midpoint_fails_for_opposite_hemisphere_pair():
    # Unit q1 with dot(e, q1) = -cos(pi/6) < 0: the shortest-arc midpoint is
    # equidistant from -q1 instead, so angle(m, q1) = pi - angle(q0, m)
    q1 = Quadray(-math.sqrt(3.0) / 2.0, 0.5, 0, 0)
    report = check_slerp_midpoint(E, q1)
    assert report.passed is False
    assert report.detail


def test_check_slerp_midpoint_propagates_non_unit_error():
    with pytest.raises(ValueError, match="unit"):
        check_slerp_midpoint(E, _scaled())


def test_check_slerp_midpoint_rejects_non_finite_components():
    with pytest.raises(ValueError, match="non-finite"):
        check_slerp_midpoint(E, _nan_mutated())


def test_check_associativity_holds_for_basis_triple():
    # (i*j)*k = k*k = -e and i*(j*k) = i*i = -e: exact integer arithmetic
    report = check_associativity(I, J, K)
    assert report.passed is True
    assert "-1.0" in report.detail  # the measured product (-1, 0, 0, 0)


def test_check_associativity_fails_for_nan_mutation():
    report = check_associativity(E, _nan_mutated(), I)
    assert report.passed is False
    assert report.name == "check_associativity"
    assert report.detail


def test_exp_log_roundtrip_is_documented_as_skipped():
    assert SKIPPED_CHECKS == ("check_exp_log_roundtrip",)
    assert "check_exp_log_roundtrip" not in [c.__name__ for c in DEFAULT_CHECKS]
    assert any("check_exp_log_roundtrip" in note for note in NOTES)
    assert all(note for note in NOTES)


def test_run_validation_default_order_is_stable():
    reports = run_validation([E, I, J])
    expected_names = (
        ["check_normalization"] * 3
        + ["check_conjugate_inverse"] * 3
        + ["check_double_cover"] * 3
        + ["check_slerp_midpoint"] * 3
        + ["check_associativity"]
    )
    assert [r.name for r in reports] == expected_names
    assert all(r.passed for r in reports)


def test_run_validation_records_failures_for_corrupted_quaternion():
    reports = run_validation([E, _scaled()])
    outcomes = [(r.name, r.passed) for r in reports]
    assert outcomes == [
        ("check_normalization", True),
        ("check_normalization", False),
        ("check_conjugate_inverse", True),
        ("check_conjugate_inverse", False),
        ("check_double_cover", True),
        ("check_slerp_midpoint", False),
    ]
    # The slerp failure is the non-unit ValueError converted by the runner
    assert "raised ValueError" in reports[-1].detail


def test_run_validation_accepts_custom_checks():
    reports = run_validation([E, I], checks=[check_normalization])
    assert [r.name for r in reports] == ["check_normalization", "check_normalization"]
    assert all(r.passed for r in reports)


def test_run_validation_rejects_unsupported_arity():
    def four_operand(a, b, c, d):
        return check_normalization(a)

    with pytest.raises(ValueError, match="1-3"):
        run_validation([E], checks=[four_operand])


def test_run_validation_rejects_variadic_checks():
    def variadic(*quads):
        return check_normalization(quads[0])

    with pytest.raises(ValueError, match="fixed positional"):
        run_validation([E], checks=[variadic])


def test_run_validation_propagates_non_value_error_exceptions():
    def broken(_q):
        raise RuntimeError("not a validation failure")

    with pytest.raises(RuntimeError, match="not a validation failure"):
        run_validation([E], checks=[broken])


def test_run_validation_empty_quaternions_yields_no_reports():
    assert run_validation([]) == []
