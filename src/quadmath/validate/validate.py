"""Deterministic quaternion-algebra validation checks for QuadMath.

This layer treats :class:`~quadmath.core.quadray.Quadray` values as
quaternions: the components ``(a, b, c, d)`` are read in order as the
scalar-first ``(w, x, y, z)`` tuple used by :func:`qmul`, :func:`qconjugate`
and :func:`slerp` in :mod:`quadmath.core.quadray` (``w = a``, ``x = b``,
``y = c``, ``z = d``).  Those helpers are representation-independent; the
``(a, b, c, d) -> (w, x, y, z)`` reading is a convention of this module and
is applied uniformly by every check.

Each ``check_*`` function takes the value(s) under test plus an absolute
tolerance and returns a :class:`ValidationReport`.  Nothing is printed and
no module-level state changes: identical inputs always yield identical
reports, so runs are byte-for-byte reproducible.

Rotation matrices are built privately in this module (see
:func:`check_double_cover`) with the fixed convention documented there.
Quaternion ``exp``/``log`` are intentionally not fabricated here: the core
layer exposes none, so the exp/log round-trip check is skipped entirely
(see ``SKIPPED_CHECKS`` and ``NOTES``).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from quadmath.core.quadray import Quadray, qconjugate, qmul, slerp

DEFAULT_TOLERANCE: float = 1e-9

SKIPPED_CHECKS: Tuple[str, ...] = ("check_exp_log_roundtrip",)

NOTES: Tuple[str, ...] = (
    "check_exp_log_roundtrip is skipped: quadmath.core.quadray exposes no "
    "quaternion exp/log, and fabricating them in the validation layer is "
    "out of scope. It is excluded from DEFAULT_CHECKS and never run by "
    "run_validation.",
)


@dataclass(frozen=True)
class ValidationReport:
    """Immutable outcome of a single validation check.

    Parameters
    - name: Check identifier (the check function's __name__)
    - passed: True when the check succeeded within tolerance
    - detail: Deterministic, human-readable description of the outcome that
      includes the measured quantities (e.g. the quaternion norm)
    """

    name: str
    passed: bool
    detail: str


def _components(q: Quadray) -> Tuple[float, float, float, float]:
    """Return the (w, x, y, z) float view of a Quadray (a, b, c, d)."""
    a, b, c, d = q.as_tuple()
    return (float(a), float(b), float(c), float(d))


def _quat_norm(q4: Tuple[float, float, float, float]) -> float:
    """Return the Euclidean norm of a (w, x, y, z) quaternion tuple."""
    return math.sqrt(sum(c * c for c in q4))


def _unit(q4: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Return the unit rescaling of a (w, x, y, z) quaternion tuple.

    Parameters
    - q4: Quaternion as (w, x, y, z) tuple of four numbers

    Returns
    - Tuple[float, float, float, float]: q4 / |q4|; a non-finite norm flows
      through (components become NaN) so callers can report the corruption

    Raises
    - ValueError: If q4 has zero norm (no SO(3) representative exists)
    """
    n = _quat_norm(q4)
    if n == 0.0:
        raise ValueError("zero-norm quaternion has no SO(3) rotation representative")
    return (q4[0] / n, q4[1] / n, q4[2] / n, q4[3] / n)


def _rotation_matrix(q4: Tuple[float, float, float, float]) -> np.ndarray:
    """SO(3) rotation matrix of a UNIT quaternion (w, x, y, z).

    Fixed convention: active rotation of column vectors, ``v' = q v q*``
    with the Hamilton product, right-handed about the rotation axis
    (counterclockwise when viewed from the tip of the axis toward the
    origin).  Row-major 3x3 float array; assumes |q4| = 1 (callers
    normalize via :func:`_unit` first).
    """
    w, x, y, z = q4
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _signed_angle(
    u4: Tuple[float, float, float, float], v4: Tuple[float, float, float, float]
) -> float:
    """Geodesic angle in [0, pi] between quaternions as R^4 vectors.

    Uses the signed cosine (no double-cover folding): q and -q sit at
    angle pi, not 0.  Callers pass non-zero quaternions.

    Parameters
    - u4, v4: Quaternions as (w, x, y, z) tuples of four numbers

    Returns
    - float: arccos of the clamped cosine of the normalized inner product

    Raises
    - ValueError: If the cosine is non-finite (corrupted components)
    """
    cos = sum(a * b for a, b in zip(u4, v4)) / (_quat_norm(u4) * _quat_norm(v4))
    if not math.isfinite(cos):
        raise ValueError("quaternion cosine is non-finite; components are corrupted")
    cos = max(-1.0, min(1.0, cos))
    return math.acos(cos)


def check_normalization(q: Quadray, tol: float = DEFAULT_TOLERANCE) -> ValidationReport:
    """Verify |q| is within ``tol`` of 1 (quaternion norm of (a, b, c, d)).

    Parameters
    - q: Quadray read as a quaternion (w, x, y, z) = (a, b, c, d)
    - tol: Absolute tolerance on the norm

    Returns
    - ValidationReport: passed=True when the norm is within tol of 1; the
      failure detail includes the measured norm
    """
    norm = _quat_norm(_components(q))
    deviation = abs(norm - 1.0)
    if deviation <= tol:
        return ValidationReport(
            "check_normalization", True, f"|q| = {norm!r} within tol {tol!r} of 1"
        )
    return ValidationReport(
        "check_normalization",
        False,
        f"|q| = {norm!r} deviates from 1 by {deviation!r} (tol {tol!r})",
    )


def check_conjugate_inverse(
    q: Quadray, tol: float = DEFAULT_TOLERANCE
) -> ValidationReport:
    """Verify q * conj(q) equals the identity (1, 0, 0, 0) within ``tol``.

    Uses :func:`qmul` and :func:`qconjugate` from quadmath.core.quadray.
    The identity holds exactly when |q| = 1; a scaled quaternion yields
    |q|^2 times the identity, and non-finite components fail the comparison.

    Parameters
    - q: Quadray read as a quaternion (w, x, y, z) = (a, b, c, d)
    - tol: Absolute tolerance applied componentwise

    Returns
    - ValidationReport: passed=True when q * conj(q) matches the identity
      within tol; the measured product is reported either way
    """
    q4 = _components(q)
    product = qmul(q4, qconjugate(q4))
    identity = (1.0, 0.0, 0.0, 0.0)
    if np.allclose(product, identity, rtol=0.0, atol=tol):
        return ValidationReport(
            "check_conjugate_inverse",
            True,
            f"q*qconjugate(q) = {product!r} equals the identity within tol {tol!r}",
        )
    return ValidationReport(
        "check_conjugate_inverse",
        False,
        f"q*qconjugate(q) = {product!r} differs from the identity "
        f"(1.0, 0.0, 0.0, 0.0) beyond tol {tol!r}",
    )


def check_double_cover(
    q1: Quadray, q2: Quadray, tol: float = DEFAULT_TOLERANCE
) -> ValidationReport:
    """Verify the SO(3) homomorphism R(q1*q2) == R(q1) R(q2) within ``tol``.

    Rotation matrices are built privately in this module.  Fixed convention:
    each quaternion is first rescaled to unit norm (its double-cover class
    representative, so q, -q and any positive rescaling give the same
    matrix); then R maps a unit quaternion (w, x, y, z) to the active,
    right-handed rotation matrix of ``v' = q v q*`` for column vectors (see
    :func:`_rotation_matrix`).  Under this convention R(q1*q2) ==
    R(q1) @ R(q2) holds for every finite non-zero pair, so only corrupted
    (non-finite) values can break the identity; zero-norm quaternions have
    no rotation representative and are rejected.

    Parameters
    - q1, q2: Quadray values read as quaternions (w, x, y, z) = (a, b, c, d)
    - tol: Absolute tolerance applied elementwise to the matrix comparison

    Returns
    - ValidationReport: passed=True when the homomorphism holds within tol;
      the maximum elementwise deviation is reported either way

    Raises
    - ValueError: If either quaternion has zero norm
    """
    u1 = _unit(_components(q1))
    u2 = _unit(_components(q2))
    lhs = _rotation_matrix(qmul(u1, u2))
    rhs = _rotation_matrix(u1) @ _rotation_matrix(u2)
    max_dev = float(np.max(np.abs(lhs - rhs)))
    if np.allclose(lhs, rhs, rtol=0.0, atol=tol):
        return ValidationReport(
            "check_double_cover",
            True,
            f"R(q1*q2) equals R(q1)R(q2) within tol {tol!r} "
            f"(max |deviation| = {max_dev!r})",
        )
    return ValidationReport(
        "check_double_cover",
        False,
        f"R(q1*q2) differs from R(q1)R(q2) by {max_dev!r} beyond tol {tol!r}",
    )


def check_slerp_midpoint(
    q0: Quadray, q1: Quadray, tol: float = DEFAULT_TOLERANCE
) -> ValidationReport:
    """Verify the shortest-arc slerp midpoint lies on the geodesic of (q0, q1).

    Computes m = slerp(q0, q1, 0.5) reusing :func:`slerp` from
    quadmath.core.quadray (which takes the shorter arc by negating q1 when
    <q0, q1> < 0), then verifies angle(q0, m) == angle(m, q1) within
    ``tol``, where angle is the signed R^4 geodesic angle between the
    quaternions as written.  For <q0, q1> >= 0 the midpoint is equidistant
    from both written endpoints and the check passes.  For <q0, q1> < 0 the
    shortest-arc midpoint is equidistant from -q1 instead, so
    angle(m, q1) = pi - angle(q0, m) and the check reports a failure: the
    two written representatives do not lie on a common short arc (a
    representative-consistency defect of the inputs, not of slerp).

    Parameters
    - q0, q1: Unit Quadray quaternions (w, x, y, z) = (a, b, c, d)
    - tol: Absolute tolerance on the angle difference

    Returns
    - ValidationReport: passed=True when both geodesic angles agree within
      tol; both angles are reported either way

    Raises
    - ValueError: Propagated from slerp when either quaternion is not unit
      (|q| = 1 within 1e-9), or raised here when the quaternion cosine is
      non-finite (corrupted components)
    """
    q0_4 = _components(q0)
    q1_4 = _components(q1)
    midpoint = slerp(q0_4, q1_4, 0.5)
    angle_q0_m = _signed_angle(q0_4, midpoint)
    angle_m_q1 = _signed_angle(midpoint, q1_4)
    deviation = abs(angle_q0_m - angle_m_q1)
    if deviation <= tol:
        return ValidationReport(
            "check_slerp_midpoint",
            True,
            f"angle(q0, m) = {angle_q0_m!r} equals angle(m, q1) = "
            f"{angle_m_q1!r} within tol {tol!r}",
        )
    return ValidationReport(
        "check_slerp_midpoint",
        False,
        f"angle(q0, m) = {angle_q0_m!r} differs from angle(m, q1) = "
        f"{angle_m_q1!r} by {deviation!r} beyond tol {tol!r}",
    )


def check_associativity(
    a: Quadray, b: Quadray, c: Quadray, tol: float = DEFAULT_TOLERANCE
) -> ValidationReport:
    """Verify (a*b)*c == a*(b*c) within ``tol`` via the core Hamilton product.

    Associativity is an exact structural identity of the quaternion algebra,
    so it holds for every finite triple at any sane tolerance; only
    corrupted (non-finite) values can break it.

    Parameters
    - a, b, c: Quadray values read as quaternions (w, x, y, z) = (a, b, c, d)
    - tol: Absolute tolerance applied componentwise

    Returns
    - ValidationReport: passed=True when both groupings agree within tol;
      the measured left-hand product is reported either way
    """
    a4 = _components(a)
    b4 = _components(b)
    c4 = _components(c)
    left = qmul(qmul(a4, b4), c4)
    right = qmul(a4, qmul(b4, c4))
    if np.allclose(left, right, rtol=0.0, atol=tol):
        return ValidationReport(
            "check_associativity",
            True,
            f"(a*b)*c = {left!r} equals a*(b*c) within tol {tol!r}",
        )
    return ValidationReport(
        "check_associativity",
        False,
        f"(a*b)*c = {left!r} differs from a*(b*c) = {right!r} beyond tol {tol!r}",
    )


DEFAULT_CHECKS: Tuple[Callable[..., ValidationReport], ...] = (
    check_normalization,
    check_conjugate_inverse,
    check_double_cover,
    check_slerp_midpoint,
    check_associativity,
)


def _required_positional_arity(check: Callable[..., ValidationReport]) -> int:
    """Return the number of required positional operands of a check (1-3).

    Parameters
    - check: Callable returning a ValidationReport

    Returns
    - int: Count of required positional parameters (1, 2, or 3); parameters
      with defaults (e.g. tol) are not counted

    Raises
    - ValueError: If the check uses *args/**kwargs or does not declare
      between one and three required positional parameters
    """
    name = getattr(check, "__name__", repr(check))
    arity = 0
    for parameter in inspect.signature(check).parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                f"check {name!r} must declare fixed positional operands, "
                "not *args/**kwargs"
            )
        if (
            parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ):
            arity += 1
    if not 1 <= arity <= 3:
        raise ValueError(
            f"check {name!r} must take 1-3 required positional operands, got {arity}"
        )
    return arity


def _operand_groups(quads: List[Quadray], arity: int) -> List[Tuple[Quadray, ...]]:
    """Deterministic operand tuples for one check over ``quads``.

    Arity 1 -> each quaternion in given order; arity 2 -> each pair i < j;
    arity 3 -> each triple i < j < k, all lexicographic.

    Parameters
    - quads: Quadray operands in run order
    - arity: Required positional arity of the check (1-3)

    Returns
    - List[Tuple[Quadray, ...]]: Operand tuples in deterministic order
    """
    if arity == 1:
        return [(q,) for q in quads]
    if arity == 2:
        return [
            (quads[i], quads[j])
            for i in range(len(quads))
            for j in range(i + 1, len(quads))
        ]
    return [
        (quads[i], quads[j], quads[k])
        for i in range(len(quads))
        for j in range(i + 1, len(quads))
        for k in range(j + 1, len(quads))
    ]


def run_validation(
    quaternions: Sequence[Quadray],
    checks: Optional[Sequence[Callable[..., ValidationReport]]] = None,
) -> List[ValidationReport]:
    """Run checks over ``quaternions`` and collect deterministic reports.

    Default ``checks`` is :data:`DEFAULT_CHECKS` (every implemented check;
    the skipped exp/log round-trip is never included, see
    ``SKIPPED_CHECKS`` and ``NOTES``).  Each check is dispatched by its
    number of required positional operands: single-operand checks run once
    per quaternion, two-operand checks on every pair i < j, three-operand
    checks on every triple i < j < k, in the given quaternion order.
    Reports are appended check-major (all reports of the first check, then
    the second, ...), so the ordering is stable.  A check that raises
    ValueError is recorded as a failed report whose detail carries the
    exception message; any other exception propagates.

    Parameters
    - quaternions: Quadray values read as quaternions (w, x, y, z) = (a, b, c, d)
    - checks: Optional sequence of callables returning ValidationReport;
      each must declare 1-3 required positional operands

    Returns
    - List[ValidationReport]: One report per (check, operand tuple) in
      deterministic order; empty when there are no quaternions or no checks

    Raises
    - ValueError: If a check declares an unsupported signature
    """
    selected = tuple(DEFAULT_CHECKS) if checks is None else tuple(checks)
    quads = list(quaternions)
    reports: List[ValidationReport] = []
    for check in selected:
        name = getattr(check, "__name__", repr(check))
        for operands in _operand_groups(quads, _required_positional_arity(check)):
            try:
                reports.append(check(*operands))
            except ValueError as exc:
                reports.append(ValidationReport(name, False, f"raised ValueError: {exc}"))
    return reports


__all__ = [
    "DEFAULT_TOLERANCE",
    "DEFAULT_CHECKS",
    "SKIPPED_CHECKS",
    "NOTES",
    "ValidationReport",
    "check_normalization",
    "check_conjugate_inverse",
    "check_double_cover",
    "check_slerp_midpoint",
    "check_associativity",
    "run_validation",
]
