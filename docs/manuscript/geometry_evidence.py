"""Record exact and distance-based volumes for fixed Quadray tetrahedra."""

from __future__ import annotations

from fractions import Fraction
import json
from math import isclose
from pathlib import Path

from quadmath.core.cayley_menger import (
    ivm_tetra_volume_cayley_menger,
    squared_distances_from_quadrays,
)
from quadmath.core.quadray import (
    DEFAULT_EMBEDDING,
    Quadray,
    ace_tetravolume_5x5,
    integer_tetra_volume,
)


def main() -> None:
    origin = (0, 0, 0, 0)
    primitive = (origin, (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
    unit = (origin, (2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 2, 0))
    doubled = tuple(tuple(2 * coefficient for coefficient in vertex) for vertex in unit)
    shifted = tuple(
        tuple(coefficient + offset for coefficient in vertex)
        for vertex, offset in zip(primitive, (0, 3, -2, 5), strict=True)
    )
    cases = (
        ("primitive", primitive, Fraction(1, 4)),
        ("unit IVM", unit, Fraction(1)),
        ("doubled unit IVM", doubled, Fraction(8)),
        ("primitive with independent common offsets", shifted, Fraction(1, 4)),
    )
    rows = []
    for name, vertices, expected in cases:
        points = tuple(Quadray(*vertex) for vertex in vertices)
        projected = integer_tetra_volume(*points)
        ace = ace_tetravolume_5x5(*points)
        # Squared distances in the sphere-radius-one convention are half
        # the default scale-one XYZ distances squared.
        d2 = squared_distances_from_quadrays(*points, DEFAULT_EMBEDDING) / 2
        bridged = ivm_tetra_volume_cayley_menger(d2)
        if projected != expected or ace != expected or not isclose(
            bridged, float(expected), rel_tol=1e-12
        ):
            raise ValueError(f"Volume disagreement for {name}: {projected}, {ace}, {bridged}")
        rows.append(
            {
                "case": name,
                "quadrays": vertices,
                "ivm_volume_exact": str(projected),
                "ivm_volume_numeric": bridged,
            }
        )
    path = Path("output/data/tetra-volume-fixture.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": rows}, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
