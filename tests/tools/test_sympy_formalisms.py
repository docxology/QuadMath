"""Independent symbolic identities and native/length volume agreement."""
import csv
from fractions import Fraction

from sympy import Rational, simplify, sqrt, symbols, sympify

from sympy_formalisms import (
    cayley_menger_symbolic_unit_tetra,
    compare_ace_vs_cm_examples,
    embedding_symbolic_magnitude,
    magnitude_via_vector_module,
)


def test_symbolic_unit_edge_volume_uses_sphere_radius_one_units():
    xyz, ivm = map(sympify, cayley_menger_symbolic_unit_tetra())
    assert simplify(xyz - sqrt(2) / 12) == 0
    assert ivm == Rational(1, 8)


def test_symbolic_magnitudes_match_the_actual_embedding_quadratic_forms():
    a, b, c, d, x, y, z = symbols('a b c d x y z', real=True)
    local = {str(value): value for value in (a, b, c, d, x, y, z)}
    native = sympify(embedding_symbolic_magnitude(), locals=local)
    xyz = sympify(magnitude_via_vector_module(), locals=local)
    assert simplify(native**2 - (4 * (a*a + b*b + c*c + d*d) - (a+b+c+d)**2)) == 0
    assert simplify(xyz**2 - x*x - y*y - z*z) == 0


def test_native_and_length_formulas_agree_in_exported_comparisons(tmp_path, monkeypatch):
    monkeypatch.setenv('QUADMATH_OUTPUT_DIR', str(tmp_path))
    with open(compare_ace_vs_cm_examples(), encoding='utf-8') as stream:
        rows = list(csv.reader(stream))[1:]
    for case, native, bridged, matches in rows:
        assert Fraction(native) == Fraction(bridged), case
        assert matches == 'True'
    assert [(row[0], Fraction(row[1])) for row in rows[:2]] == [
        ('unit_tetrahedron', Fraction(1)), ('scaled_unit_tetrahedron', Fraction(8)),
    ]
