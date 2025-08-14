import os
import tempfile
from typing import List

import numpy as np
import ast

from information import (
    finite_difference_gradient,
    perception_update,
    action_update,
)
from glossary_gen import build_api_index, generate_markdown_table, inject_between_markers
from glossary_gen import _first_line, _format_function_signature, _is_constant  # type: ignore


def test_finite_difference_gradient_matches_quadratic():
    def f(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    x = np.array([1.0, -3.0, 2.5])
    grad = finite_difference_gradient(f, x)
    assert np.allclose(grad, 2.0 * x, rtol=1e-5, atol=1e-5)


def test_finite_difference_gradient_bad_shape():
    def f(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    x = np.array([[1.0, 2.0]])
    try:
        finite_difference_gradient(f, x)
        assert False
    except ValueError:
        assert True


def test_perception_and_action_updates():
    # Quadratic free-energy: F(z) = 0.5 * ||z||^2
    def F_mu(mu: np.ndarray) -> float:
        return 0.5 * float(np.dot(mu, mu))

    def D_op(mu: np.ndarray) -> np.ndarray:
        return np.zeros_like(mu)

    mu = np.array([1.0, -2.0])
    dmu = perception_update(mu, D_op, F_mu, step_size=1.0)
    assert np.allclose(dmu, -mu)

    def F_a(a: np.ndarray) -> float:
        return 0.5 * float(np.dot(a, a))

    a = np.array([0.5, -0.25, 1.5])
    da = action_update(a, F_a, step_size=0.2)
    assert np.allclose(da, -0.2 * a)


def test_perception_and_action_shape_checks():
    def F(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    def D(mu: np.ndarray) -> np.ndarray:
        return mu

    try:
        perception_update(np.zeros((1, 2)), D, F)
        assert False
    except ValueError:
        assert True

    try:
        action_update(np.zeros((1, 2)), F)
        assert False
    except ValueError:
        assert True


def _write(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def test_glossary_gen_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.join(tmp, "srcpkg")
        os.makedirs(src_dir, exist_ok=True)
        # module with function, class, constant and annotated constant
        mod_a = os.path.join(src_dir, "a.py")
        _write(
            mod_a,
            [
                "\"\"\"Module A\"\"\"",
                "VALUE = 3",
                "OTHER: int = 5",
                "def foo(x, y, **kw):\n    \"\"\"add\"\"\"\n    return x + y",
                "class C:\n    \"\"\"klass\"\"\"\n    pass",
            ],
        )
        # module with private functions and ignored
        mod_b = os.path.join(src_dir, "b.py")
        _write(
            mod_b,
            [
                "def _hidden():\n    return 0",
                "CONST = 1",
            ],
        )
        # Non-Python file should be ignored
        _write(os.path.join(src_dir, "notes.txt"), ["ignore me"])
        # File with a syntax error should be safely skipped
        _write(os.path.join(src_dir, "bad.py"), ["def bad(:\n  pass"])
        # File with tuple assignment and lowercase constant (should not be indexed)
        mod_c = os.path.join(src_dir, "c.py")
        _write(
            mod_c,
            [
                "(A, B) = (1, 2)",  # targets are Tuple, not Name
                "lower = 7",        # lowercase name should be ignored by _is_constant
            ],
        )

        entries = build_api_index(src_dir)
        # Expect function foo, class C, constants VALUE, OTHER, CONST
        names = sorted([(e.module, e.name, e.kind) for e in entries])
        assert ("a", "foo", "function") in [(m.split(".")[-1], n, k) for m, n, k in names]
        assert any(k == "class" and n == "C" for _m, n, k in names)
        assert sum(1 for _m, n, k in names if k == "constant") >= 2
        # Ensure lowercase and tuple targets were not added
        assert not any(n == "lower" for _m, n, _k in names)
        assert not any(n in ("A", "B") for _m, n, _k in names)

        md = generate_markdown_table(entries)
        assert "| Module | Symbol |" in md
        assert "`foo`" in md and "`C`" in md

        # injection with present markers
        text = "pre\n<!--BEGIN-->\nold\n<!--END-->\npost"
        out = inject_between_markers(text, "<!--BEGIN-->", "<!--END-->", "PAYLOAD")
        assert "PAYLOAD" in out and "old" not in out

        # injection when markers are absent appends them
        text2 = "no markers here"
        out2 = inject_between_markers(text2, "<A>", "</A>", "X")
        assert "<A>" in out2 and "</A>" in out2 and "X" in out2

        # case where begin present but end missing should also fall back to append
        text3 = "prefix\n<A> present"
        out3 = inject_between_markers(text3, "<A>", "</A>", "Y")
        assert out3.rstrip().endswith("</A>") and "Y" in out3


def test_glossary_private_helpers_and_signatures():
    # _first_line behavior with None and with multi-line
    assert _first_line(None) == ""
    assert _first_line("line1\nline2") == "line1"

    # Build AST for a function with varargs and kwargs
    code1 = "def f(a, *args, **kw):\n    pass\n"
    node1 = [n for n in ast.parse(code1).body if isinstance(n, ast.FunctionDef)][0]
    sig1 = _format_function_signature(node1)
    assert "a" in sig1 and "*args" in sig1 and "**kw" in sig1

    # Build AST for a function with keyword-only argument to exercise that branch
    code2 = "def g(*, b):\n    pass\n"
    node2 = [n for n in ast.parse(code2).body if isinstance(n, ast.FunctionDef)][0]
    sig2 = _format_function_signature(node2)
    assert "b=" in sig2

    # _is_constant only true for ALL_CAPS Assign/AnnAssign
    assign_node = ast.parse("X = 1").body[0]
    annassign_node = ast.parse("Y: int = 2").body[0]
    lower_assign = ast.parse("z = 1").body[0]
    assert _is_constant("X", assign_node) is True
    assert _is_constant("Y", annassign_node) is True
    assert _is_constant("z", lower_assign) is False


