#!/usr/bin/env python3
"""Symbolic demonstrations (SymPy) for Quadray formalisms.

- Cayley–Menger (Euclidean, Coxeter.4D) symbolic volume with exact radicals
- S3 conversion to IVM (Fuller.4D)
- Example dot/magnitude via embedding matrix in symbolic form
- Bridging vs native volumes: compare Tom Ace 5×5 (IVM) vs CM+S3 across examples

Outputs:
- sympy_symbolics.txt: key expressions
- bridging_vs_native.csv: per-example comparison table
- bridging_vs_native.png: bar chart comparing V_ivm (Ace vs CM+S3)
"""
from __future__ import annotations

import os
import sys
from typing import Tuple


def _ensure_src_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _get_output_dir() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = os.path.join(repo_root, "quadmath", "output")
    os.makedirs(out, exist_ok=True)
    return out


def _get_data_dir() -> str:
    out = _get_output_dir()
    data_dir = os.path.join(out, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _get_figure_dir() -> str:
    out = _get_output_dir()
    figure_dir = os.path.join(out, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def cayley_menger_symbolic_unit_tetra() -> Tuple[str, str]:
    """Return simplified symbolic V_xyz and V_ivm for unit-edge regular tetra.

    Uses Cayley–Menger with squared distances set to 1 off-diagonal.
    """
    from sympy import Matrix, sqrt, simplify

    CM = Matrix(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )
    det = CM.det()
    V2 = det / 288
    V_xyz = simplify(sqrt(V2))
    S3 = sqrt(9) / sqrt(8)
    V_ivm = simplify(S3 * V_xyz)
    return (str(V_xyz), str(V_ivm))


def embedding_symbolic_magnitude() -> str:
    """Symbolic magnitude for a generic Quadray vector (a,b,c,d) under Urner embedding."""
    from sympy import Matrix, symbols, sqrt

    a, b, c, d = symbols("a b c d", real=True)
    M = Matrix([[1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1]])
    v = Matrix([a, b, c, d])
    x, y, z = (M @ v)
    mag = sqrt(x * x + y * y + z * z)
    return str(mag)


def compare_ace_vs_cm_examples() -> str:
    """Compare Ace 5x5 (IVM) vs CM+S3 on a few integer Quadray tetra examples.

    Returns
    - str: path to generated CSV file
    """
    from sympy import Matrix, sqrt, simplify, N
    import csv
    import matplotlib.pyplot as plt

    _ensure_src_on_path()
    # Import numeric Ace and embedding from src
    from quadray import Quadray, ace_tetravolume_5x5, DEFAULT_EMBEDDING
    from symbolic import cayley_menger_volume_symbolic, convert_xyz_volume_to_ivm_symbolic

    M = Matrix(DEFAULT_EMBEDDING)
    # Deterministic small examples (non-degenerate, simple)
    examples = [
        # unit tetra from origin-like set
        (Quadray(0, 0, 0, 0), Quadray(2, 1, 0, 1), Quadray(2, 1, 1, 0), Quadray(2, 0, 1, 1)),
        # scaled variant (should scale volume by 8)
        (Quadray(0, 0, 0, 0), Quadray(4, 2, 0, 2), Quadray(4, 2, 2, 0), Quadray(4, 0, 2, 2)),
        # a mixed tetra
        (Quadray(1, 0, 0, 0), Quadray(0, 2, 1, 1), Quadray(1, 1, 2, 0), Quadray(0, 1, 0, 2)),
        # additional small integer examples
        (Quadray(0, 0, 0, 0), Quadray(3, 1, 1, 1), Quadray(1, 3, 1, 1), Quadray(1, 1, 3, 1)),
        (Quadray(0, 0, 0, 0), Quadray(5, 2, 1, 2), Quadray(2, 5, 2, 1), Quadray(1, 2, 5, 2)),
    ]

    rows = [("case", "V_ace_ivm_int", "V_cm_s3_ivm_sym", "match")]
    ace_vals = []
    cm_vals = []

    # Define descriptive case names
    case_names = [
        "unit_tetrahedron",
        "scaled_unit_tetrahedron", 
        "mixed_tetrahedron",
        "centered_tetrahedron",
        "large_mixed_tetrahedron"
    ]

    for idx, (p0, p1, p2, p3) in enumerate(examples):
        V_ace = ace_tetravolume_5x5(p0, p1, p2, p3)
        ace_vals.append(float(V_ace))
        # Build sympy coordinates and squared distances
        pts = []
        for p in (p0, p1, p2, p3):
            v = Matrix([p.a, p.b, p.c, p.d])
            xyz = M @ v
            pts.append(xyz)
        d2 = Matrix(4, 4, lambda i, j: 0)
        for i in range(4):
            for j in range(4):
                if i == j:
                    d2[i, j] = 0
                else:
                    diff = pts[i] - pts[j]
                    d2[i, j] = simplify(diff.dot(diff))
        V_xyz = cayley_menger_volume_symbolic(d2)
        V_ivm_sym = simplify(convert_xyz_volume_to_ivm_symbolic(V_xyz))
        cm_vals.append(float(N(V_ivm_sym)))
        match = bool(simplify(V_ivm_sym - V_ace) == 0)
        rows.append((case_names[idx], str(V_ace), str(V_ivm_sym), str(match)))

    data_dir = _get_data_dir()
    figure_dir = _get_figure_dir()
    csv_path = os.path.join(data_dir, "bridging_vs_native.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Plot comparison (improved styling)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(1, len(ace_vals) + 1))
    bar_width = 0.35
    color_ace = "#377eb8"  # blue
    color_cm = "#4daf4a"   # green

    # Define descriptive labels for each example
    example_labels = [
        "Unit Tetra\n(0,0,0,0)-(2,1,0,1)-(2,1,1,0)-(2,0,1,1)",
        "Scaled Unit\n(0,0,0,0)-(4,2,0,2)-(4,2,2,0)-(4,0,2,2)", 
        "Mixed Tetra\n(1,0,0,0)-(0,2,1,1)-(1,1,2,0)-(0,1,0,2)",
        "Centered\n(0,0,0,0)-(3,1,1,1)-(1,3,1,1)-(1,1,3,1)",
        "Large Mixed\n(0,0,0,0)-(5,2,1,2)-(2,5,2,1)-(1,2,5,2)"
    ]

    ax.bar([i - bar_width / 2 for i in x], ace_vals, width=bar_width, label="Ace 5×5 (IVM)", color=color_ace)
    ax.bar([i + bar_width / 2 for i in x], cm_vals, width=bar_width, label="CM + S3 (IVM)", color=color_cm)

    ax.set_xlabel("Tetrahedron Examples")
    ax.set_xticks(x)
    ax.set_xticklabels(example_labels, rotation=0, ha='center')
    ax.set_ylabel("Tetravolume (IVM units)")
    ax.set_title("Bridging (CM+S3) vs Native (Ace) IVM tetravolumes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.6)

    # Annotate bar heights for readability
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(f"{height:.3g}",
                        (bar.get_x() + bar.get_width() / 2.0, height),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # Footnote explaining equivalence and data artifact
    ax.text(0.01, -0.22,
            "Lengths→IVM via S3 (CM+S3) agree with native Ace 5×5 on canonical integer-quadray examples.\n"
            "CSV with exact values: quadmath/output/bridging_vs_native.csv",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)

    # Start y-axis at 0 and add a small headroom
    top = max(max(ace_vals, default=0.0), max(cm_vals, default=0.0))
    ax.set_ylim(0.0, top * 1.12 if top > 0 else 1.0)

    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.35)
    fig.savefig(os.path.join(figure_dir, "bridging_vs_native.png"), dpi=220)
    return csv_path


def magnitude_via_vector_module() -> str:
    """Compute magnitude using sympy.vector CoordSys3D for (x,y,z)."""
    from sympy.vector import CoordSys3D
    from sympy import symbols

    x, y, z = symbols("x y z", real=True)
    C = CoordSys3D("C")
    v = x * C.i + y * C.j + z * C.k
    return str(v.magnitude())


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _ensure_src_on_path()

    data_dir = _get_data_dir()
    V_xyz, V_ivm = cayley_menger_symbolic_unit_tetra()
    mag_expr = embedding_symbolic_magnitude()
    mag_vec_expr = magnitude_via_vector_module()

    with open(os.path.join(data_dir, "sympy_symbolics.txt"), "w") as f:
        f.write("V_xyz_unit_regular_tetra = " + V_xyz + "\n")
        f.write("V_ivm_unit_regular_tetra = " + V_ivm + "\n")
        f.write("magnitude_symbolic = " + mag_expr + "\n")
        f.write("magnitude_vector_module_symbolic = " + mag_vec_expr + "\n")

    print(os.path.join(data_dir, "sympy_symbolics.txt"))
    csv_path = compare_ace_vs_cm_examples()
    print(csv_path)


if __name__ == "__main__":
    main()


