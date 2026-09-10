from __future__ import annotations

import os

import numpy as np


def test_sympy_formalisms_manifest(tmp_path, monkeypatch) -> None:
    from sympy_formalisms import (
        cayley_menger_symbolic_unit_tetra,
        embedding_symbolic_magnitude,
        magnitude_via_vector_module,
    )

    V_xyz, V_ivm = cayley_menger_symbolic_unit_tetra()
    # Expect canonical values for unit-edge regular tetrahedron
    assert "sqrt(2)/12" in V_xyz or V_xyz == str(np.sqrt(2) / 12)
    # IVM conversion via S3 factor: for unit XYZ edge, V_ivm simplifies to 1/8
    assert V_ivm == "1/8"

    mag_expr = embedding_symbolic_magnitude()
    assert "sqrt(" in mag_expr and "a" in mag_expr and "b" in mag_expr
    mag_vec_expr = magnitude_via_vector_module()
    assert mag_vec_expr.startswith("sqrt(") and "x" in mag_vec_expr

    # Redirect the script's output tree to tmp_path so the shared
    # quadmath/output/ tree that the manuscript references is never touched.
    monkeypatch.setattr("sympy_formalisms._get_output_dir", lambda: str(tmp_path))
    from sympy_formalisms import main as run

    run()

    # Check that files were written to the redirected data directory
    data_dir = os.path.join(str(tmp_path), "data")
    path = os.path.join(data_dir, "sympy_symbolics.txt")
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "V_xyz_unit_regular_tetra" in content
    assert "V_ivm_unit_regular_tetra" in content


def test_ace_and_cm_s3_agree(tmp_path, monkeypatch) -> None:
    # Contract: the Ace 5x5 tetravolume must equal the Cayley-Menger + S3
    # conversion on every shared example. The committed
    # bridging_vs_native.csv previously showed match=False on all rows (S3
    # applied under the wrong embedding scale); this test locks the fixed
    # convention in place.
    import csv

    from sympy_formalisms import compare_ace_vs_cm_examples

    monkeypatch.setattr("sympy_formalisms._get_output_dir", lambda: str(tmp_path))
    csv_path = compare_ace_vs_cm_examples()
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["case", "V_ace_ivm_int", "V_cm_s3_ivm_sym", "match"]
    assert len(rows) == 6
    for case, ace, cm, match in rows[1:]:
        assert match == "True", f"{case}: ace={ace} cm_s3={cm}"
        assert ace == cm
