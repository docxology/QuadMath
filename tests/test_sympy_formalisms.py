from __future__ import annotations

import os

import numpy as np


def test_sympy_formalisms_manifest(tmp_path) -> None:
    # Run the script in-process to generate outputs
    import sys
    import os
    
    # Add the scripts directory to path to import functions
    script_dir = os.path.join(os.path.dirname(__file__), "..", "quadmath", "scripts")
    sys.path.insert(0, script_dir)
    
    from sympy_formalisms import (
        cayley_menger_symbolic_unit_tetra,
        embedding_symbolic_magnitude,
        _get_output_dir,
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

    # Write file via main
    from sympy_formalisms import main as run

    run()
    
    # Check that files were written to data directory
    outdir = _get_output_dir()
    data_dir = os.path.join(outdir, "data")
    path = os.path.join(data_dir, "sympy_symbolics.txt")
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "V_xyz_unit_regular_tetra" in content
    assert "V_ivm_unit_regular_tetra" in content
    assert "magnitude_symbolic" in content


