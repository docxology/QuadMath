# Changelog

## 0.4.0 — 2026-11-06

### Mathematical methods

- Preserve common-offset gauge equivalence before floating embedding and exact centroid accumulation, including fixed-width NumPy inputs.
- Stabilize the inverse embedding solve and admit reconstructed integer candidates only when they replay the actual forward XYZ values. Component-wise half-up snapping remains distinct from closest-point lattice search.
- Promote integral scalars before Bareiss elimination, edge subtraction and determinant projection. Signed/unsigned NumPy integer overflow no longer corrupts exact determinants, rank or tetrahedron volumes.
- Compute distances, angles and Cayley–Menger pairwise geometry from integer displacements rather than subtracting large absolute floating positions. Materialize iterable embeddings before reuse.
- Use range-safe norms and normalized angle calculations at extreme finite scales, and retain finite scaled products when a large integer cannot first be converted to float.
- Normalize Cayley–Menger calculations and restore dimensional units without premature volume/radius overflow or underflow.
- Widen low-precision lattice-query inputs before gauge subtraction and use nonnegative Cartesian squared distances.
- Evaluate shell/cumulative counts exactly, widening array results when necessary. Remove arbitrarily large site gauges before bounded index lookup; enumeration order and `MAX_SHELL=32` are unchanged.

### Orchestration and artifacts

- Anchor both shell entrypoints to their checkout, regardless of the caller's working directory.
- Check the existing API glossary for freshness in the non-PDF coherence gate; stale documentation fails instead of being silently accepted.
- Accept emitted figure paths containing spaces while retaining repository containment, suffix filtering and deduplication.
- Preserve generation diagnostics, explicitly check combined-source reads/writes, and publish PDFs only after successful XeTeX passes. Failed builds cannot leave stale or partially written canonical PDFs looking current.
- Honor `SOURCE_DATE_EPOCH` for the visible UTC manuscript date. This is not a byte-reproducible PDF guarantee.

### Distribution and suite integration

- Build an installable `quadmath` wheel and source distribution using the existing suite's Hatchling convention; metadata names the shipped Apache-2.0 license.
- Exclude generated output and manuscript figure copies from source distributions.
- Support explicit `QUADMATH_OUTPUT_DIR` for installed plotting/data consumers without a checkout. Source tools retain `quadmath/output/` by default.
- Add Python 3.11/3.13 coverage/package gates and a Linux manuscript pipeline with the actual font, TeX, video and Pandoc prerequisites.
- Cross-reference symergetics and QuadCraft, document the orientation/unit conversion, and check independently enumerated FCC sites as well as actual Python/JavaScript/C++ coordinate APIs.

### Compatibility and verification

The four-component quotient remains three-dimensional. Exact volume documentation now correctly describes `Fraction(abs(det), 4)`, including the primitive `1/4` tetrahedron; no integer-volume rounding was introduced. Existing inverse, shell numbering and raw-versus-physical identity conventions remain explicit.

Run `UV_PYTHON=3.13 ./run_all.sh --test-only`, then `UV_PYTHON=3.13 ./run_all.sh --with-pdf` for the full local manuscript pipeline. The coordinated [suite verifier](https://github.com/docxology/symergetics/blob/main/verify_suite.py) also exercises the native and browser-language mathematical consumers. See the [suite guide](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md) for contracts and verification receipts.

Release verification: **988 tests passed with 100% statement/branch coverage**. The coordinated command executed all 13 stages; interoperability covered 279 coordinates per implementation, 34 exact volumes and 216 independently enumerated FCC sites.
