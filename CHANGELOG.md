# Changelog

## 0.7.0 — 2026-11-09

### Statistical geometry, stable fitting and honest examples

- Draw confidence-interval endpoints independently of point estimates, including intervals that exclude an estimate and zero-width intervals; validate before allocating figures and close only owned figures after real drawing failures.
- Evaluate fitted scaling curves through a shared anchored, centered/scaled log-ratio fit with binary-exponent prediction rather than reconstructing a global intercept. Preserve adjacent represented sizes/runtimes, sorted fit geometry, constant timings and explicit failure for unrepresentable curves.
- Preserve tied empirical mass and a visible zero-to-one jump for singleton ECDFs. Compute representable histogram means without overflowing an intermediate sum.
- Replace arbitrary interval widths and misleading benchmark labels with computed seeded percentile-bootstrap intervals and explicitly synthetic inputs. Show known population means in the bootstrap/jackknife comparison.
- Modularize method sections and clarify nominal coverage, individual zero exclusion versus between-group tests, figure ownership and numerical versus encoded-file reproducibility. Replace copied inventories/byte assertions with consumer-visible geometry and resource regressions.

### Graphical evidence and numerical diagnostics

- Add importable `quadmath.viz.abstracts.render_quadray_abstract`, producing PNG/SVG with actual native basis endpoints, the rank-three gauge quotient and all twelve FCC neighbors in contact with a central sphere. Preserve caller figures and atomic per-format saves.
- Publish the scalable quadray/FCC abstract on the repository landing page with direct producer and reproduction links.
- Include SVG companions in isolated output manifests. Replace the regression demo's unrelated polynomial and framework tetrahedron with actual predictions; orient spectral tips along eigenvector columns.
- Distinguish sample-gradient Gram matrices, statistical Fisher information, metric descent and geodesic integration in figures, API docstrings and manuscript captions.
- Align regression state/loss/gradient arrays, report actual stopping reasons and expose raw model inputs. Restrict categorical landscapes to their valid simplex face, identify sampled minima and label ideal interior entropy curvature explicitly.
- Mark the simplex's actual final recorded best, include the final snapshot and distinguish simplex slots from persistent vertex identities.

### Modular documentation and verification

- Audit mathematical chapters, specification and operational guides against current source; maintain canonical API-glossary and scripted manuscript-projection ownership.
- Check fragment-only links, duplicate heading slugs, encoded/reference targets and linked Markdown outside scanned roots. Preserve long code fences. Missing/empty authored roots fail; multi-root repository scans prune dependency/generated trees without exempting linked targets.
- Add numerical data/domain/eigenvector regressions and real multi-root link-gate fixtures. Historical releases and deposited PDFs remain version-specific evidence, not claims about current main.

### Exact geometry, range-safe inference and field transport

- Subtract quadray points exactly before embedding floating distances, preserving small displacements between large native coordinates. Preserve mixed integer/rational matrix entries without homogeneous float coercion, and reject malformed embeddings consistently.
- Accept finite NumPy floating scalars, including `np.float32`, in strict inverse coordinates and matrices as their exact represented binary rationals; retain tolerance-free lattice membership, exact row sums and rank checks.
- Add public `sample_mean` and `scaling_predict(sizes, times, query_sizes)`; retain the `scaling_fit` three-tuple summary. Use compensated means with an overflow-only exact fallback, exact endpoint quantile interpolation and local centered/scaled moments.
- Stream bootstrap resamples through one seeded generator with linear sample-plus-iteration storage. Bootstrap and jackknife now default to `sample_mean`; require finite one-dimensional samples and finite real scalar callbacks. Reject unrepresentable jackknife bias/intervals.
- Use centered normalized pool-range permutation tie tolerance rather than absolute-offset tolerance. Stabilize Cohen's d and Welch inference across extreme representable offsets/units, and retain representable tiny Student-t tails without a normal approximation. Singleton descriptive standard deviation remains undefined; genuinely out-of-range descriptive spread can be infinite.
- Transport nonnegative Gaussian observation mixtures through the dense regularized learning system, preserving mixed-magnitude observations and cancellation-sensitive targets. Zero regularization evaluates the kernel target directly. Require observations in each graph component; reject nonfinite, nonconvex or physically backward-inaccurate fits atomically, without promising forward accuracy in ill-conditioned systems.
- Keep physical convex means portable on Python 3.11: compensate ordinary same-sign sums and accumulate mixed-sign or extreme products exactly. Preserve the zero-field optimum and require either an accurate ill-conditioned fit or an atomic numerical rejection.
- Retain conservative edge-flux transport at extreme finite magnitudes using exact accumulation when incident flux plus old state could overflow. Floating ordinary-path conservation remains subject to roundoff.
- Separate original wheel and source-distribution-rebuilt wheel proofs per supported Python version; distinguish installed-artifact receipts from editable-source checks and fixed-toolchain repeatability from universal byte reproducibility.

Verification for this release requires current [workflow runs](https://github.com/docxology/QuadMath/actions) and dated artifact receipts; this note does not report a new test, build, rendering, package-publication or CI result. Coordinated contracts and evidence boundaries live in the authoritative [suite guide](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md).

## 0.6.0 — 2026-11-07

### Exact methods and simulation

- Add exact rational dot products, squared norms/distances, signed tetrahedral volume, triangle squared area and affine combinations with signed weights. Preserve explicit rational forward embeddings and resolved native cancellation before rounding.
- Add hard-boundary harmonic extension with component anchoring, scale-safe solves and atomic failure. Keep regularized learning as a distinct soft-observation method.
- Add cached undirected graph edges, a graph Laplacian, Dirichlet energy and conservative transport preserving ordinary mass/constants in exact arithmetic. Keep symmetric-normalized heat's different stationary mode explicit.
- Preserve unbounded integer majority decisions, exact count objectives and fractional noisy observations; compute representable floating mean-square scores without premature squared overflow.

### Geometry, visualization and orchestration

- Replace malformed six-solid construction meshes with closed supporting-plane/indexed geometry; measure native volumes and edge lengths instead of copying labels. The displayed truncated octahedron has 24 vertices, 36 edges, 14 faces and native volume 96 at the declared scale.
- Render exact sparse quotient-plane cells without dense huge-span allocation or float-collapsed cell edges. Compare hard/soft boundaries and conservative/normalized dynamics with shared scales and explicit conservation diagnostics.
- Honor external `QUADMATH_OUTPUT_DIR` throughout all figure/data generators. Manifest entries are now relative to the configured output root, including the default root; readers must resolve against that root.
- Add manuscript `--figures-dir` for isolated generation/porting; correct rational native-volume, reachability, unit and centered-volume examples. Rename the symbolic CSV field to `V_ace_ivm`; it is not restricted to integers.
- Keep mathematical Unicode out of code listings: render the construction table's square roots and signed coordinates as mathematics. PDF failures now print actual TeX error context before the compiler tail.
- Check harmonic residuals with degree/scale-aware floating backward-error bounds, retaining exact boundary and independent linear-field recovery assertions across BLAS platforms.

### Formal scope

- Prove the integer-quadray physical quotient/canonical uniqueness, distance positivity and gauge congruence, determinant/volume bridges and universal shell lower bounds in dependency-free Lean.
- Add pinned `lake build`/`leanchecker` CI. Universal shell counts/BFS equivalence, rational-coordinate extensions, floating simulations and gameplay remain outside those proofs.

Local verification: the full mathematical suite passed **100% statement/branch coverage**; `lake build`, `leanchecker`, all 19 figure generators, strict manuscript/glossary/link checks and installed Python 3.11 wheel consumers were exercised successfully.

Cross-suite consumers and limits: [interoperability guide](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md).

## 0.5.0 — 2026-11-07

### Stable information geometry

- Normalize finite nonnegative probability weights without overflowing their sum, including joint arrays, integer totals and subnormal scales.
- Retain positive support in log space across the full finite-weight range, including finite KL, tiny entropy/mutual information and recoverable rare-state likelihood contributions.
- Define `eps` consistently as normalized-probability pseudocounts followed by renormalization. `eps=0` preserves exact support boundaries, including infinite KL for missing reference support.
- Preserve subnormal support in Jensen–Shannon mixtures and smooth joint distributions once before computing mutual information marginals.
- Widen integral Fisher scores before products. Ordinary batches retain the floating Gram fast path; overflow-risk normalized batches use exact accumulation so unrelated small curvature is not erased.
- Estimate Quadray Fisher information from transformed scores; safely symmetrize and analyze large/subnormal Fisher spectra and curvature.
- Keep zero-posterior likelihood terms out of expected surprise while retaining genuine infinite/undefined likelihood contributions on positive support.

### Exact geometry and suite integration

- Add exported `centroid_exact(points)`, returning a canonical four-`Fraction` tuple without lattice snapping, including arbitrary-size and rational inputs.
- Promote NumPy-backed `Fraction` fields before coordinate arithmetic, exact centroids and strict inverse elimination; fixed-width numerators/denominators cannot wrap these exact paths.
- Preserve existing half-up snapped centroid and componentwise inverse contracts.
- Compare exact continuous centroids with symergetics alongside the five-runtime coordinate, determinant and independently enumerated FCC checks.
- Update the canonical manuscript, generated API glossary and companion-project contracts for the coordinated symergetics 1.2.0 / QuadCraft v0.4.0 release.

Compatibility: public information-method signatures are unchanged, but explicit
non-negligible `eps` values now denote real pseudocount smoothing rather than
logarithm/overlap offsets. Irrational/floating calculations are not exactness
certificates; exceptional exact Fisher accumulation has a higher runtime cost.

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
- Require the patched pytest ≥9.0.3 floor consistently in both development dependency declarations ([GHSA-6w46-j5rx-g56g](https://github.com/advisories/GHSA-6w46-j5rx-g56g)); the locked test runner was already exercised at 9.2.0.
- Cross-reference symergetics and QuadCraft, document the orientation/unit conversion, and check independently enumerated FCC sites as well as actual Python/JavaScript/C++ coordinate APIs.

### Compatibility and verification

The four-component quotient remains three-dimensional. Exact volume documentation now correctly describes `Fraction(abs(det), 4)`, including the primitive `1/4` tetrahedron; no integer-volume rounding was introduced. Existing inverse, shell numbering and raw-versus-physical identity conventions remain explicit.

Run `UV_PYTHON=3.13 ./run_all.sh --test-only`, then `UV_PYTHON=3.13 ./run_all.sh --with-pdf` for the full local manuscript pipeline. The coordinated [suite verifier](https://github.com/docxology/symergetics/blob/main/verify_suite.py) also exercises the native and browser-language mathematical consumers. See the [suite guide](https://github.com/docxology/symergetics/blob/main/docs/synergetics-suite.md) for contracts and verification receipts.

Release verification: **988 tests passed with 100% statement/branch coverage**. The coordinated command executed all 13 stages; interoperability covered 279 coordinates per implementation, 34 exact volumes and 216 independently enumerated FCC sites.
