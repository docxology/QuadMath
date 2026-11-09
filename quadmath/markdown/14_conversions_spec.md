# Embedding, Exact Inversion, and Floating Snapping Contracts {#conversions-specification}

## The conversion layer and its mathematical authority

Sections [2](02_4d_namespaces.md), [3](03_quadray_methods.md) and [13](13_lattice_tooling.md)
establish the Quadray/IVM framework and its computational tooling; this section documents the
conversion layer that joins its two coordinate namespaces — `conversions.py`, the module that maps
between Fuller.4D quadray coordinates and Coxeter.4D Cartesian coordinates. Three concerns live
here: the embedding itself and its projective fiber, the Gram identities that make lattice
distances exact, and an **exact rational inversion** that answers "which lattice point is this
point?" with `fractions.Fraction` arithmetic instead of floating-point rounding.

The full mathematical authority is [SPEC.md](../../SPEC.md), whose worked
examples are transcribed in `tests/unit/lattice/test_spec_examples.py`.
This chapter explains the same representation, numerical boundaries, and
error taxonomy. Its dependencies are [coordinate methods](03_quadray_methods.md),
[FCC membership and graph shells](11_ivm_field_learning.md), and
[float64 nearest-site search](13_lattice_tooling.md). Exact identities
must not be confused with arbitrary-precision query implementations.

## The embedding (Fuller.4D to Coxeter.4D)

The default embedding maps the four quadray axes $(A, B, C, D)$ to the vertices of a regular
tetrahedron in $\mathbb{R}^3$. With scale factor $s$ (uniform; scales all resulting coordinates):

\begin{equation}
\label{eq:conv-embedding}
M \;=\; s \times
\begin{pmatrix}
 1 & -1 & -1 &  1 \\
 1 &  1 & -1 & -1 \\
 1 & -1 &  1 & -1
\end{pmatrix},
\qquad \mathrm{xyz}(q) \;=\; M\,q .
\end{equation}

`urner_embedding(scale=1.0)` builds exactly this matrix (each row scaled by `scale`), and
`quadray.DEFAULT_EMBEDDING` is the same unscaled matrix stored as a tuple-of-rows; the two agree
entry for entry at scale 1. Every row of $M$ sums to zero:

\begin{equation}
\label{eq:conv-fiber}
\textstyle\sum_{j} M_{ij} \;=\; 0 \quad \forall\, i,
\qquad \text{hence} \qquad
q \;\sim\; q + t\,\mathbf{1}, \qquad t \in \mathbb{Z},
\end{equation}

where $\mathbf{1} = (1, 1, 1, 1)$. The vector $\mathbf{1}$ spans the kernel of $M$, so the map is
projective: `to_xyz(q) == to_xyz(q + t*(1, 1, 1, 1))` for every integer $t$, and quadray
normalization (see [Quadray Methods](03_quadray_methods.md)) simply selects the min-0 representative
of that equivalence class. The forward map `quadray_to_xyz(q, M=None)` delegates to
`quadray.to_xyz`; passing `M=None` selects `quadray.DEFAULT_EMBEDDING`, and an explicit `M` is
shape-validated against the shared internal helper `_embedding_rows` (three rows
of four entries are required, `ValueError` otherwise) before the product is taken.
Mixed integer, rational and floating entries retain their individual values;
validation does not first coerce the whole matrix to a homogeneous floating dtype.

## Gram identities and exact distances

The embedding matrix satisfies two exact Gram identities. Row-wise, at scale $s$:

\begin{equation}
\label{eq:conv-gram-row}
M\,M^{\mathsf{T}} \;=\; 4\,s^{2}\,I_{3},
\end{equation}

and column-wise, with $C_j$ the four columns of $M$ and $J_4$ the all-ones matrix:

\begin{equation}
\label{eq:conv-gram-col}
M^{\mathsf{T}}M \;=\; s^2(4\,I_{4} - J_{4}),
\qquad \text{i.e.} \qquad
C_i \cdot C_j \;=\; s^2(4\,\delta_{ij} - 1).
\end{equation}

`embedding_basis(M=None)` returns the columns of the validated embedding as a `(4, 3)` array $B$
whose *rows* are the $C_j$; at scale 1 the identity \eqref{eq:conv-gram-col} reads
$B\,B^{\mathsf{T}} = 4\,I_4 - J_4$ exactly (diagonal 3, off-diagonal $-1$), and it scales by
$s^2$ for `urner_embedding(scale)`. At **scale one**, for integer quadrays
$p,q$ with $\delta=p-q$, the exact squared-distance identity is

\begin{equation}
\label{eq:conv-distance}
d^{2}(p, q) \;=\; \sum_{i} (M\delta)_i^{2}
\;=\; 4 \sum_{j} \delta_j^{2} \;-\; \Bigl(\sum_{j} \delta_j\Bigr)^{2},
\end{equation}

the same identity documented as Eq. \eqref{eq:lattice-distance-identity} in
[Lattice Tooling](13_lattice_tooling.md), where it powers ranking and filtering.
The implementation removes common gauge offsets before float conversion where
possible and evaluates nonnegative Cartesian squares, avoiding cancellation
between large quadratic terms. Exact algebra does not imply arbitrary float64
queries or large squared distances are represented exactly.

Executable check of the two identities:

```python
import numpy as np
from quadmath.lattice.conversions import urner_embedding, embedding_basis
from quadmath.lattice.lattice_search import squared_distance

M = urner_embedding()
B = embedding_basis()                              # (4, 3): rows are the columns of M
assert np.allclose(M @ M.T, 4.0 * np.eye(3))       # Eq. (eq:conv-gram-row)
assert np.allclose(B @ B.T, 4.0 * np.eye(4) - np.ones((4, 4)))
delta = np.array([2, 1, 1, 0])                     # shell-1 displacement
assert squared_distance(delta, np.zeros((1, 4), dtype=np.int64)) == 8   # Eq. (eq:conv-distance)
```

## Exact canonical inversion

The forward map is projective but not injective: by \eqref{eq:conv-fiber} the fiber of an embedded
point is the full coset $\{\,q + t\,\mathbf{1} : t \in \mathbb{Z}\,\}$, and an inverse must select
one representative from it.

\begin{equation}
\label{eq:conv-canonical}
y \;=\; (x_1, x_2, x_3, 0), \qquad
\begin{pmatrix} C_0 & C_1 & C_2 \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} \;=\; \mathrm{xyz},
\qquad q^\ast \;=\; \mathrm{normalize}(y).
\end{equation}

`xyz_to_quadray_canonical(xyz, M=None)` resolves that ambiguity deterministically by exact rational
arithmetic (`fractions.Fraction`; a Python float enters as its exact binary rational — no rounding,
no tolerance). The algorithm, in prose: (1) convert `xyz` and `M` to exact `Fraction` entries and
require every row of `M` to sum to exactly zero, so that normalization preserves the image point
\eqref{eq:conv-fiber}; (2) because $C_3 = -(C_0 + C_1 + C_2)$, rank $M = 3$ is equivalent to
$\det\,[\,C_0\; C_1\; C_2\,] \neq 0$ (a `ValueError` otherwise); (3) solve the $3 \times 3$ system
on columns 0–2 by Cramer's rule for the particular preimage $y = (x_1, x_2, x_3, 0)$ of
\eqref{eq:conv-canonical}; (4) the embedded point lies in the image of the integer lattice
$\{\,M\,q : q \in \mathbb{Z}^{4}\,\}$ if and only if $x_1, x_2, x_3$ are
all integers — then every integer preimage is $y + t\,\mathbf{1}$, and the min-0 representative
`Quadray(x1, x2, x3, 0).normalize()` is the unique deterministic tie-break on the fiber
\eqref{eq:conv-fiber} (the same rule as `Quadray.normalize`); a non-integral preimage
raises `ValueError` (the point is not in the image).

The taxonomy is fail-closed. `ValueError` covers: `xyz` of length other than 3; `M` of the wrong
shape; any row sum different from zero; $\det = 0$ (rank below 3); a non-integral preimage.
`TypeError` covers entries outside `int`/`float`/`Fraction`/`numbers.Integral`/
`numpy.floating`. Finite NumPy floating scalars, including `numpy.float32`,
enter as their exact represented binary rationals, not rounded decimal guesses.
Nonfinite floating entries raise `ValueError` or `OverflowError`.
There is no floating-point fuzz: an off-lattice point raises rather than snapping.

The float-valued inverse `quadray.quadray_from_xyz` is a distinct snapping
contract: it retains component-wise half-up rounding of the minimum-norm real
preimage, with coherent recovery of numerically resolvable lattice images.
Independent rounding of roundoff-perturbed half-integer ties can leave the
correct common-offset class, even at ordinary scales; the numerical algorithm
must account for that shared phase. Off-lattice snapping is **not necessarily
nearest in embedded XYZ distance**. It remains distinct from strict rational
membership and from a spatial nearest-point search. `quadray_roundtrip(q,
M=None)` checks

\begin{equation}
\label{eq:conv-roundtrip}
\mathrm{from\_xyz}\bigl(\mathrm{to\_xyz}(q)\bigr) \;=\; q ,
\end{equation}

and raises `AssertionError` explicitly (not a bare `assert`, so the check survives
`python -O`) when it fails.
In exact arithmetic this is scale-independent for normalized integer quadrays
when both legs use the same $M=cM_0$, $c\neq0$:

\begin{equation}
\label{eq:conv-roundtrip-scale}
\mathrm{pinv}(cM)\,(cM\,q) \;=\; q \;-\; \frac{\sum_i q_i}{4}\,\mathbf{1},
\end{equation}

since $\mathrm{pinv}(cM)=(1/c)\,\mathrm{pinv}(M)$. Numerical regression cases
also cover non-dyadic scales such as $0.1$ and $1/3$, both signs and all
component-sum residue classes. This does not promise reconstruction after XYZ
overflow, unresolved ill-conditioning or loss of integer information in a
float. An unnormalized input intentionally fails the identity because inversion
canonicalizes it: `(2,2,2,1)` returns `(1,1,1,0)`. Nonzero row sums or deficient
rank lie outside the quadray gauge contract.

Coherent integer reconstruction is accepted by replaying the actual forward
floating-point map and matching the supplied XYZ floats, not by granting a
residual allowance proportional to unrelated large components. Strict rational
membership remains separate and tolerance-free.

Executable check of exact recovery and round-trips:

```python
from fractions import Fraction
from quadmath.lattice.conversions import (
    quadray_to_xyz, xyz_to_quadray_canonical, quadray_roundtrip, urner_embedding,
)
from quadmath.core.quadray import Quadray
from quadmath.lattice.omni_numbering import sites_through_shell

q0 = Quadray(2, 1, 1, 0)                          # cuboctahedron vertex, shell 1
assert xyz_to_quadray_canonical(quadray_to_xyz(q0)) == q0
assert xyz_to_quadray_canonical(
    (Fraction(-1), Fraction(1), Fraction(1))
) == Quadray(1, 1, 1, 0)                          # exact entries, no float fuzz
for site in sites_through_shell(2)[:8]:           # round-trips over enumerated sites
    assert quadray_roundtrip(Quadray(*site)) == Quadray(*site)
assert quadray_roundtrip(q0, urner_embedding(0.5)) == q0   # scale-robust (same rows, both legs)
```

## Lattice context

The canonical representatives produced by \eqref{eq:conv-canonical} sit in the IVM site structure
of [IVM Field Learning](11_ivm_field_learning.md). A normalized quadray whose component sum is
$\equiv 0 \pmod 4$ is an IVM sphere center (`is_ivm_site`); the other three residue classes are the
octahedral and tetrahedral voids of the packing. The graph-shell observable is

\begin{equation}
\label{eq:conv-shell}
N(q) \;=\; \sum_{i} \Bigl|\, q_i \;-\; \sigma/4 \,\Bigr| \;=\; 2k,
\qquad \sigma \;=\; \sum_{i} q_i ,
\end{equation}

computed by `quadray_shell_norm`: shell $k$ is the set of sites with $N(q) = 2k$, populated by
$10k^2+2$ centers for $k\ge1$ (one center at $k=0$).
`ivm_field.shell_sites(k)` enumerates each graph shell lexicographically.
Graph shells are not surfaces of constant Euclidean radius. Sections
[11](11_ivm_field_learning.md) and [13](13_lattice_tooling.md) detail the field learner and the
query machinery respectively.

## API Summary

| Function | Purpose |
| --- | --- |
| `conversions.urner_embedding(scale=1.0)` | The $(3, 4)$ embedding matrix of Eq. \eqref{eq:conv-embedding} |
| `conversions.quadray_to_xyz(q, M=None)` | Forward map via `quadray.to_xyz`; `M=None` = `quadray.DEFAULT_EMBEDDING` |
| `conversions.xyz_to_quadray_canonical(xyz, M=None)` | Exact rational canonical inverse, Eqs. \eqref{eq:conv-fiber} and \eqref{eq:conv-canonical} |
| `conversions.quadray_roundtrip(q, M=None)` | Asserts the round-trip identity \eqref{eq:conv-roundtrip}, scale-robust per \eqref{eq:conv-roundtrip-scale} |
| `conversions.embedding_basis(M=None)` | `(4, 3)` array of embedding columns; Gram identity \eqref{eq:conv-gram-col} |
| `quadray.to_xyz(q, embedding)` | Underlying forward product $\mathrm{xyz} = M\,q$ |
| `quadray.quadray_from_xyz(x, y, z, embedding)` | Float inverse: pseudoinverse + round-half-up (snapping, not exact inversion) |
| `lattice_search.squared_distance(p, sites)` | Float64 gauge-free squared distances corresponding to Eq. \eqref{eq:conv-distance}, within numerical limits |

## Conversion regressions and worked-example scope

`tests/unit/lattice/test_conversions.py` covers the layer end to end: round-trips of
\eqref{eq:conv-roundtrip} over shells 0–4 (309 enumerated sites, center included) and at scaled
embeddings per \eqref{eq:conv-roundtrip-scale}, exact canonical
recovery of every site through \eqref{eq:conv-canonical}, the Gram identities
\eqref{eq:conv-gram-row} and \eqref{eq:conv-gram-col} with `embedding_basis`, the distance identity
\eqref{eq:conv-distance} cross-checked against float embeddings and `quadray.distance`, and the
fail-closed error taxonomy (`ValueError` for off-image points, bad shapes, zero determinants and
non-zero row sums; `TypeError` for unsupported entry types such as strings;
`ValueError` or `OverflowError` for nonfinite floating entries). Finite NumPy
floating entries, including `numpy.float32`, retain represented-value semantics. The
repository-root `SPEC.md` (the full mathematical specification) and `tests/unit/lattice/test_spec_examples.py`
pin the worked examples of this section numerically as they land. Run the suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 coverage run -m pytest -q
uv run --python 3.13 coverage report
```
