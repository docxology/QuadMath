# Conversions & Specification

## Overview

Sections [2](02_4d_namespaces.md), [3](03_quadray_methods.md) and [13](13_lattice_tooling.md)
establish the Quadray/IVM framework and its computational tooling; this section documents the
conversion layer that joins its two coordinate namespaces — `conversions.py`, the module that maps
between Fuller.4D quadray coordinates and Coxeter.4D Cartesian coordinates. Three concerns live
here: the embedding itself and its projective fiber, the Gram identities that make lattice
distances exact, and an **exact rational inversion** that answers "which lattice point is this
point?" with `fractions.Fraction` arithmetic instead of floating-point rounding.

The full mathematical specification of this layer is the repository-root file `SPEC.md`; the
`tests/test_spec_examples.py` pins its worked examples numerically. This section is the
manuscript projection of that specification: same definitions, same error taxonomy, same
guarantees, prose instead of source. The analytical foundations it draws on are
[Quadray Methods](03_quadray_methods.md) (normalization, the `Quadray` class), the shell machinery
of [IVM Field Learning](11_ivm_field_learning.md), and the nearest-site search of
[Lattice Tooling](13_lattice_tooling.md), whose exact-distance identity this section re-derives
from the embedding's Gram matrix.

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
shape-validated against the shared internal helper `_embedding_rows` (a `(3, 4)` array is required,
`ValueError` otherwise) before the product is taken. Existing callers that pass an explicit
embedding — for example `quadmath/scripts/quadray_clouds.py` — are unaffected.

## Gram identities and exact distances

The embedding matrix satisfies two exact Gram identities. Row-wise, at scale $s$:

\begin{equation}
\label{eq:conv-gram-row}
M\,M^{\mathsf{T}} \;=\; 4\,s^{2}\,I_{3},
\end{equation}

and column-wise, with $C_j$ the four columns of $M$ and $J_4$ the all-ones matrix:

\begin{equation}
\label{eq:conv-gram-col}
M^{\mathsf{T}}M \;=\; 4\,I_{4} \;-\; J_{4},
\qquad \text{i.e.} \qquad
C_i \cdot C_j \;=\; 4\,\delta_{ij} \;-\; 1 \;\; (s = 1).
\end{equation}

`embedding_basis(M=None)` returns the columns of the validated embedding as a `(4, 3)` array $B$
whose *rows* are the $C_j$; at scale 1 the identity \eqref{eq:conv-gram-col} reads
$B\,B^{\mathsf{T}} = 4\,I_4 - J_4$ exactly (diagonal 3, off-diagonal $-1$), and it scales by
$s^2$ for `urner_embedding(scale)`. This is the identity that `lattice_search.squared_distance`
relies on: for integer quadrays $p, q$ with $\delta = p - q$, the Euclidean squared distance in
embedded coordinates is the exact integer

\begin{equation}
\label{eq:conv-distance}
d^{2}(p, q) \;=\; \sum_{i} (M\delta)_i^{2}
\;=\; 4 \sum_{j} \delta_j^{2} \;-\; \Bigl(\sum_{j} \delta_j\Bigr)^{2},
\end{equation}

the same identity documented as Eq. \eqref{eq:lattice-distance-identity} in
[Lattice Tooling](13_lattice_tooling.md), where it powers ranking and filtering; the test suite
cross-validates it against float embeddings and `quadray.distance`. Because the identity is exact
on integers, nearest-site decisions never suffer floating-point boundary errors.

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
\eqref{eq:conv-canonical}; (4) `xyz` lies in the lattice image if and only if $x_1, x_2, x_3$ are
all integers — then every integer preimage is $y + t\,\mathbf{1}$, and the min-0 representative
`Quadray(x1, x2, x3, 0).normalize()` is the unique deterministic tie-break on the fiber
\eqref{eq:conv-fiber} (the same rule as `Quadray.normalize`); a non-integral preimage
raises `ValueError` (the point is not in the image).

The taxonomy is fail-closed. `ValueError` covers: `xyz` of length other than 3; `M` of the wrong
shape; any row sum different from zero; $\det = 0$ (rank below 3); a non-integral preimage.
`TypeError` covers entries that are not `int`/`float`/`Fraction`/`numbers.Integral` — note that
`numpy.float64` (a `float` subclass) and `numpy.int64` (integral) are accepted, while
`numpy.float32` is rejected. There is no floating-point fuzz anywhere: an off-lattice point raises
rather than snapping to a neighbor.

The float-valued inverse `quadray.quadray_from_xyz` is a different, deliberately fuzzier contract:
it applies the pseudoinverse $M^{+} = M^{\mathsf{T}}(MM^{\mathsf{T}})^{-1}$ and rounds half-up
(see its docstring argument for why round-half-up preserves the $(1,1,1,1)$-coset). For points on
the lattice it round-trips exactly; for general $\mathbb{R}^3$ points it returns the nearest lattice
point in quadray coordinates, which is a snapping operation, not an inverse. `quadray_roundtrip(q,
M=None)` pins the round-trip contract

\begin{equation}
\label{eq:conv-roundtrip}
\mathrm{from\_xyz}\bigl(\mathrm{to\_xyz}(q)\bigr) \;=\; q ,
\end{equation}

and raises `AssertionError` explicitly (not a bare `assert`, so the check survives
`python -O`) when it fails.
The contract is exact for normalized integer quadrays and, because
`quadray_roundtrip` threads the *same* embedding through both legs, it is **scale-robust**: for
$M = c\,M_0$ with any $c \neq 0$ the pseudoinverse projection is scale-independent,

\begin{equation}
\label{eq:conv-roundtrip-scale}
\mathrm{pinv}(cM)\,(cM\,q) \;=\; q \;-\; \frac{\sum_i q_i}{4}\,\mathbf{1},
\end{equation}

since $\mathrm{pinv}(cM) = (1/c)\,\mathrm{pinv}(M)$ and the row space of $cM_0$ is that of $M_0$
(the $(1/c)\cdot c$ cancels). A shell site embedded at scale $1/2$ — the half-integer FCC picture
of the same sites — round-trips through the same scaled embedding. The single documented
`AssertionError` case is an **unnormalized** input: the inverse canonicalizes to the min-0
representative, e.g. `(2, 2, 2, 1)` comes back as `(1, 1, 1, 0)`. Embeddings whose rows do not sum
to zero lie outside the Urner family of \eqref{eq:conv-embedding} and outside this guarantee.

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
octahedral and tetrahedral voids of the packing. The radial observable is the shell norm

\begin{equation}
\label{eq:conv-shell}
N(q) \;=\; \sum_{i} \Bigl|\, q_i \;-\; s/4 \,\Bigr| \;=\; 2k,
\qquad s \;=\; \sum_{i} q_i ,
\end{equation}

computed by `quadray_shell_norm`: shell $k$ is the set of sites with $N(q) = 2k$, populated by
$10k^2 + 2$ centers (Eq. \eqref{eq:lattice-shell-population} in [Lattice Tooling](13_lattice_tooling.md)),
and `ivm_field.shell_sites(k)` enumerates shell $k$ in lexicographic order (1, 12, 42, 92, 162
sites for $k = 0..4$). Conversions, shells, and nearest-site search share one geometry; Sections
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
| `lattice_search.squared_distance(p, sites)` | Exact squared distances via Eq. \eqref{eq:conv-distance} |

## Verification

`tests/test_conversions.py` covers the layer end to end: round-trips of
\eqref{eq:conv-roundtrip} over shells 0–4 (309 enumerated sites, center included) and at scaled
embeddings per \eqref{eq:conv-roundtrip-scale}, exact canonical
recovery of every site through \eqref{eq:conv-canonical}, the Gram identities
\eqref{eq:conv-gram-row} and \eqref{eq:conv-gram-col} with `embedding_basis`, the distance identity
\eqref{eq:conv-distance} cross-checked against float embeddings and `quadray.distance`, and the
fail-closed error taxonomy (`ValueError` for off-image points, bad shapes, zero determinants and
non-zero row sums; `TypeError` for inexact-foreign entry types such as `numpy.float32`). The
repository-root `SPEC.md` (the full mathematical specification) and `tests/test_spec_examples.py`
pin the worked examples of this section numerically as they land. Run the suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q
uv run coverage report
```
