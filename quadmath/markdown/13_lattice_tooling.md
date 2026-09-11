# Lattice Tooling: Omnidirectional Numbering and Nearest-Site Search

## Overview

Sections [2](02_4d_namespaces.md) and [3](03_quadray_methods.md) develop the Quadray/IVM framework analytically; this section documents the computational tooling that operates directly on the close-packed lattice. Two modules extend the analytical core without touching it:

- `omni_numbering` — omnidirectional close-packing numbering: vectorized enumeration of the IVM shell sequence, cumulative counts, and bidirectional site/index mappings.
- `lattice_search` — fast nearest-lattice-point queries: `nearest(site, R, k)` and `within_radius(site, R)` built on a precomputed ball index, entirely in NumPy (no scipy).

Both modules operate on canonical quadray integer 4-tuples (non-negative components, at least one zero — the projective normalization of [Section 3](03_quadray_methods.md)), and both rely on the same verified shell invariants.

## Omnidirectional Close Packing and Frequency Shells

In the isotropic vector matrix (IVM), equal spheres pack omnidirectionally — closest packing. Starting from a central sphere, the packing builds up in consecutive **frequency shells**: the shell of frequency $k \in \mathbb{Z}_{\ge 0}$ sits $k$ radius-length increments outward of the center, the same four-dimensional buildup language synergetics uses for the growing vector equilibrium. Writing $N_k$ for the population of shell $k$, the lone center is the whole of shell 0 ($N_0 = 1$), and every shell with $k \ge 1$ carries exactly

\begin{equation}\label{eq:lattice-shell-population}
N_k \;=\; 10\,k^{2} \;+\; 2
\end{equation}

sphere centers. The first shell — the 12 permutations of $(2, 1, 1, 0)$ — is the cuboctahedron (vector equilibrium) of the "twelve around one" motif; each subsequent shell is the next layer of the omnidirectional buildup. The cumulative count through frequency $k$, written $C_k$, is the centered-cuboctahedral closed form

\begin{equation}\label{eq:lattice-cumulative}
C_k \;=\; 1 \;+\; \sum_{j=1}^{k} \left(10\,j^{2} + 2\right) \;=\; 1 + 2k + \frac{10\,k\,(k+1)\,(2k+1)}{6}.
\end{equation}

The functions `shell_count(k)` and `cumulative_count(k)` evaluate Eqs. \eqref{eq:lattice-shell-population} and \eqref{eq:lattice-cumulative} (scalar or array input), and the test suite cross-validates them against direct summation.

### Reachability is enumerative, not congruence-based

A normalized integer quadray $q$ with $\sum_i q_i \equiv 0 \pmod 4$ is a candidate IVM point, but not every candidate is reachable from the origin by the 12 neighbor moves. The module therefore builds the site set **layer by layer**: each new shell is produced at once by adding all 12 moves to the previous frontier, re-normalizing rows, and removing sites already seen (packed-key membership tests). The result is cached per depth: `sites_through_shell(k)` returns the `(C_k, 4)` integer array in canonical order — the center first, then each shell in lexicographic `(a, b, c, d)` order — and `generate_shell(k)` returns one shell. An independent breadth-first reference over `itertools.permutations` confirms exact agreement through frequency 8, including the counts of Eq. \eqref{eq:lattice-shell-population} on every shell.

### Site/index mapping

`site_index(site, max_shell)` returns the position of a site in that canonical enumeration (or `-1` if absent within the depth), and `site_at_index(index, max_shell)` is its inverse. Lookups accept any projective representative — the input is translated by `-(k, k, k, k)` before matching — so `(2, 1, 1, 0)` and `(3, 2, 2, 1)` map to the same index, while `(0, 1, 1, 2)`, a different permutation and hence a different sphere, maps elsewhere. Component overflow is guarded: inputs whose normalized components reach the 16-bit packed-key width ($2^{16}$, the per-component field of the int64 keys) return `-1` rather than wrapping.

Executable check of the bookends:

```python
from quadmath.lattice.omni_numbering import (
    shell_count, cumulative_count, sites_through_shell,
    site_index, site_at_index,
)

assert shell_count(1) == 12 and shell_count(2) == 42      # 10k^2 + 2
assert cumulative_count(2) == 55                          # 1 + 12 + 42
assert site_index(site_at_index(12, 3), 3) == 12          # roundtrip
assert site_index((2, 0, 0, 0), 6) == -1                  # unreachable site
```

## Fast Nearest-Site Queries

### Exact lattice distances

Under `quadray.DEFAULT_EMBEDDING` the four embedding columns $C_j$ satisfy $C_i \cdot C_j = 4\,\delta_{ij} - 1$ at unit scale, so the Euclidean squared distance between integer 4-vectors $p, q$ is the exact integer

\begin{equation}\label{eq:lattice-distance-identity}
d^{2}(p, q) \;=\; 4 \sum_{j} \delta_j^{2} \;-\; \Bigl(\sum_{j} \delta_j\Bigr)^{2},
\qquad \delta = p - q,
\end{equation}

implemented in `squared_distance` and cross-validated in the tests against direct embedding coordinates. Because the identity is exact on integers, ranking and filtering never suffer floating-point boundary errors: a site is inside the query radius $R > 0$ iff $d^{2} \le R^{2}$ evaluated exactly (in float64 the same algebra is exact for the magnitudes involved here).

### Truncation bound

Every site $s$ on frequency shell $g$ satisfies $d^{2}(\mathbf{0}, s) \ge 8\,g$ — verified by exhaustive enumeration through frequency 8 in the test suite — and the shell maximum is exactly $8\,g^{2}$. Combining this invariant with the triangle inequality gives the depth at which a query can stop:

\begin{equation}\label{eq:lattice-truncation-bound}
g \;\leq\; \frac{\left(\lVert c \rVert + R\right)^{2}}{8},
\end{equation}

for a query center $c$ and radius $R > 0$, where $\lVert c \rVert$ is the Euclidean norm of the embedded center: any site within $R$ of $c$ must live on a shell at most this deep. `within_radius(site, R)` therefore enumerates exactly through $\lfloor (\lVert c \rVert + R)^{2} / 8 \rfloor + 1$ shells, filters with Eq. \eqref{eq:lattice-distance-identity}, and returns all hits sorted by squared distance with lexicographic `(a, b, c, d)` tie-breaking. Requests whose required depth exceeds the precomputed index depth (`MAX_SHELL = 32`, about $1.1 \times 10^{5}$ sites — $C_{32} = 114465$ by Eq. \eqref{eq:lattice-cumulative}) raise `ValueError` rather than silently truncating.

### Shell-sweep `nearest`

`nearest(site, R, k)` sweeps shells outward, ranking each shell's squared distances with `numpy.argsort`; once `k` candidates are in hand, the invariant behind Eq. \eqref{eq:lattice-truncation-bound} supplies an early-stop test — no unvisited shell can contain a closer site: a site at depth $g + 1$ satisfies $d^{2}(\mathbf{0}, s) \ge 8\,(g+1)$, so the sweep stops once $8\,(g+1) > (\sqrt{d^{2}_{(k)}} + \lVert c \rVert)^{2}$, with $d^{2}_{(k)}$ the current k-th best squared distance. The sweep never under-enumerates (the bound is conservative; ties at the boundary are kept), and answers always match the brute-force reference over the same region, as the tests assert on fixed-seed random queries.

```python
import numpy as np
from quadmath.lattice.lattice_search import nearest, within_radius

sites, d2 = nearest((0.4, -0.3, 0.9, 0.1), R=2.0, k=5)   # 5 nearest centers
ball, ball_d2 = within_radius((2, 1, 1, 0), R=4.0)       # everything within 4
assert np.all(np.diff(ball_d2) >= 0)                      # ascending distances
```

## API Summary

| Function | Purpose |
| --- | --- |
| `omni_numbering.shell_count(k)` | Shell population, Eq. \eqref{eq:lattice-shell-population} |
| `omni_numbering.cumulative_count(k)` | Cumulative count through shell `k`, Eq. \eqref{eq:lattice-cumulative} |
| `omni_numbering.generate_shell(k)` | Sites of one frequency shell (lexicographic) |
| `omni_numbering.sites_through_shell(k)` | Whole enumeration `(C_k, 4)` in canonical order |
| `omni_numbering.site_index` / `site_at_index` | Bidirectional site/index mapping with projective normalization |
| `lattice_search.squared_distance` | Exact squared distances via Eq. \eqref{eq:lattice-distance-identity} |
| `lattice_search.within_radius(site, R)` | All sites within radius, exact and sorted |
| `lattice_search.nearest(site, R, k)` | `k` nearest sites within radius, shell-sweep with early stop |

## Verification

Both modules ship with 100% test coverage (branch coverage included): the enumeration is checked against an independent breadth-first reference through frequency 8, distance identities against direct embedding computations, and query results against brute-force filtering on fixed-seed random centers. Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run coverage run -m pytest -q
uv run coverage report
```
