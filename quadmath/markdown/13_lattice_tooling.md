# Lattice Tooling: Omnidirectional Numbering and Nearest-Site Search

## Shell indexing and bounded spatial queries

Sections [2](02_4d_namespaces.md) and [3](03_quadray_methods.md) develop the Quadray/IVM framework analytically; this section documents the computational tooling that operates directly on the close-packed lattice. Two modules extend the analytical core without touching it:

- `omni_numbering` — omnidirectional close-packing numbering: vectorized enumeration of the IVM shell sequence, cumulative counts, and bidirectional site/index mappings.
- `lattice_search` — fast nearest-lattice-point queries: `nearest(site, R, k)` and `within_radius(site, R)` built on a precomputed ball index, entirely in NumPy (no scipy).

Both modules operate on canonical quadray integer 4-tuples (non-negative components, at least one zero — the projective normalization of [Section 3](03_quadray_methods.md)), and both rely on the same verified shell invariants.

## Omnidirectional Close Packing and Frequency Shells

The IVM builds consecutive **graph-distance frequency shells**, not constant-radius Euclidean spheres. For example, shell 2 contains scale-one squared radii 16 and 32. Writing $N_k$ for the population of shell $k$, the lone center is the whole of shell 0 ($N_0 = 1$), and every shell with $k \ge 1$ carries exactly

\begin{equation}\label{eq:lattice-shell-population}
N_k \;=\; 10\,k^{2} \;+\; 2
\end{equation}

sphere centers. The first shell — the 12 permutations of $(2, 1, 1, 0)$ — is the cuboctahedron (vector equilibrium) of the "twelve around one" motif; each subsequent shell is the next layer of the omnidirectional buildup. The cumulative count through frequency $k$, written $C_k$, is the centered-cuboctahedral closed form

\begin{equation}\label{eq:lattice-cumulative}
C_k \;=\; 1 \;+\; \sum_{j=1}^{k} \left(10\,j^{2} + 2\right) \;=\; 1 + 2k + \frac{10\,k\,(k+1)\,(2k+1)}{6}.
\end{equation}

The functions `shell_count(k)` and `cumulative_count(k)` evaluate Eqs. \eqref{eq:lattice-shell-population} and \eqref{eq:lattice-cumulative} (scalar or array input), and the test suite cross-validates them against direct summation.

### Membership and constructive reachability

A canonical integer quadray is an IVM site **iff**
$\sum_i q_i\equiv0\pmod4$. Its centered representative
$w=q-(\sum_iq_i/4)(1,1,1,1)$ is an integer sum-zero vector. The twelve
centered neighbor moves are precisely $e_i-e_j$ for $i\ne j$. Balancing
positive and negative entries constructs every such $w$ in
$\sum_i|w_i|/2$ moves; no shorter route can change that many unit transfers.
Thus membership is not an additional enumeration-dependent restriction.

The implementation expands frontiers, normalizes, removes previously seen
sites and caches deterministic shell-major numbering. `sites_through_shell(k)`
returns the `(C_k,4)` array; `generate_shell(k)` returns one lexicographic
shell. The independent BFS comparisons exercise this correspondence through
frequency 8. The constructive argument here is not a claim that its full
BFS equivalence or universal shell count has been formalized in Lean.

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
assert site_index((2, 0, 0, 0), 6) == -1                  # octahedral void, not FCC
```

## Fast Nearest-Site Queries

### Exact lattice distances

Under `quadray.DEFAULT_EMBEDDING` the four embedding columns $C_j$ satisfy $C_i \cdot C_j = 4\,\delta_{ij} - 1$ at unit scale, so the Euclidean squared distance between integer 4-vectors $p, q$ is the exact integer

\begin{equation}\label{eq:lattice-distance-identity}
d^{2}(p, q) \;=\; 4 \sum_{j} \delta_j^{2} \;-\; \Bigl(\sum_{j} \delta_j\Bigr)^{2},
\qquad \delta = p - q,
\end{equation}

The identity is exact on integers; `lattice_search.squared_distance` evaluates gauge-free Cartesian squares in float64. Ranking and radius filtering therefore retain the documented floating limits, not arbitrary-integer exactness. Use `quadmath.squared_distance_exact` when an exact individual rational distance is required; floating input cannot restore information lost before the call.

### Truncation bound

Every shell-$g$ site satisfies $d^2(\mathbf0,s)\ge8g$, now proved universally for the centered enumeration in Lean as `shellSites_dist2_lower`. Finite BFS comparisons check its graph-shell interpretation; they do not replace a universal BFS proof. The shell maximum is $8g^2$. The radial lower bound and triangle inequality give the depth at which a query can stop:

\begin{equation}\label{eq:lattice-truncation-bound}
g \;\leq\; \frac{\left(\lVert c \rVert + R\right)^{2}}{8},
\end{equation}

for a query center $c$ and radius $R > 0$, where $\lVert c \rVert$ is the Euclidean norm of the embedded center: any site within $R$ of $c$ must live on a shell at most this deep. `within_radius(site, R)` therefore enumerates exactly through $\lfloor (\lVert c \rVert + R)^{2} / 8 \rfloor + 1$ shells, filters with Eq. \eqref{eq:lattice-distance-identity}, and returns all hits sorted by squared distance with lexicographic `(a, b, c, d)` tie-breaking. Requests whose required depth exceeds the precomputed index depth (`MAX_SHELL = 32`, about $1.1 \times 10^{5}$ sites — $C_{32} = 114465$ by Eq. \eqref{eq:lattice-cumulative}) raise `ValueError` rather than silently truncating.

### Shell-sweep `nearest`

`nearest(site, R, k)` ranks FCC candidates within the Euclidean cutoff `R`, sweeping graph shells outward. It returns at most `k` hits, possibly fewer or none. After obtaining `k` candidates it can stop if the lower bound for every unvisited shell exceeds the current k-th distance. Results use float64 distances and lexicographic tie-breaking; this is not arbitrary-integer exact nearest-neighbor arithmetic.

```python
import numpy as np
from quadmath.lattice.lattice_search import nearest, within_radius

sites, d2 = nearest((0.4, -0.3, 0.9, 0.1), R=4.0, k=5)   # at most 5 within radius
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
| `lattice_search.squared_distance` | Float64 gauge-free squared distances, matching the integer identity within its numeric range |
| `lattice_search.within_radius(site, R)` | All sites within the supported radius, sorted with documented floating boundary semantics |
| `lattice_search.nearest(site, R, k)` | `k` nearest supported sites, shell sweep with a conservative early-stop bound |

## Enumeration and query regression scope

The regressions compare enumeration with independent BFS through frequency 8, metric identities with actual embeddings and queries with brute-force filtering. Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python 3.13 coverage run -m pytest -q
uv run --python 3.13 coverage report
```
