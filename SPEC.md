# QuadMath Mathematical Specification — Quadray Lattice Core

**Status.** This is the canonical mathematical specification of the QuadMath lattice core
(quadray coordinates, the Urner embedding, exact conversion and inversion, exact distances,
shell structure, close-packing numbering, and nearest-site search). Its executable
transcription is `tests/test_spec_examples.py`: every normative claim below carries an
identifier `S#` and is exercised by exactly one named test; the set of test names in that
file equals the set of names in the claim index (§7.5), enforced mechanically in both
directions. The manuscript projection of this specification is
`quadmath/markdown/14_conversions_spec.md` (same definitions, same error taxonomy, same
guarantees, prose instead of source); that section already points back here.

All claims are stated against the landed code in `src/` and were verified numerically on it.

---

## 1. Scope, conventions, and notation

### 1.1 What is specified

Normative for these public surfaces:

| Module | Surfaces specified here |
| --- | --- |
| `src/quadray.py` | `Quadray` (dataclass, `normalize`, `as_tuple`, `add`, `sub`), `to_xyz`, `DEFAULT_EMBEDDING`, `quadray_from_xyz`, `integer_tetra_volume`, `ace_tetravolume_5x5`, `magnitude`, `dot`, `distance`, `angle`, `centroid` |
| `src/conversions.py` | `urner_embedding`, `quadray_to_xyz`, `xyz_to_quadray_canonical`, `quadray_roundtrip`, `embedding_basis` |
| `src/ivm_field.py` | lattice predicates and enumeration only: `is_ivm_site`, `quadray_shell_norm`, `shell_sites`, `ball_sites`, `shell_cardinalities`, `IVM_NEIGHBOR_STEPS` |
| `src/omni_numbering.py` | `NEIGHBOR_MOVES`, `MAX_SHELL`, `shell_count`, `cumulative_count`, `generate_shell`, `sites_through_shell`, `site_index`, `site_at_index` |
| `src/lattice_search.py` | `squared_distance`, `nearest`, `within_radius` |
| `lean/` | the Lean 4 mirror of the same mathematics (status in §7.3) |

**Out of scope** (not specified here; they carry their own tests and manuscript sections):
the field-learning machinery (`IVMField`, `fit_geometry` in `src/ivm_field.py`), the
dynamics modules (`discrete_variational`, `ivm_dynamics`), optimization and symbolic
modules (`nelder_mead_quadray`, `cayley_menger`, `symbolic`), the Einstein.4D namespace
(`geometry.py`), information geometry (`information.py`, `metrics.py`), and all
visualization/pipeline scripts.

### 1.2 Conventions

- A **quadray** is an integer 4-tuple $(a, b, c, d) \in \mathbb{Z}^4$ (Fuller.4D).
- Two quadrays are **projectively equivalent**, written $q \sim q'$, when
  $q' = q + t\,\mathbf{1}$ for some $t \in \mathbb{Z}$, where
  $\mathbf{1} = (1, 1, 1, 1)$. The **canonical representative** of a class is its unique
  representative with all components non-negative and at least one component zero;
  `Quadray.normalize` selects it by subtracting $(k,k,k,k)$ with $k = \min_i q_i$.
- **XYZ** is the Coxeter.4D Cartesian image $\mathrm{xyz}(q) = M\,q \in \mathbb{R}^3$.
- An **IVM lattice site** is a quadray whose canonical representative has component sum
  $s \equiv 0 \pmod 4$. The other three residue classes ($s \equiv 1, 2, 3 \pmod 4$) are
  the tetrahedral ($1, 3$) and octahedral ($2$) **voids** of the close packing — not sites.
- **Shell $k$** ($k \ge 0$) is the set of IVM sites with shell norm $N(q) = 2k$ (§4.6).

### 1.3 Notation

- $M$: a $3 \times 4$ embedding matrix with rows $r_0, r_1, r_2$ and columns
  $C_0, \dots, C_3$; $c$: the uniform scale of an Urner embedding (the manuscript's
  "scale factor"); $M_0$: the unscaled Urner matrix ($c = 1$).
- $\mathbf{1} = (1,1,1,1)$; $J_4$: the $4 \times 4$ all-ones matrix; $\delta_{ij}$: the
  Kronecker delta; $I_n$: the $n \times n$ identity.
- $s = \sum_i q_i$ (component sum of a quadray); $w = q - \frac{s}{4}\,\mathbf{1}$: the
  sum-zero representative of the class (defined on $\mathbb{Q}^4$ for arbitrary $q$).
- $N(q) = \sum_i \lvert q_i - s/4 \rvert$: the shell norm (§4.6).
- $d^2(p, q)$: squared Euclidean distance of the images $\mathrm{xyz}(p), \mathrm{xyz}(q)$.

---

## 2. Quadray coordinates, projective classes, and tetra-volume

### 2.1 Canonical representative (S1)

`Quadray` is a frozen dataclass of four `int` components. `Quadray.normalize` subtracts
$(k,k,k,k)$ with $k = \min_i q_i$, producing the unique canonical representative
(non-negative components, minimum zero) of the projective class. Normalization is
idempotent and stays inside the class:

$$\mathrm{normalize}(q) \sim q, \qquad \mathrm{normalize}(\mathrm{normalize}(q)) = \mathrm{normalize}(q).$$

Worked examples (exact): $\mathrm{normalize}(3,2,2,1) = (2,1,1,0)$ and
$\mathrm{normalize}(1,0,0,-1) = (2,1,1,0)$ — two different representatives of one class
map to the same canonical quadray.

### 2.2 The forward map (S2)

`quadray.to_xyz(q, embedding)` evaluates the matrix product $\mathrm{xyz}(q) = M\,q$
(rows of `embedding` are the rows of $M$). Under `DEFAULT_EMBEDDING` (§3.1):

$$(2,1,1,0) \mapsto (0, 2, 2), \qquad (1,1,1,0) \mapsto (-1, 1, 1).$$

The twelve shell-1 sites (§4.8) map to the twelve vertices of type $(\pm 2, \pm 2, 0)$ —
the cuboctahedron (vector equilibrium) of edge length $2\sqrt{2}$.

### 2.3 The projective fiber (S3)

Every row of the Urner embedding sums to exactly zero, so $M\,\mathbf{1} = 0$: the forward
map is constant on projective classes,

$$\mathrm{to\_xyz}(q + t\,\mathbf{1}) \;=\; \mathrm{to\_xyz}(q) \quad \text{for every } t \in \mathbb{Z}.$$

Conversely, because $\operatorname{rank} M = 3$ (S23), the kernel of $M$ on $\mathbb{R}^4$
is exactly $\operatorname{span}\{\mathbf{1}\}$: two quadrays with the same image differ by
an integer multiple of $\mathbf{1}$. Normalization (S1) therefore selects *the* canonical
representative of the fiber.

### 2.4 Exact tetra-volume (S4)

The exact IVM tetra-volume of the lattice tetrahedron spanned by $p_0, p_1, p_2, p_3$ is

$$V_{\mathrm{ivm}} = \frac{\lvert \det \,\mathrm{edges} \rvert}{4},$$

where `quadray.integer_tetra_volume` evaluates $\lvert\det\rvert$ with the exact integer
Bareiss determinant on the $(a-d, b-d, c-d)$ projection of the edge vectors
$p_i - p_0$, and `quadray.ace_tetravolume_5x5` evaluates the Tom Ace $5\times 5$
determinant of $[[a\,b\,c\,d\,1]; \ldots; [1\,1\,1\,1\,0]]$, again returning
$\lvert\det\rvert/4$ as an exact `fractions.Fraction`. For integer quadray vertices the
two determinants have equal magnitude, so **the two functions agree exactly on every
integer input**. Worked values:

- **Unit tetrahedron** $(0,0,0,0), (2,1,1,0), (1,2,1,0), (1,1,2,0)$: determinant $4$,
  volume exactly $1$ (one IVM unit — four ABCD volumes).
- **Primitive tetrahedron** $(0,0,0,0), (1,0,0,0), (0,1,0,0), (0,0,1,0)$: determinant
  $1$, volume exactly $1/4$.
- General tetrahedron $(0,0,0,0), (4,2,2,0), (2,4,2,0), (2,2,4,0)$: volume exactly $8$
  (both functions, exactly equal). General lattice tetrahedra may have non-integral
  volume; no integrality is assumed.

### 2.5 Embedded geometry (S5)

`quadray.magnitude`, `dot`, `distance`, `angle` evaluate Euclidean norms, dot products,
distances, and angles of the embedded images under a given embedding. Worked values under
`DEFAULT_EMBEDDING` (all exact up to float representation of $\sqrt{8}$ and $\pi/3$):

$$\lVert (2,1,1,0) \rVert = 2\sqrt{2}, \quad
\langle (2,1,1,0),\, (1,2,1,0) \rangle = 4, \quad
\angle\big((1,2,1,0),\, (0,0,0,0),\, (2,1,1,0)\big) = \frac{\pi}{3},$$

and `distance((0,0,0,0), (2,1,1,0))` $= 2\sqrt{2}$ — the dot product and angle follow
from the column Gram identity (S9).

---

## 3. The embedding and its projective fiber

### 3.1 The Urner matrix (S6)

`conversions.urner_embedding(scale=1.0)` returns exactly

$$M(c) \;=\; c \times
\begin{pmatrix}
\;\,1 & -1 & -1 & \;\,1 \\
\;\,1 & \;\,1 & -1 & -1 \\
\;\,1 & -1 & \;\,1 & -1
\end{pmatrix},$$

and every row sums to exactly $0$. `quadray.DEFAULT_EMBEDDING` is the same unscaled
matrix stored as a tuple of rows: the two agree entry for entry at $c = 1$, and
`quadray_to_xyz(q, M=None)` with `M=None` selects exactly this default (S7). The rows of
$M$ send the four quadray axes to the vertices of a regular tetrahedron in $\mathbb{R}^3$.

### 3.2 Forward delegation and shape validation (S7)

`conversions.quadray_to_xyz(q, M=None)` delegates to `quadray.to_xyz`:
`quadray_to_xyz(q)` is *identical* to `quadray.to_xyz(q, quadray.DEFAULT_EMBEDDING)` (an
explicit `M` is threaded through the shared internal helper `_embedding_rows`). An
explicit embedding that is not of shape $(3,4)$ raises `ValueError`.

---

## 4. Gram identities, exact distances, and the IVM lattice

### 4.1 Row Gram identity (S8)

For $M = M(c)$:

$$M\,M^{\mathsf T} \;=\; 4\,c^{2}\,I_{3}.$$

Each row has two $+1$ and two $-1$ entries (norm $2\lvert c\rvert$); distinct rows are
orthogonal. This identity is what makes `quadray.quadray_from_xyz`'s pseudoinverse
$M^+ = M^{\mathsf T}(M M^{\mathsf T})^{-1}$ a division by $4c^2$ in effect (§6.1).

### 4.2 Column Gram identity and the embedding basis (S9)

With $C_j$ the four columns of $M$ and $J_4$ the all-ones matrix:

$$M^{\mathsf T} M \;=\; 4\,I_{4} - J_{4},
\qquad\text{i.e.}\qquad
C_i \cdot C_j \;=\; 4\,\delta_{ij} - 1 \quad (c = 1),$$

diagonal $3$, off-diagonal $-1$. `conversions.embedding_basis(M=None)` returns the
$(4,3)$ matrix $B$ whose **rows** are the embedding **columns** $C_0, \dots, C_3$ (row
$i$ of $B$ is column $i$ of $M$), so $B\,B^{\mathsf T} = 4I_4 - J_4$ at $c=1$; scaling
the embedding by $c$ scales the Gram matrix by $c^2$.

### 4.3 The exact distance identity (S10)

For integer quadrays $p, q$ with $\delta = p - q$, the squared Euclidean distance of
their images is the exact integer

$$d^{2}(p, q) \;=\; \sum_i (M\delta)_i^{2} \;=\; 4\sum_j \delta_j^{2} \;-\; \Bigl(\sum_j \delta_j\Bigr)^{2},$$

the expansion of $\delta^{\mathsf T}(M^{\mathsf T}M)\delta$ under (S9).
`lattice_search.squared_distance(p, sites)` evaluates exactly this identity (float64
arithmetic; exact for the integer magnitudes used in search). Because the identity is
exact on integers, nearest-site decisions never suffer floating-point boundary errors.
Worked values: $d^2\big(0, (2,1,1,0)\big) = 4\cdot 6 - 16 = 8$;
$d^2\big(0, (4,2,2,0)\big) = 4\cdot 24 - 64 = 32$. The identity is cross-validated
against the float pipeline `quadray.distance` (e.g.
$\mathrm{distance}\big(0, (2,1,1,0)\big) = 2\sqrt{2} = \sqrt{8}$).

### 4.4 The shell truncation bound (S11)

Every shell-$g$ IVM site ($g \ge 1$) satisfies

$$d^{2}\big(\mathbf{0},\, s\big) \;\ge\; 8\,g .$$

Machine-verified exhaustively through shell 4 (309 sites): the minima are exactly
$8, 16, 40, 64$ for $g = 1..4$ — the bound is **tight** at $g = 1$ and $g = 2$. This
invariant is what makes shell-ordered sweeps in `lattice_search` safe to truncate: a site
within Euclidean radius $R$ of a center of norm $\rho$ lies on a shell
$g \le (\rho + R)^2 / 8$ (triangle inequality plus the bound), which is the enumeration
depth `_max_shell_for` computes.

### 4.5 IVM membership (S12)

`ivm_field.is_ivm_site(q)` is true if and only if the canonical representative of $q$ has
component sum $\equiv 0 \pmod 4$. Normalization shifts the sum by a multiple of $4$, so
membership is a property of the projective class. Worked facts:
$\mathrm{is\_ivm\_site}(2,1,1,0)$ true; the void directions $(1,0,0,0)$, $(1,1,0,0)$,
$(1,1,1,0)$ (sums $1, 2, 3$) false; $(3,2,2,1)$ has the same class as $(2,1,1,0)$, so it
is a site too; $(2,2,2,1)$ (canonical $(1,1,1,0)$, sum $3$) is not.

### 4.6 The shell norm (S13)

For a quadray with component sum $s$:

$$N(q) \;=\; \sum_{i} \Bigl\lvert q_i - \frac{s}{4} \Bigr\rvert ,$$

the $L_1$ magnitude of the sum-zero representative $w = q - \frac{s}{4}\mathbf{1}$.
`ivm_field.quadray_shell_norm(q)` normalizes $q$ first, so the norm is a class property;
it raises `ValueError` when the canonical component sum is not divisible by $4$ (a void).
On shell-$k$ sites $N(q) = 2k$. Worked values: $N(0,0,0,0) = 0$;
$N(2,1,1,0) = 2$; $N(0,0,1,3) = 4$ and $N(4,2,2,0) = 4$ (shell-2 sites);
$N(3,2,2,1) = 2$ (class property); voids $(1,0,0,0)$, $(1,1,0,0)$, $(1,1,1,0)$ raise.

### 4.7 Shell enumeration and cardinalities (S14)

`ivm_field.shell_sites(k)` enumerates all IVM sites with shell norm $2k$: it scans the
bounding box $[0, 2k]^4$ (any shell-$k$ site has components at most $2k$ after
normalization — proved in Lean as `shellNorm4_component_bound`, §7.3), keeps canonical
quadrays whose component sum is divisible by 4 and whose shell norm is $2k$, and returns
them in lexicographic $(a,b,c,d)$ order. The cardinality is the cuboctahedral number

$$\lvert \mathrm{shell}_k \rvert = 10\,k^{2} + 2 \quad (k \ge 1), \qquad \lvert \mathrm{shell}_0 \rvert = 1,$$

i.e. $1, 12, 42, 92, 162$ for $k = 0..4$. `ivm_field.ball_sites(radius)` is the
shell-ordered union (309 sites through shell 4); `ivm_field.shell_cardinalities(max_shell)`
returns the per-shell counts. Worked order facts: `shell_sites(2)[:3]` =
$(0,0,1,3), (0,0,2,2), (0,0,3,1)$.

### 4.8 Twelve-around-one (S15)

`ivm_field.IVM_NEIGHBOR_STEPS` is exactly the twelve normalized permutations of the base
move $(2,1,1,0)$, in lexicographic order; `shell_sites(1)` is the same twelve sites. Two
sites are graph-adjacent exactly when their difference normalizes to one of these steps.
Under `DEFAULT_EMBEDDING` each move's image is a vertex of type $(\pm 2, \pm 2, 0)$, and
each sits at squared distance exactly $8$ from the origin (`d^2(0, s) = 8`, S10) — the
cuboctahedron (vector equilibrium) of the first frequency shell.

### 4.9 Close-packing numbering (S16, S17)

`omni_numbering` enumerates the same shells by vectorized layer BFS from the center using
the 12 neighbor moves (`NEIGHBOR_MOVES`, the $(12,4)$ sorted int64 permutation array of
$(2,1,1,0)$), agreeing with §4.7:

- **Shell counts (S16).** `shell_count(k)` $= 1$ for $k = 0$ and $10k^2 + 2$ for
  $k \ge 1$; `cumulative_count(k)` $= 1 + \sum_{j=1}^{k}(10j^2+2)$ with the closed form

  $$\mathrm{cumulative\_count}(k) \;=\; 1 \;+\; 2k \;+\; \frac{10\,k\,(k+1)\,(2k+1)}{6},$$

  giving $1, 13, 55, 147, 309, 561, 923$ through shell $6$.

- **Canonical global index (S17).** `sites_through_shell(max_shell)` lists sites in
  canonical order — shell-major, lexicographic within each shell, center first (row 0 is
  $(0,0,0,0)$; the first shell-1 row is $(0,1,1,2)$). `site_index(site, max_shell)` and
  `site_at_index(index, max_shell)` are mutual inverses on that enumeration (bijection);
  a non-site such as the void $(1,0,0,0)$ has index $-1$. The enumeration supports up to
  `MAX_SHELL = 32`; `generate_shell(k)` returns shell $k$'s slice in the same
  lexicographic order.

### 4.10 Nearest-site search (S18, S19)

`lattice_search.within_radius(site, R)` returns all sites with $d^2 \le R^2$ of the
center, enumerated exactly through the shell depth required by (S11) and filtered with
the exact identity (S10); `lattice_search.nearest(site, R, k)` returns the $k$ closest
such sites, sweeping shells outward (S18). Both are **deterministic**: results are sorted by
$d^2$ ascending with lexicographic $(a,b,c,d)$ order breaking ties. Worked case: center
$(0.5, 1.0, -0.5, 2.0)$, $R = 3$, $k = 5$ returns five sites at
$d^2 = 1, 5, 5, 5, 5$, the first being $(1,1,0,2)$; `within_radius((0,0,0,0), 2\sqrt{2})`
returns 13 sites — the center plus the twelve shell-1 sites, all twelve tied at $d^2=8$
in lexicographic order. Queries whose required enumeration depth exceeds `MAX_SHELL = 32`
raise `ValueError` rather than silently truncating results (S19).

---

## 5. Exact canonical inversion

### 5.1 Exact recovery (S20)

`conversions.xyz_to_quadray_canonical(xyz, M=None)` inverts the embedding **exactly**:
all arithmetic is `fractions.Fraction` (a float entry enters as its exact binary
rational; no rounding, no tolerance anywhere). For every canonical IVM site $q$ through
shell 4 — all 309 sites of §4.7 —

$$\mathrm{xyz\_to\_quadray\_canonical}\big(\mathrm{to\_xyz}(q)\big) \;=\; q,$$

and the same holds for every shell-0..4 site embedded at scale $c = 1/2$ (the
half-integer FCC picture of the same sites, through the same scaled matrix).

### 5.2 Exact entry types (S21)

`xyz` and `M` entries may be `int`, `float`, `fractions.Fraction`, or a NumPy integer
(`np.int64` via `numbers.Integral`); `np.float64` is a `float` subclass and is accepted.
Every accepted entry is converted to its **exact** rational value. Worked examples, all
recovering `Quadray(1,1,1,0)` from the image of $(1,1,1,0)$, namely $(-1, 1, 1)$:
a `Fraction` triple, an `np.int64` triple, and an `np.float64` array.

### 5.3 The fiber and its tie-break (S22)

Because every row of $M$ sums to zero (S6) and the kernel is
$\operatorname{span}\{\mathbf{1}\}$ (S3), the fiber of an image point is the full coset
$\{\,y + t\,\mathbf{1} : t \in \mathbb{Z}\,\}$, where $y = (x_1, x_2, x_3, 0)$ is the
particular preimage obtained by solving the $3\times 3$ system on columns $0..2$ by
Cramer's rule. The function returns
`Quadray(x1, x2, x3, 0).normalize()` — **the min-0 representative of the fiber**, which
is unique (S1), so the tie-break is deterministic. Worked example: the unnormalized
representative $(3,2,2,1)$ has the same image $(0,2,2)$ as $(2,1,1,0)$, and the function
returns exactly $(2,1,1,0)$.

### 5.4 The rank lemma (S23)

With all row sums zero, $C_3 = -(C_0 + C_1 + C_2)$, so

$$\operatorname{rank}(M) = 3 \quad\Longleftrightarrow\quad \det\,[\,C_0\;C_1\;C_2\,] \neq 0,$$

and for the Urner embedding $\det\,[\,C_0\;C_1\;C_2\,] = 4\,c^{3}$ (exact $4$ at $c=1$;
exactly $4s^3$ over `Fraction` for rational $s$, e.g. $1/2$ at $c = 0.5$). The function
requires every row to sum to exactly $0$ (`ValueError` otherwise, §5.6) and raises
`ValueError` ("rank < 3") when the column determinant vanishes — e.g. for
$[[1,-1,-1,1], [2,-2,-2,2], [0,0,0,0]]$, whose rows are dependent with zero row sums.

### 5.5 Image membership (S24)

With $y$ as in (S22), $xyz$ lies in the lattice image $\{M q : q \in \mathbb{Z}^4\}$ if
and only if $x_1, x_2, x_3$ are all integers (every integer preimage is
$y + t\,\mathbf{1}$, $t \in \mathbb{Z}$). A non-integral preimage raises `ValueError`
("not in the embedding image") — **the point is rejected, never snapped** to a neighbor.
Worked rejections: $(0.5, 0.5, 0.5)$ has exact preimage $(\tfrac12, 0, 0, 0)$ and
$(0.1, 0, 0)$ has non-integral preimage $(-\tfrac{1}{20}, \tfrac{1}{20}, \tfrac{1}{20})$
— both rejected.

### 5.6 Fail-closed taxonomy (S25)

The inverse is fail-closed, with `ValueError` for structural mismatches and `TypeError`
for non-numeric entries:

- `ValueError`: `xyz` not a length-3 sequence ("xyz must be a length-3 sequence");
  `xyz` of the wrong length ("xyz must have length 3, got N"); `M` of the wrong shape
  ("shape"); any embedding row not summing to exactly 0 ("does not sum to exactly 0");
  rank-deficient embedding ("rank < 3", S23); non-image point ("not in the embedding
  image", S24).
- `TypeError`: an `xyz` or `M` entry that is not `int`/`float`/`Fraction`/
  `numbers.Integral` — e.g. a `str` entry (note: a length-3 `str` passes the length
  check and then fails entry conversion) or an `np.float32` entry, which is rejected.

---

## 6. The float inverse and the round-trip contract

### 6.1 The round-trip identity (S26)

`conversions.quadray_roundtrip(q, M=None)` runs both legs through the **same** embedding
rows: forward via `quadray.to_xyz`, inverse via `quadray.quadray_from_xyz`, and asserts
the exact identity

$$\mathrm{from\_xyz}\big(\mathrm{to\_xyz}(q)\big) \;=\; q .$$

It holds exactly for every canonical (normalized) integer quadray through shell 4 — all
309 sites — under the default embedding. It raises `AssertionError` explicitly (not a
bare `assert`, so the check survives `python -O`) when it fails.

### 6.2 Scale robustness (S27)

The identity is scale-independent across the whole Urner family: for $M = c\,M_0$ with
any $c \neq 0$,

$$\operatorname{pinv}(cM)\,(cM\,q) \;=\; q \;-\; \frac{\sum_i q_i}{4}\,\mathbf{1},$$

because $\operatorname{pinv}(cM) = \tfrac{1}{c}\operatorname{pinv}(M)$ and the $(1/c)$
cancels the scale of the image. The projection lands on the same sum-zero representative
at every scale, and round-half-up preserves the $(1,1,1,1)$-coset (S29), so
`quadray_roundtrip(q, urner_embedding(c))` is exact for all 309 shell-0..4 sites at
$c = 1/2$ and $c = 2$ (the half-integer FCC picture round-trips through the same scaled
embedding).

### 6.3 The documented failure (S28)

The single documented `AssertionError` case is an **unnormalized** input: the inverse
canonicalizes to the min-0 representative. `quadray_roundtrip(Quadray(2,2,2,1))` raises
`AssertionError` with message "roundtrip is not the identity", because
$(2,2,2,1) \mapsto (-1,1,1) \mapsto (1,1,1,0) = \mathrm{normalize}(2,2,2,1)$. Embeddings
whose rows do not sum to zero lie outside the Urner family (S6) and outside this
guarantee.

### 6.4 Pseudoinverse + round-half-up preserves the coset (S29)

`quadray.quadray_from_xyz(x, y, z, embedding)` computes the minimum-norm real preimage
$q_{\mathrm{real}} = M^{+}\,\mathrm{xyz}$ (the sum-zero representative
$w = q - \frac{s}{4}\mathbf{1}$ of the true class, since $\sum_i q_{\mathrm{real},i} = 0$
by (S6)), rounds **half-up** (`floor(v + 0.5)`, not banker's rounding), and normalizes.
For points that lie exactly on the quadray lattice — of any residue class — the rounded
point stays in the input's $(1,1,1,1)$-coset, so the round trip through `from_xyz` is
exact. Worked cases, exact at scales $c = 1$ and $c = 1/2$ alike: $(1,1,0,0)$ (sum
$\equiv 2 \pmod 4$: the projection lands on exact half-integer ties in all four
components, and half-up keeps the class), $(1,0,0,0)$, $(1,1,1,0)$, the unnormalized
$(2,2,2,1)$ (returned as its canonical $(1,1,1,0)$), and the site $(0,1,1,2)$.

### 6.5 Snapping, not inversion, for general points (S30)

For a general $\mathbb{R}^3$ point the result is the nearest lattice point **in quadray
coordinates** (component-wise rounding of $q_{\mathrm{real}}$), which is **not always the
nearest in embedded XYZ distance**. Worked counterexample under `DEFAULT_EMBEDDING`: the
point $xyz = (-\tfrac14, \tfrac34, \tfrac34)$ is the image of the sum-zero vector
$w = (\tfrac{5}{16}, \tfrac{1}{16}, \tfrac{1}{16}, -\tfrac{7}{16})$; every component of
$w$ rounds to $0$, so `quadray_from_xyz` returns $(0,0,0,0)$ at squared XYZ distance
$\tfrac{19}{16}$ — while the genuinely closest quadray lattice point is $(1,1,1,0)$, at
squared distance $\tfrac{11}{16} < \tfrac{19}{16}$. Component-wise rounding minimizes
$\sum \delta^2$ but ignores the $-(\sum\delta)^2$ term of the exact identity (S10); the
function is a snapping operation, not an exact inverse (that is S20's role).

---

## 7. Verification, status, and the claim index

### 7.1 Executable transcription

`tests/test_spec_examples.py` transcribes every claim of this specification as one named
test (deterministic, real numerical examples, no mocks, stdlib `fractions` allowed, no
new dependencies). Run it from the repo root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest tests/test_spec_examples.py -q
```

The set of test functions in that file equals the set of names in the claim index
(§7.5) — in both directions.

### 7.2 Companion coverage

`tests/test_conversions.py` (landed with the conversions wave) covers the same layer
independently: delegation, 309-site round-trips through shells 0–4, scale-robust
round-trips ($c = 0.5, 2.0$), exact canonical recovery (default and half-scale
embeddings), the fiber tie-break, the fail-closed taxonomy, both Gram identities, and
the distance identity cross-checked over 13 centers × 55 targets. The two suites are
complementary: the companion tests the code, this specification's tests pin the
mathematical claims and their worked examples.

### 7.3 Lean 4 mirror status

The Lean 4 formalization in `lean/` (core Lean only, no Mathlib) mirrors this
specification. **The tree carries no `sorry`.** Its honest status:

- **Proved (kernel-checked).** In `QuadMath.Quadray`: normalization lands in the class,
  is idempotent, and yields non-negative components with a zero component
  (`normalize_sameClass`, `normalize_spec`, `quadMin_normalize`, `normalize_idem`); the
  tetra-determinant is class-invariant (`tetraDet_congr`); the unit tetrahedron has
  determinant 4 / volume 1 and the primitive tetrahedron determinant 1 / volume 1/4
  (`tetraDet_unit`, `tetraVolume_unit`, `tetraDet_primitive`, `tetraVolume_primitive`).
  In `QuadMath.IVM`: the 12 neighbor moves are pairwise distinct and are all IVM sites
  (`neighborMoves_nodup`, `neighborMoves_all_sites`); normalization shifts the component
  sum by a multiple of 4 (`ivmSum_normalize`); membership and shell norm are class
  properties (`isIVMSite_normalize`, `shellNorm4_sameClass`); **scan-box completeness** —
  every shell-`k` site has all components in $[0, 2k]$, so the enumeration of §4.7 loses
  no sites (`shellNorm4_component_bound`); the **enumeration characterization**
  (`mem_shellSites_iff`); shell membership determines the shell index
  (`shellSites_unique`). In `QuadMath.Distance`: the exact distance identity (S10)
  **proved universally over all integer deltas** (`deltaDist2_eq`), with symmetry
  (`dist2_comm`), zero self-distance (`dist2_self`), and the twelve-around-one instance
  $d^2 = 8$ (`dist2_neighborMove`). In `QuadMath.Lattice`: `omniNumber k = 10k² + 2`
  with the enumeration realizing it on shells 1–3 (`shellSites_omni_1/2/3`).
- **Machine-checked, not proved.** The *universal* shell-count theorem
  `shellSites_card_target k : (shellSites k).length = 10k² + 2` is stated as a
  proposition-valued definition and verified **by computation** for shells 1–4 with
  kernel-evaluated `decide` (axiom audit: depending on no axioms) and shells 5–8 with
  `native_decide` (the kernel trusts the VM result through one per-instance
  `native_decide` axiom). The general case for arbitrary $k$ remains an **open proof
  obligation**; keeping it as a proposition definition (rather than a `sorry`) leaves
  the tree with zero unproved declarations. The proof sketch recorded in
  `lean/README.md` (counting sum-zero representatives by sign pattern via
  stars-and-bars) is future work.
- See `lean/README.md`, section "What is proved vs. machine-checked", for the
  authoritative per-theorem list and the build commands.

### 7.4 Manuscript projection

`quadmath/markdown/14_conversions_spec.md` is the manuscript projection of this
specification: its equations `eq:conv-embedding`, `eq:conv-fiber`, `eq:conv-gram-row`,
`eq:conv-gram-col`, `eq:conv-distance`, `eq:conv-canonical`, `eq:conv-roundtrip`,
`eq:conv-roundtrip-scale`, and `eq:conv-shell` correspond to the claims of §§3–4 and
§§5–6 here (S6, S3/S22, S8, S9, S10, S20/S22, S26, S27, S13 respectively). The shell
machinery it references in its "Lattice context" section is specified in §§4.5–4.9.

### 7.5 Claim index

One row per normative claim. "Implemented by" names the landed function that realizes
the claim; "Exercised by" names the test in `tests/test_spec_examples.py` whose name
must match exactly (the test file contains these names and no others, and every test
function appears here).

| # | Claim | Implemented by | Exercised by |
| --- | --- | --- | --- |
| S1 | `Quadray.normalize` selects the unique canonical min-0 representative; idempotent; stays in the projective class | `quadray.Quadray.normalize` | `test_normalize_selects_unique_canonical_representative` |
| S2 | `to_xyz` evaluates the matrix product $M\,q$ with the pinned worked values | `quadray.to_xyz` | `test_forward_map_matrix_product_values` |
| S3 | Every embedding row sums to 0, so the forward map is constant on projective classes $q + t\,\mathbf{1}$ | `quadray.to_xyz`, `conversions.urner_embedding` | `test_forward_map_constant_on_projective_fiber` |
| S4 | Tetra-volume is $\lvert\det\rvert/4$ exactly; unit tetra = 1, primitive = 1/4; `integer_tetra_volume` and `ace_tetravolume_5x5` agree exactly on integer vertices | `quadray.integer_tetra_volume`, `quadray.ace_tetravolume_5x5` | `test_tetra_volume_unit_primitive_and_ace_agreement` |
| S5 | Embedded geometry: magnitude/dot/distance/angle consistent with the Gram matrix; worked values $2\sqrt2$, $4$, $\pi/3$ | `quadray.magnitude`, `quadray.dot`, `quadray.distance`, `quadray.angle` | `test_embedded_geometry_exact_values` |
| S6 | `urner_embedding(scale)` returns exactly the pinned $3\times4$ matrix times the scale; every row sums to exactly 0; `DEFAULT_EMBEDDING` equals it at scale 1 | `conversions.urner_embedding`, `quadray.DEFAULT_EMBEDDING` | `test_urner_embedding_matrix_rows_and_row_sums` |
| S7 | `quadray_to_xyz(q, M=None)` is identical to `quadray.to_xyz(q, DEFAULT_EMBEDDING)`; explicit $M$ must have shape $(3,4)$ | `conversions.quadray_to_xyz` | `test_quadray_to_xyz_delegates_and_shape_validates` |
| S8 | $M\,M^{\mathsf T} = 4c^{2} I_3$ for the Urner embedding at scale $c$ | `conversions.urner_embedding` | `test_row_gram_identity_scales_quadratically` |
| S9 | $M^{\mathsf T}M = 4I_4 - J_4$ ($C_i \cdot C_j = 4\delta_{ij} - 1$ at $c=1$); `embedding_basis` returns the $(4,3)$ column basis $B$ with $B\,B^{\mathsf T} = 4I_4 - J_4$, scaling by $c^2$ | `conversions.embedding_basis` | `test_embedding_basis_column_gram_identity` |
| S10 | $d^2 = 4\sum\delta^2 - (\sum\delta)^2$ is the exact squared distance; `squared_distance` implements it; cross-checked against the float pipeline `quadray.distance` | `lattice_search.squared_distance`, `quadray.distance` | `test_distance_identity_matches_float_pipeline` |
| S11 | Every shell-$g$ site ($g \ge 1$) satisfies $d^2(\mathbf{0}, s) \ge 8g$; tight at $g = 1, 2$ | `lattice_search.squared_distance`, `omni_numbering.sites_through_shell` | `test_shell_truncation_bound_8g` |
| S12 | IVM site membership: normalized $\wedge$ sum $\equiv 0 \pmod 4$; a class property; void classes rejected | `ivm_field.is_ivm_site` | `test_ivm_membership_residue_classes` |
| S13 | Shell norm $N(q) = \sum\lvert q_i - s/4 \rvert = 2k$ on shell-$k$ sites; class property; voids raise `ValueError` | `ivm_field.quadray_shell_norm` | `test_shell_norm_values_and_void_rejection` |
| S14 | `shell_sites(k)` enumerates shell $k$ exactly (box $[0,2k]^4$, lexicographic) with cuboctahedral cardinalities $10k^2+2$ ($1,12,42,92,162$ for $k=0..4$); `ball_sites` is the shell-ordered union | `ivm_field.shell_sites`, `ivm_field.shell_cardinalities`, `ivm_field.ball_sites` | `test_shell_enumeration_cuboctahedral_cardinalities` |
| S15 | The neighbor moves are exactly the 12 normalized permutations of $(2,1,1,0)$ = shell 1; images are the $(\pm2,\pm2,0)$ cuboctahedron vertices, each at $d^2 = 8$ | `ivm_field.IVM_NEIGHBOR_STEPS` | `test_twelve_neighbor_moves_cuboctahedron` |
| S16 | `shell_count(k)` $= 10k^2+2$ ($1$ for $k=0$) and `cumulative_count(k)` $= 1 + 2k + 10k(k+1)(2k+1)/6$ match the partial sums through shell 6 | `omni_numbering.shell_count`, `omni_numbering.cumulative_count` | `test_shell_count_and_cumulative_closed_form` |
| S17 | `sites_through_shell` is shell-major lexicographic (center first); `site_index`/`site_at_index` are mutual inverses; non-sites map to $-1$; `MAX_SHELL = 32` | `omni_numbering.sites_through_shell`, `omni_numbering.site_index`, `omni_numbering.site_at_index` | `test_omni_numbering_bidirectional_bijection` |
| S18 | `within_radius`/`nearest` return exact-$d^2$ results consistent with brute force, sorted by $d^2$ then lexicographic $(a,b,c,d)$ | `lattice_search.nearest`, `lattice_search.within_radius` | `test_nearest_within_radius_exact_and_deterministic` |
| S19 | Search is fail-closed: queries needing enumeration beyond `MAX_SHELL` raise `ValueError` instead of truncating | `lattice_search.nearest`, `lattice_search.within_radius` | `test_search_fails_closed_beyond_max_shell` |
| S20 | Exact canonical recovery: `xyz_to_quadray_canonical(to_xyz(q)) == q` for all 309 canonical sites through shell 4, also at scale $1/2$ | `conversions.xyz_to_quadray_canonical` | `test_canonical_inversion_exact_recovery_shells_0_to_4` |
| S21 | Exact entry types: `int`/`float`/`Fraction`/`np.int64`/`np.float64` accepted and converted to exact rationals | `conversions.xyz_to_quadray_canonical` | `test_canonical_inversion_exact_entry_types` |
| S22 | The fiber is $y + t\,\mathbf{1}$; the inverse returns the unique min-0 representative (worked: $(3,2,2,1)$'s image $\to$ $(2,1,1,0)$) | `conversions.xyz_to_quadray_canonical` | `test_canonical_inversion_fiber_tiebreak_min_zero` |
| S23 | Rank lemma: with zero row sums, rank $M = 3$ iff $\det[C_0\,C_1\,C_2] \neq 0$; Urner determinant $= 4c^3$; rank-deficient input raises `ValueError` | `conversions.xyz_to_quadray_canonical` | `test_canonical_inversion_rank_lemma` |
| S24 | $xyz$ is in the image iff the preimage is integral; non-image points raise `ValueError` (never snapped) | `conversions.xyz_to_quadray_canonical` | `test_canonical_inversion_rejects_non_image` |
| S25 | Fail-closed taxonomy: wrong length/shape/row sums $\to$ `ValueError`; non-numeric entries (`str`, `np.float32`) $\to$ `TypeError` | `conversions.xyz_to_quadray_canonical` | `test_canonical_inversion_fail_closed_taxonomy` |
| S26 | Round-trip identity `quadray_roundtrip(q) == q` for all 309 canonical sites through shell 4; both legs share the same rows | `conversions.quadray_roundtrip` | `test_roundtrip_identity_shells_0_to_4` |
| S27 | Scale robustness: exact at every Urner scale $c \neq 0$; $\operatorname{pinv}(cM)(cM\,q) = q - (\sum q / 4)\mathbf{1}$ | `conversions.quadray_roundtrip` | `test_roundtrip_scale_robustness` |
| S28 | Documented failure: unnormalized input raises `AssertionError` "roundtrip is not the identity"; $(2,2,2,1) \to (1,1,1,0)$ | `conversions.quadray_roundtrip` | `test_roundtrip_documented_failure_unnormalized` |
| S29 | `quadray_from_xyz` = pseudoinverse projection + round-half-up preserving the $(1,1,1,1)$-coset; lattice points of every residue class round-trip exactly (scales 1 and 1/2) | `quadray.quadray_from_xyz` | `test_quadray_from_xyz_coset_preserving_roundtrip` |
| S30 | For general points the result is the component-wise quadray-nearest, not always the XYZ-nearest: worked counterexample $(-\tfrac14, \tfrac34, \tfrac34)$ | `quadray.quadray_from_xyz` | `test_quadray_from_xyz_not_always_xyz_nearest` |
