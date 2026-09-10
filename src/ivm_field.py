"""Static field learning on the IVM (isotropic vector matrix) lattice.

This module implements machine learning on a static synergetic geometry:
scalar fields defined over the IVM (close-packed / face-centered cubic)
lattice whose sites are described by quadray coordinates.

Pieces:

- :func:`quadray_shell_norm` / :func:`is_ivm_site` — the IVM shell norm
  (centered L1 norm of the quadray's sum-zero representative) and the
  IVM-lattice membership test.
- :func:`shell_sites` / :func:`ball_sites` / :func:`shell_cardinalities` —
  exact integer enumeration of lattice shells.  Shell ``k`` collects all
  IVM sites of quadray norm ``2k`` and has cardinality ``10k**2 + 2`` for
  ``k >= 1`` (the cuboctahedral numbers 1, 12, 42, 92, 162, ...).
- :class:`IVMField` — a scalar field over a lattice ball of radius ``R``,
  stored as a numpy array keyed by a deterministic site index, with
  :meth:`IVMField.learn` (Laplacian-regularized kernel-weighted least
  squares on the lattice graph), :meth:`IVMField.predict` and
  :meth:`IVMField.score`.
- :func:`fit_geometry` — least-squares recovery of the best-fit tetrahedron
  orientation+scale from noisy 3D points via the quadray basis in
  :mod:`quadray` (the ``to_xyz`` mapping).

The learner is deterministic and uses numpy only (no ML frameworks); all
randomness (observation sampling, noise) is supplied by callers through a
seeded ``numpy.random.default_rng``.

Terms for the norm: a *lattice site* is a normalized quadray ``q`` (all
components non-negative, minimum zero) whose component sum is divisible by
4; these are exactly the close-packed sphere centers.  The remaining three
cosets of ``(1,1,1,1)`` are the octahedral (sum ≡ 2 mod 4) and tetrahedral
(sum ≡ 1, 3 mod 4) voids of the packing, not IVM sites.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from quadray import DEFAULT_EMBEDDING, Quadray, to_xyz

__all__ = [
    "IVM_NEIGHBOR_STEPS",
    "IVMField",
    "TetrahedronFit",
    "ball_sites",
    "fit_geometry",
    "is_ivm_site",
    "quadray_shell_norm",
    "shell_cardinalities",
    "shell_sites",
]


#: The 12 IVM neighbor moves from any lattice site: all permutations of
#: (2, 1, 1, 0), normalized.  Two sites are graph-adjacent iff their
#: difference normalizes to one of these steps.
IVM_NEIGHBOR_STEPS: Tuple[Quadray, ...] = tuple(
    Quadray(*p).normalize()
    for p in sorted(
        {
            (a, b, c, d)
            for a, b, c, d in (
                (2, 1, 1, 0), (2, 1, 0, 1), (2, 0, 1, 1),
                (1, 2, 1, 0), (1, 2, 0, 1), (1, 0, 2, 1),
                (1, 1, 2, 0), (1, 0, 1, 2), (0, 1, 1, 2),
                (0, 1, 2, 1), (0, 2, 1, 1), (1, 1, 0, 2),
            )
        }
    )
)

#: Numerical floor for kernel confidence weights; keeps the normal-equation
#: matrix ``W + lam * L`` positive definite on the connected lattice ball.
_WEIGHT_FLOOR = 1e-12


def is_ivm_site(q: Quadray) -> bool:
    """Return True iff ``q`` (after normalization) is an IVM lattice site.

    An IVM site is a normalized quadray (non-negative components, minimum
    zero) whose component sum is divisible by 4.  Normalization adds or
    subtracts ``(k, k, k, k)``, changing the sum by ``4k``, so membership is
    a property of the projective quadray class.
    """
    return sum(q.normalize().as_tuple()) % 4 == 0


def quadray_shell_norm(q: Quadray) -> int:
    """Return the IVM shell norm of ``q``: an even integer equal to ``2k``.

    The norm is the L1 magnitude of the sum-zero (Coxeter.4D hyperplane)
    representative of the quadray: with ``s = sum(q)``,
    ``N(q) = sum_i |q_i - s/4|``.  Shell ``k`` of the IVM lattice is the set
    of sites with ``N(q) = 2k`` and has exactly ``10k**2 + 2`` sites.

    Parameters
    - q: Quadray point; normalized internally, so any representative of the
      projective class is accepted.

    Returns
    - int: Even shell norm ``N(q)``; ``0`` for the origin.

    Raises
    - ValueError: If ``q`` is not an IVM lattice site (component sum not
      divisible by 4 after normalization; an octahedral or tetrahedral void).
    """
    qn = q.normalize()
    comps = qn.as_tuple()
    s = sum(comps)
    if s % 4 != 0:
        raise ValueError(
            f"{comps} is not an IVM lattice site: component sum {s} is not divisible by 4"
        )
    return sum(abs(c - s // 4) for c in comps)


def shell_sites(k: int) -> List[Quadray]:
    """Enumerate all IVM lattice sites with quadray shell norm ``2k``.

    Sites are scanned over the bounding box ``[0, 2k]^4`` (any shell-``k``
    site has components at most ``2k`` after normalization), kept when their
    component sum is divisible by 4 and their shell norm is ``2k``, and
    returned in deterministic lexicographic order.  The cardinality is the
    cuboctahedral number ``10k**2 + 2`` for ``k >= 1`` and ``1`` for ``k = 0``.

    Parameters
    - k: Non-negative shell index.

    Returns
    - List[Quadray]: Deterministically ordered shell sites.

    Raises
    - ValueError: If ``k`` is negative.
    """
    if k < 0:
        raise ValueError(f"shell index must be non-negative, got {k}")
    target = 2 * k
    sites: List[Quadray] = []
    for a in range(target + 1):
        for b in range(target + 1):
            for c in range(target + 1):
                for d in range(target + 1):
                    q = Quadray(a, b, c, d)
                    if min(q.as_tuple()) != 0:
                        continue  # not the canonical (normalized) representative
                    comps = q.as_tuple()
                    s = sum(comps)
                    if s % 4 != 0:
                        continue
                    if sum(abs(c - s // 4) for c in comps) == target:
                        sites.append(q)
    sites.sort(key=lambda q: q.as_tuple())
    return sites


def ball_sites(radius: int) -> List[Quadray]:
    """Enumerate the IVM lattice ball of the given shell radius, deterministically.

    The ball is the union of shells ``0 .. radius`` (all sites with quadray
    shell norm at most ``2 * radius``), ordered shell by shell and
    lexicographically within each shell — the deterministic site index used
    by :class:`IVMField`.

    Parameters
    - radius: Non-negative ball radius in shell units.

    Returns
    - List[Quadray]: Deterministically ordered ball sites.

    Raises
    - ValueError: If ``radius`` is negative.
    """
    if radius < 0:
        raise ValueError(f"ball radius must be non-negative, got {radius}")
    sites: List[Quadray] = []
    for k in range(radius + 1):
        sites.extend(shell_sites(k))
    return sites


def shell_cardinalities(max_shell: int) -> List[int]:
    """Cardinalities of shells ``0 .. max_shell`` (cuboctahedral numbers).

    Parameters
    - max_shell: Non-negative largest shell index to count.

    Returns
    - List[int]: ``[1, 12, 42, 92, 162, ...]`` — shell ``k >= 1`` has
      ``10k**2 + 2`` sites.

    Raises
    - ValueError: If ``max_shell`` is negative.
    """
    ball = ball_sites(max_shell)
    return [sum(1 for q in ball if quadray_shell_norm(q) == 2 * k) for k in range(max_shell + 1)]


@dataclass
class IVMField:
    """Scalar field over an IVM lattice ball, stored on a deterministic site index.

    Sites are ordered by (shell, lexicographic quadray) — the output order of
    :func:`ball_sites` — and ``values`` holds one float per site in that
    order.  ``site_index`` maps each normalized site ``Quadray`` to its array
    position.  Use :meth:`lattice_ball` to construct instances.
    """

    radius: int
    sites: Tuple[Quadray, ...]
    site_index: Dict[Quadray, int] = field(default_factory=dict, repr=False)
    adjacency: Tuple[Tuple[int, ...], ...] = field(default_factory=tuple, repr=False)
    values: "np.ndarray" = field(default_factory=lambda: np.zeros(0, dtype=float))

    @classmethod
    def lattice_ball(cls, radius: int) -> "IVMField":
        """Build a zero field over the IVM ball of the given shell radius.

        Parameters
        - radius: Non-negative shell radius; the ball holds
          ``1 + sum_{k=1..radius} (10k**2 + 2)`` sites.

        Returns
        - IVMField: Field with all values initialized to zero.

        Raises
        - ValueError: If ``radius`` is negative.
        """
        sites = tuple(ball_sites(radius))
        site_index = {q: i for i, q in enumerate(sites)}
        adjacency = []
        for q in sites:
            neighbors: List[int] = []
            for step in IVM_NEIGHBOR_STEPS:
                j = site_index.get(q.add(step).normalize())
                if j is not None:
                    neighbors.append(j)
            adjacency.append(tuple(sorted(set(neighbors))))
        values = np.zeros(len(sites), dtype=float)
        return cls(
            radius=radius,
            sites=sites,
            site_index=site_index,
            adjacency=tuple(adjacency),
            values=values,
        )

    def predict(self, site: Quadray) -> float:
        """Return the learned field value at a lattice site.

        Parameters
        - site: Quadray site; normalized internally, so any representative
          of the projective class is accepted.

        Returns
        - float: Field value at the site.

        Raises
        - ValueError: If the site is not inside the field's lattice ball.
        """
        idx = self.site_index.get(site.normalize())
        if idx is None:
            raise ValueError(f"site {site.as_tuple()} is outside the lattice ball of radius {self.radius}")
        return float(self.values[idx])

    def score(self, sites: Sequence[Quadray], values: Sequence[float]) -> float:
        """Return the mean squared error of the field against reference values.

        Parameters
        - sites: Query lattice sites (must lie inside the ball).
        - values: Reference scalar values, one per site.

        Returns
        - float: Mean squared error ``mean((predict(site) - value)**2)``.

        Raises
        - ValueError: If ``sites`` is empty, lengths differ, or a site lies
          outside the ball.
        """
        if len(sites) != len(values):
            raise ValueError("sites and values must have the same length")
        if len(sites) == 0:
            raise ValueError("at least one site is required to score")
        preds = np.array([self.predict(s) for s in sites], dtype=float)
        refs = np.asarray(values, dtype=float)
        return float(np.mean((preds - refs) ** 2))

    def learn(
        self,
        observation_sites: Sequence[Quadray],
        observation_values: Sequence[float],
        lam: float = 1e-2,
        kernel_width: float = 1.5,
    ) -> "IVMField":
        """Fit the field by Laplacian-regularized kernel-weighted least squares.

        Observations ``y_j`` are given at sampled lattice sites ``Omega``.
        Graph distances ``d`` are hop counts under the 12-neighbor IVM
        adjacency restricted to the ball, and ``K(d) = exp(-(d/tau)**2)``
        is the kernel (``tau = kernel_width``).  The estimator minimizes

        ``sum_i c_i (f_i - t_i)**2 + lam * sum_{i~j} (f_i - f_j)**2``

        with a two-regime data term:

        - Observed sites: ``c_i = 1`` and target ``t_i = y_i`` — sampled
          data are fit directly, with no self-smoothing.
        - Unobserved sites: confidence ``c_i = max(K(d_i), floor)`` with
          ``d_i = min_j dist(i, j)`` the hop distance to the nearest
          observation, and Nadaraya–Watson kernel target
          ``t_i = sum_j K(d_ij) y_j / sum_j K(d_ij)``.

        The Laplacian regularizer (``L``, the graph Laplacian) propagates
        field structure from observed sites across the lattice graph.  The
        normal equations ``(C + lam * L) f = C t`` are solved densely with
        ``numpy`` — no ML frameworks.  The fit is fully deterministic;
        callers seed any observation sampling themselves (e.g. with
        ``numpy.random.default_rng(seed)``).

        Parameters
        - observation_sites: Sites carrying observations (inside the ball).
        - observation_values: Observed scalar values, one per site.
        - lam: Non-negative Laplacian regularization strength.
        - kernel_width: Positive kernel width ``tau`` (in graph hops).

        Returns
        - IVMField: Self, with ``values`` replaced by the fitted field.

        Raises
        - ValueError: If ``lam`` is negative, ``kernel_width`` is not
          positive, no observations are given, lengths differ, an
          observation value is not finite, or an observation site lies
          outside the ball.
        """
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if kernel_width <= 0.0:
            raise ValueError(f"kernel_width must be positive, got {kernel_width}")
        obs = list(observation_sites)
        y = np.asarray(list(observation_values), dtype=float)
        if len(obs) == 0:
            raise ValueError("at least one observation is required")
        if y.shape[0] != len(obs):
            raise ValueError("observation_sites and observation_values must have the same length")
        if not np.all(np.isfinite(y)):
            raise ValueError("observation values must be finite")
        obs_idx: List[int] = []
        for s in obs:
            j = self.site_index.get(s.normalize())
            if j is None:
                raise ValueError(f"observation site {s.as_tuple()} is outside the lattice ball of radius {self.radius}")
            obs_idx.append(j)

        n = len(self.sites)
        observed = np.zeros(n, dtype=bool)
        observed[np.asarray(obs_idx, dtype=int)] = True
        d_min = _multi_source_distances(self.adjacency, obs_idx)
        confidence = np.where(
            observed,
            1.0,
            np.maximum(np.exp(-((d_min / kernel_width) ** 2)), _WEIGHT_FLOOR),
        )

        num = np.zeros(n, dtype=float)
        den = np.zeros(n, dtype=float)
        for j, y_j in zip(obs_idx, y):
            d_j = _multi_source_distances(self.adjacency, [j])
            k_row = np.exp(-((d_j / kernel_width) ** 2))
            num += k_row * y_j
            den += k_row
        y_hat = num / den

        lap = self._laplacian()
        target = y_hat.copy()
        target[obs_idx] = y  # observed rows are pinned to their data
        system = np.diag(confidence) + lam * lap
        self.values = np.linalg.solve(system, confidence * target)
        return self

    def _laplacian(self) -> "np.ndarray":
        """Return the symmetric graph Laplacian ``D - A`` of the ball."""
        n = len(self.sites)
        lap = np.zeros((n, n), dtype=float)
        for i, neighbors in enumerate(self.adjacency):
            lap[i, i] = float(len(neighbors))
            for j in neighbors:
                lap[i, j] = -1.0
        return lap


def _multi_source_distances(
    adjacency: Sequence[Sequence[int]],
    sources: Sequence[int],
) -> "np.ndarray":
    """BFS hop distances from every site to its nearest source (-1 if unreachable).

    Parameters
    - adjacency: Neighbor index lists, one per site.
    - sources: Indices of source sites.

    Returns
    - np.ndarray: Integer array of hop distances; unreached sites are -1
      (cannot occur on a connected ball with at least one source).
    """
    n = len(adjacency)
    dist = np.full(n, -1, dtype=np.int64)
    frontier: List[int] = []
    for s in sources:
        if dist[s] < 0:
            dist[s] = 0
            frontier.append(s)
    depth = 0
    while frontier:
        nxt: List[int] = []
        for u in frontier:
            for v in adjacency[u]:
                if dist[v] < 0:
                    dist[v] = depth + 1
                    nxt.append(v)
        frontier = nxt
        depth += 1
    return dist


@dataclass(frozen=True)
class TetrahedronFit:
    """Result of :func:`fit_geometry`.

    Attributes
    - matrix: 3x3 orientation+scale matrix mapping canonical quadray-basis
      tetrahedron vertices into the observed frame.
    - scale: Signed cube root of ``det(matrix)`` (edge-scale of the fit).
    - residual: RMS distance between fitted and observed vertex centroids.
    """

    matrix: "np.ndarray"
    scale: float
    residual: float


def fit_geometry(
    points: "np.ndarray",
    labels: Sequence[int],
    embedding: Sequence[Sequence[float]] = DEFAULT_EMBEDDING,
) -> TetrahedronFit:
    """Least-squares recovery of a tetrahedron's orientation+scale from noisy 3D points.

    The canonical tetrahedron has vertices ``t_i = to_xyz(unit_quadray_i)``
    under the given embedding (columns of the 3x4 quadray basis image).
    Each observed point is a noisy sample of one vertex, indicated by its
    label ``0..3``.  Vertex centroids ``c_i`` are formed, and the linear map
    ``G`` minimizing ``sum_i ||G t_i - c_i||^2`` is solved in closed form by
    least squares, yielding the orientation+scale
    ``G = scale * rotation``.

    Parameters
    - points: Array-like of shape ``(n, 3)``; noisy 3D observations.
    - labels: Integer vertex label in ``{0, 1, 2, 3}`` for each point; all
      four labels must occur.
    - embedding: 3x4 quadray-to-XYZ embedding (default
      ``quadray.DEFAULT_EMBEDDING``).

    Returns
    - TetrahedronFit: Fitted matrix, scale, and centroid residual.

    Raises
    - ValueError: If ``points`` is not a finite ``(n, 3)`` array, labels do
      not match the point count, labels are non-integer, or any of the four
      vertex labels is missing.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("at least one point is required")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite")
    labs = np.asarray(labels, dtype=float)
    if labs.shape != (pts.shape[0],):
        raise ValueError("labels must have the same length as points")
    if not np.all(labs == np.floor(labs)):
        raise ValueError("labels must be integers in {0, 1, 2, 3}")
    lab_int = labs.astype(int)
    if set(lab_int.tolist()) != {0, 1, 2, 3}:
        raise ValueError("labels must cover all four tetrahedron vertices 0, 1, 2, 3")

    unit_sites = (Quadray(1, 0, 0, 0), Quadray(0, 1, 0, 0), Quadray(0, 0, 1, 0), Quadray(0, 0, 0, 1))
    basis = np.array([to_xyz(q, embedding) for q in unit_sites], dtype=float).T  # (3, 4) columns
    centroids = np.stack([pts[lab_int == i].mean(axis=0) for i in range(4)], axis=1)  # (3, 4)
    solution, _, _, _ = np.linalg.lstsq(basis.T, centroids.T, rcond=None)
    matrix = solution.T
    fitted = matrix @ basis
    residual = float(np.sqrt(np.mean(np.sum((fitted - centroids) ** 2, axis=0))))
    det = float(np.linalg.det(matrix))
    scale = math.copysign(abs(det) ** (1.0 / 3.0), det)
    return TetrahedronFit(matrix=matrix, scale=float(scale), residual=residual)
