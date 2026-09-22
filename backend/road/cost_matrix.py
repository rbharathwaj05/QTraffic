"""Static OD cost matrices + live-inflated travel time, keyed by traffic_version.

[SPEC 10.2] "Precompute: one OSRM /table call. Then: O(P x T_iter x S) pure array
lookups, SP factor gone." Everything here that talks to OSRM runs ONCE per scenario in
`build_cost_matrices`; the optimiser only ever reads `effective_duration(matrix)`.

Layering [SPEC v3 10.3]:
    OSRM /table  -> D[i,j], T_base[i,j]                        (static, frozen)
    PathIndex    -> edges on path(i,j)                          (static, frozen)
    live         -> T[i,j,t] = T_base[i,j] * factor[i,j]        (per traffic_version)

Cache key for the live matrix is (scenario_id, traffic_version, i, j) [v4 42]. Bumping
`traffic_version` never touches D / T_base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from backend.road.osrm_client import OSRMClient
from backend.road.path_index import PathIndex


class TrafficVersion:
    """Monotone integer `traffic_version` per scenario, in Redis when a client is given
    (shared across processes) else in-process [v4 42]."""

    def __init__(self, scenario_id: str, redis: Any | None = None) -> None:
        self.scenario_id = scenario_id
        self._redis = redis
        self._local = 0
        self._key = f"qtraffic:{scenario_id}:traffic_version"

    @property
    def value(self) -> int:
        if self._redis is None:
            return self._local
        return int(self._redis.get(self._key) or 0)

    def bump(self) -> int:
        if self._redis is None:
            self._local += 1
            return self._local
        return int(self._redis.incr(self._key))

    def cache_key(self, i: int, j: int) -> tuple[str, int, int, int]:
        return (self.scenario_id, self.value, i, j)


@dataclass
class CostMatrix:
    duration_s: np.ndarray  # (n, n) T_base: free-flow durations from OSRM, frozen
    distance_m: np.ndarray  # (n, n) D, frozen
    factor: np.ndarray  # (n, n) current traffic multiplier, init ones
    edge_t0: np.ndarray | None = None  # (E,) per-edge free-flow time, weights for inflate
    version: TrafficVersion = field(default_factory=lambda: TrafficVersion("default"))

    # -- persistence: data/scenarios/<id>/matrices.npz ------------------------------
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            duration_s=self.duration_s,
            distance_m=self.distance_m,
            edge_t0=np.empty(0) if self.edge_t0 is None else self.edge_t0,
        )

    @classmethod
    def load(cls, path: Path, version: TrafficVersion | None = None) -> CostMatrix:
        z = np.load(path)
        t0 = z["edge_t0"]
        return cls(
            duration_s=z["duration_s"],
            distance_m=z["distance_m"],
            factor=np.ones_like(z["duration_s"]),
            edge_t0=None if t0.size == 0 else t0,
            version=version or TrafficVersion("default"),
        )


def build_cost_matrices(
    points: list[tuple[float, float]], client: OSRMClient
) -> tuple[np.ndarray, np.ndarray]:
    """(D, T_base), each (N+1, N+1) with depot at index 0, from OSRM /table
    [SPEC 10.2]. Computed ONCE per scenario; raises if any pair is unroutable so no
    NaN ever reaches the optimiser."""
    dur, dist = client.table(points)
    bad = ~np.isfinite(dur) | ~np.isfinite(dist)
    if bad.any():
        raise ValueError(
            f"OSRM table: {int(bad.sum())} unroutable pairs, first {np.argwhere(bad)[0]}"
        )
    return dist, dur


def build_cost_matrix(
    points: list[tuple[float, float]],
    client: OSRMClient,
    edge_t0: np.ndarray | None = None,
    version: TrafficVersion | None = None,
) -> CostMatrix:
    """`build_cost_matrices` wrapped with `factor=ones` (spec: cost matrix construction)."""
    D, T_base = build_cost_matrices(points, client)
    version = version or TrafficVersion("default")
    return CostMatrix(T_base, D, np.ones_like(T_base), edge_t0, version)


def inflate_all(index: PathIndex, edge_t0: np.ndarray, edge_factor: np.ndarray) -> np.ndarray:
    """factor[i, j] = sum_e(t0_e f_e) / sum_e(t0_e) over e on path(i, j), all pairs at
    once via bincount over the index's CSR view; 1.0 for pairs with no edges
    (spec: dynamic cost update). No per-pair Python loop."""
    ptr, flat = index.csr()
    n2 = len(ptr) - 1
    owner = np.repeat(np.arange(n2), np.diff(ptr))
    w = edge_t0[flat]
    num = np.bincount(owner, weights=w * edge_factor[flat], minlength=n2)
    den = np.bincount(owner, weights=w, minlength=n2)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0).reshape(index.n, index.n)


def inflate(
    i: int, j: int, edge_factor: np.ndarray, index: PathIndex, edge_t0: np.ndarray
) -> float:
    """Scalar form of `inflate_all` for one (i, j): T[i,j,t] = T_base[i,j] * inflate(...)
    [SPEC v3 10.3]. Debug / display use; the optimiser reads the cached matrix."""
    e = index.edges_on(i, j)
    if e.size == 0:
        return 1.0
    w = edge_t0[e]
    return float((w * edge_factor[e]).sum() / w.sum())


def update_factors(matrix: CostMatrix, edge_factor: np.ndarray, index: PathIndex) -> np.ndarray:
    """Recompute `matrix.factor[i, j]` = sum_e(t0_e * f_e) / sum_e(t0_e) over edges e on
    path(i, j), where f_e is the per-edge congestion factor from `traffic.congestion`
    (spec: dynamic cost update). Returns the (i, j) pairs whose factor changed as an
    int array of shape (k, 2), and bumps `traffic_version` when k > 0.
    """
    if matrix.edge_t0 is None:
        raise ValueError("CostMatrix.edge_t0 required for inflation")
    # ponytail: full vectorised recompute every call (~ms for N=300); switch to
    # index.invalidate_edge-scoped recompute if profiling ever shows this on the path.
    new = inflate_all(index, matrix.edge_t0, np.asarray(edge_factor, float))
    changed = np.argwhere(~np.isclose(new, matrix.factor))
    if len(changed):
        matrix.factor = new
        matrix.version.bump()
    return changed


def update_factors_for_edges(
    matrix: CostMatrix,
    edge_factor: np.ndarray,
    index: PathIndex,
    edge_ids: Any,
) -> np.ndarray:
    """Scoped counterpart of `update_factors` [SPEC 10.2 / v4 43]: recompute `factor` ONLY
    for the pairs that `PathIndex.invalidate_edge` says depend on the changed edges.

    This is the path Phase 9 uses when a traffic event lands: an event touching k edges
    touches |pairs_using(k)| pairs, typically a small slice of the N^2 matrix, so the full
    `inflate_all` rebuild is skipped entirely. Same return contract as `update_factors`
    (changed (i, j) as an int array, `traffic_version` bumped when anything moved).
    """
    if matrix.edge_t0 is None:
        raise ValueError("CostMatrix.edge_t0 required for inflation")
    pairs = set()
    for e in edge_ids:
        pairs |= index.invalidate_edge(int(e))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)

    ij = np.array(sorted(pairs), dtype=np.int64)
    lists = [index.edges_on(int(i), int(j)) for i, j in ij]
    counts = np.array([len(x) for x in lists], dtype=np.int64)
    flat = np.concatenate(lists) if len(lists) else np.empty(0, dtype=np.int64)
    owner = np.repeat(np.arange(len(ij)), counts)
    w = matrix.edge_t0[flat]
    f = np.asarray(edge_factor, float)
    num = np.bincount(owner, weights=w * f[flat], minlength=len(ij))
    den = np.bincount(owner, weights=w, minlength=len(ij))
    new = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)

    old = matrix.factor[ij[:, 0], ij[:, 1]]
    moved = ~np.isclose(new, old)
    if not moved.any():
        return np.empty((0, 2), dtype=np.int64)
    matrix.factor[ij[moved, 0], ij[moved, 1]] = new[moved]
    matrix.version.bump()
    return ij[moved]


def effective_duration(matrix: CostMatrix) -> np.ndarray:
    """c_ij = duration_s * factor, the matrix consumed by `optimization.fitness`
    (spec: fitness, travel-time term). Pure array op, zero OSRM / graph calls."""
    return matrix.duration_s * matrix.factor
