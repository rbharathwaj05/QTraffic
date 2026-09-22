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
        self._redis = redis  # duck-typed: needs .get(key) and .incr(key) only
        self._local = 0  # in-process counter used when no Redis client is given
        self._key = f"qtraffic:{scenario_id}:traffic_version"

    @property
    def value(self) -> int:
        if self._redis is None:
            return self._local
        return int(self._redis.get(self._key) or 0)  # missing key -> version 0

    def bump(self) -> int:
        """Increment and return the new version (atomic INCR when backed by Redis)."""
        if self._redis is None:
            self._local += 1
            return self._local
        return int(self._redis.incr(self._key))

    def cache_key(self, i: int, j: int) -> tuple[str, int, int, int]:
        # Any cache keyed by this tuple is automatically stale after a bump.
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
        # Only the frozen parts are persisted; `factor` is live state and is reset to
        # ones on load (a fresh process starts at free flow).
        np.savez_compressed(
            path,
            duration_s=self.duration_s,
            distance_m=self.distance_m,
            edge_t0=np.empty(0) if self.edge_t0 is None else self.edge_t0,  # npz needs an array
        )

    @classmethod
    def load(cls, path: Path, version: TrafficVersion | None = None) -> CostMatrix:
        z = np.load(path)
        t0 = z["edge_t0"]
        return cls(
            duration_s=z["duration_s"],
            distance_m=z["distance_m"],
            factor=np.ones_like(z["duration_s"]),
            edge_t0=None if t0.size == 0 else t0,  # empty sentinel -> None
            version=version or TrafficVersion("default"),
        )


def build_cost_matrices(
    points: list[tuple[float, float]], client: OSRMClient
) -> tuple[np.ndarray, np.ndarray]:
    """(D, T_base), each (N+1, N+1) with depot at index 0, from OSRM /table
    [SPEC 10.2]. Computed ONCE per scenario; raises if any pair is unroutable so no
    NaN ever reaches the optimiser."""
    dur, dist = client.table(points)  # the one and only /table call for this scenario
    # OSRM returns null (-> nan) for pairs it cannot route; fail loudly here rather than
    # let a nan poison every fitness value later.
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
    n2 = len(ptr) - 1  # number of (i, j) pairs = n*n
    # owner[k] = which pair the k-th flat edge belongs to (segment id from CSR ptr).
    owner = np.repeat(np.arange(n2), np.diff(ptr))
    w = edge_t0[flat]  # weight each edge by its free-flow time
    # Weighted mean of edge factors per pair: two bincounts replace a per-pair loop.
    num = np.bincount(owner, weights=w * edge_factor[flat], minlength=n2)
    den = np.bincount(owner, weights=w, minlength=n2)
    # Pairs with no edges (diagonal, empty paths) have den == 0 -> factor 1.0.
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
    changed = np.argwhere(~np.isclose(new, matrix.factor))  # (k, 2) rows of [i, j]
    if len(changed):
        # Only swap the matrix and bump the version when something actually moved, so
        # no-op traffic ticks do not invalidate caches.
        matrix.factor = new
        matrix.version.bump()
    return changed


def effective_duration(matrix: CostMatrix) -> np.ndarray:
    """c_ij = duration_s * factor, the matrix consumed by `optimization.fitness`
    (spec: fitness, travel-time term). Pure array op, zero OSRM / graph calls."""
    return matrix.duration_s * matrix.factor
