"""Time-dependent cost matrix between depot + customers, updated by traffic factors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.road.osrm_client import OSRMClient
from backend.road.path_index import PathIndex


@dataclass
class CostMatrix:
    duration_s: np.ndarray  # (n, n) free-flow durations from OSRM
    distance_m: np.ndarray  # (n, n)
    factor: np.ndarray  # (n, n) current traffic multiplier, init ones


def build_cost_matrix(points: list[tuple[float, float]], client: OSRMClient) -> CostMatrix:
    """Fill `duration_s`/`distance_m` from one OSRM table call over `points`
    (index 0 = depot) and set `factor` to ones (spec: cost matrix construction).
    Diagonal is forced to 0; an unroutable pair (NaN from OSRM) raises, since a
    disconnected customer is unrepairable downstream (spec §7.3 continuity)."""
    dur, dist = client.table(points)
    np.fill_diagonal(dur, 0.0)
    np.fill_diagonal(dist, 0.0)
    if np.isnan(dur).any():
        bad = np.argwhere(np.isnan(dur))[:5].tolist()
        raise ValueError(f"OSRM table has unroutable pairs, e.g. {bad}")
    return CostMatrix(dur, dist, np.ones_like(dur))


def update_factors(matrix: CostMatrix, edge_factor: np.ndarray, index: PathIndex) -> np.ndarray:
    """Recompute `matrix.factor[i, j]` = sum_e(t0_e * f_e) / sum_e(t0_e) over edges e on
    path(i, j), where f_e is the per-edge congestion factor from `traffic.congestion`
    (spec: dynamic cost update). Returns the (i, j) pairs whose factor changed as an
    int array of shape (k, 2).

    t0_e comes from `index.edge_t0`; when absent every edge weighs 1 (plain mean).
    Pairs whose path maps to no known edge keep factor 1.
    """
    # [SPEC dynamic cost update] path factor = t0-weighted mean of edge factors
    pairs = np.array(list(index.pair_to_edges), dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    lengths = np.fromiter((len(e) for e in index.pair_to_edges.values()), np.int64, len(pairs))
    flat = np.concatenate(list(index.pair_to_edges.values())).astype(np.int64)
    seg = np.repeat(np.arange(len(pairs)), lengths)
    f = np.asarray(edge_factor, dtype=np.float64)
    t0 = np.ones(len(f)) if index.edge_t0 is None else index.edge_t0
    num = np.bincount(seg, weights=t0[flat] * f[flat], minlength=len(pairs))
    den = np.bincount(seg, weights=t0[flat], minlength=len(pairs))
    new = np.divide(num, den, out=np.ones(len(pairs)), where=den > 0)
    old = matrix.factor[pairs[:, 0], pairs[:, 1]]
    changed = pairs[new != old]
    matrix.factor[pairs[:, 0], pairs[:, 1]] = new
    return changed


def effective_duration(matrix: CostMatrix) -> np.ndarray:
    """c_ij = duration_s * factor, the matrix consumed by `optimization.fitness`
    (spec: fitness, travel-time term)."""
    # [SPEC fitness, travel-time term] c_ij = d_ij * factor_ij
    return matrix.duration_s * matrix.factor
