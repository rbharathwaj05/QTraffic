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
    (index 0 = depot) and set `factor` to ones (spec: cost matrix construction)."""
    raise NotImplementedError


def update_factors(matrix: CostMatrix, edge_factor: np.ndarray, index: PathIndex) -> np.ndarray:
    """Recompute `matrix.factor[i, j]` = sum_e(t0_e * f_e) / sum_e(t0_e) over edges e on
    path(i, j), where f_e is the per-edge congestion factor from `traffic.congestion`
    (spec: dynamic cost update). Returns the (i, j) pairs whose factor changed as an
    int array of shape (k, 2).
    """
    raise NotImplementedError


def effective_duration(matrix: CostMatrix) -> np.ndarray:
    """c_ij = duration_s * factor, the matrix consumed by `optimization.fitness`
    (spec: fitness, travel-time term)."""
    raise NotImplementedError
