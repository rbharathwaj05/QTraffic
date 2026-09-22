"""Advances traffic state over sim time and pushes edge factors into the cost matrix."""

from __future__ import annotations

import numpy as np

from backend.road.cost_matrix import CostMatrix
from backend.road.path_index import PathIndex
from backend.traffic.events import TrafficEvent


class TrafficSimulator:
    """Owns the event queue and current per-edge factor vector (spec: traffic simulation)."""

    def __init__(
        self,
        n_edges: int,
        base_volume: np.ndarray,
        capacity: np.ndarray,
        events: list[TrafficEvent],
        index: PathIndex,
        matrix: CostMatrix,
    ) -> None:
        self.n_edges = n_edges
        self.base_volume = base_volume
        self.capacity = capacity
        self.pending = sorted(events, key=lambda e: e.t_start)
        self.active: list[TrafficEvent] = []
        self.index = index
        self.matrix = matrix
        self.factor = np.ones(n_edges)
        self.t = 0.0

    def inject(self, event: TrafficEvent) -> None:
        """Add a runtime event (e.g. from the API) to `pending` (spec: live event injection)."""
        raise NotImplementedError

    def step(self, dt: float) -> set[tuple[int, int]]:
        """Advance t by dt: activate events with t_start <= t, expire those with t_end <= t,
        recompute `factor` via `congestion.compose_factors`, push into `matrix` through
        `cost_matrix.update_factors`, return affected OD pairs (spec: simulation step)."""
        raise NotImplementedError

    def snapshot(self) -> dict:
        """Serialisable view {t, active_event_ids, level per edge} for the WebSocket feed."""
        raise NotImplementedError
