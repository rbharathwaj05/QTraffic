"""Advances traffic state over sim time and pushes edge factors into the cost matrix.

One `step(dt)` is: activate/expire events -> recompose per-edge factors -> push ONLY the
pairs that the changed edges touch into the cost matrix (`update_factors_for_edges`, which
goes through `PathIndex.invalidate_edge`; the full N^2 matrix is never rebuilt) -> return
the affected pairs.

DEBOUNCE [SPEC 9.5, first step of the loop]: events that arrive close together are batched
so N simultaneous edge closures cause ONE controller pass, not N. The window is
`cfg.debounce_s` in SIM seconds -- sim, so that a 60x demo debounces the same events the
same way a 1x run does [doc2 23]. `take_batch()` hands the batch to the controller once
the window has closed; until then `pending_batch` accumulates.
"""

from __future__ import annotations

import numpy as np

from backend.config import QTrafficConfig
from backend.road.cost_matrix import CostMatrix, update_factors_for_edges
from backend.road.path_index import PathIndex
from backend.traffic import congestion
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
        cfg: QTrafficConfig | None = None,
    ) -> None:
        self.n_edges = n_edges
        self.base_volume = base_volume
        self.capacity = capacity
        self.pending = sorted(events, key=lambda e: e.t_start)
        self.active: list[TrafficEvent] = []
        self.index = index
        self.matrix = matrix
        self.cfg = cfg or QTrafficConfig()
        self.factor = np.ones(n_edges)
        self.t = 0.0
        # debounce [SPEC 9.5]: edges whose factor moved, and when the window closes
        self.pending_batch: set[int] = set()
        self.batch_deadline: float | None = None

    def inject(self, event: TrafficEvent) -> None:
        """Add a runtime event (e.g. from the API) to `pending` (spec: live event injection).

        An event whose start time is in the past is treated as "now": manual injection from
        the UI means immediately, not retroactively.
        """
        if event.t_start < self.t:
            event = TrafficEvent(
                id=event.id,
                kind=event.kind,
                edge_ids=event.edge_ids,
                t_start=self.t,
                duration_s=event.duration_s,
                severity=event.severity,
            )
        self.pending.append(event)
        self.pending.sort(key=lambda e: e.t_start)

    def step(self, dt: float, fleet_flow: np.ndarray | None = None) -> np.ndarray:
        """Advance sim time by `dt`, recompute factors, push the affected pairs only.

        Returns the changed (i, j) pairs as an int array of shape (k, 2) -- the same
        contract as `cost_matrix.update_factors`, and empty when nothing moved.
        """
        self.t += float(dt)
        started = [e for e in self.pending if e.t_start <= self.t]
        self.pending = [e for e in self.pending if e.t_start > self.t]
        self.active.extend(started)
        expired = [e for e in self.active if e.t_end <= self.t]
        self.active = [e for e in self.active if e.t_end > self.t]

        background = congestion.background_factors(
            self.t, self.base_volume, self.capacity, self.cfg, fleet_flow
        )
        new_factor = congestion.compose_factors(
            background, self.active, self.n_edges, self.cfg.rho_congestion_max
        )
        moved = np.flatnonzero(~np.isclose(new_factor, self.factor, equal_nan=True))
        self.factor = new_factor
        if moved.size == 0:
            return np.empty((0, 2), dtype=np.int64)

        self._arm_batch(moved, started, expired)
        # [SPEC 10.2] scoped update: only the pairs whose cached path uses a changed edge
        return update_factors_for_edges(
            self.matrix, self.factor, self.index, moved, self.cfg.scoped_update_max_fraction
        )

    # -- 9.8 debounce / batching [SPEC 9.5] -------------------------------------------
    def _arm_batch(self, moved: np.ndarray, started: list, expired: list) -> None:
        """Collect changed edges; the window opens on the first change and runs for
        `cfg.debounce_s` SIM seconds, so a burst of simultaneous events is one batch."""
        if not started and not expired:
            return  # background drift is not an event; it never opens a window
        self.pending_batch |= {int(e) for e in moved}
        if self.batch_deadline is None:
            self.batch_deadline = self.t + self.cfg.debounce_s

    def batch_ready(self) -> bool:
        """True once the debounce window has closed and there is something to report."""
        return (
            bool(self.pending_batch)
            and self.batch_deadline is not None
            and (self.t >= self.batch_deadline)
        )

    def take_batch(self, force: bool = False) -> list[int]:
        """Hand the batched edge ids to the controller and reset the window. Returns [] while
        the window is still open (unless `force`), which is what keeps N events -> 1 pass."""
        if not (force or self.batch_ready()):
            return []
        out = sorted(self.pending_batch)
        self.pending_batch.clear()
        self.batch_deadline = None
        return out

    def snapshot(self) -> dict:
        """Serialisable view {t, active_event_ids, level per edge} for the WebSocket feed."""
        return {
            "t": self.t,
            "active_event_ids": [e.id for e in self.active],
            "level": congestion.congestion_level(self.factor).tolist(),
            "traffic_version": self.matrix.version.value,
        }
