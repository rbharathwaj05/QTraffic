"""Edge-level congestion model: events + background load -> per-edge time factor."""

# STATUS (through Phase 6): contract only. Signatures + docstrings define the formulas;
# every body raises NotImplementedError until its phase lands. Tests for this module skip.

from __future__ import annotations

import numpy as np

from backend.road.path_index import PathIndex
from backend.traffic.events import TrafficEvent


def bpr_factor(
    volume: np.ndarray, capacity: np.ndarray, a: float = 0.15, b: float = 4.0
) -> np.ndarray:
    """Bureau of Public Roads: t/t0 = 1 + a (v/c)^b (spec: congestion model, background)."""
    raise NotImplementedError


def diurnal_volume(t_sim: float, base_volume: np.ndarray) -> np.ndarray:
    """Background volume profile v(t) = base * (1 + 0.5 sin(2 pi (t/86400 - 0.25)))
    scaled to peak at 08:00 and 17:00 (spec: congestion model, time of day)."""
    raise NotImplementedError


def compose_factors(
    background: np.ndarray, active_events: list[TrafficEvent], n_edges: int
) -> np.ndarray:
    """f_e = background_e * prod_{events on e} severity_to_factor(...)
    (spec: factor composition)."""
    raise NotImplementedError


def affected_pairs(events: list[TrafficEvent], index: PathIndex) -> set[tuple[int, int]]:
    """OD pairs whose cached path crosses any edge in `events` (spec: event impact scope)."""
    raise NotImplementedError


def congestion_level(factor: np.ndarray) -> np.ndarray:
    """Discretise f_e into {0: free, 1: moderate (>1.2), 2: heavy (>1.5), 3: blocked (inf)}
    for the frontend heat layer (spec: visualisation)."""
    raise NotImplementedError
