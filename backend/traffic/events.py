"""Traffic event model: what happens, where, when, how bad."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EventKind(str, Enum):
    ACCIDENT = "accident"
    CLOSURE = "closure"
    CONGESTION = "congestion"
    WEATHER = "weather"


@dataclass(frozen=True)
class TrafficEvent:
    id: int
    kind: EventKind
    edge_ids: np.ndarray  # affected internal edge ids
    t_start: float  # sim seconds
    duration_s: float
    severity: float  # in [0, 1]; 1 = fully blocked (spec: event severity)

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration_s


def severity_to_factor(severity: float, kind: EventKind) -> float:
    """Travel-time multiplier f = 1 / (1 - severity) for congestion/accident/weather,
    inf for closure (spec: event -> edge factor)."""
    raise NotImplementedError


def generate_events(
    n_edges: int,
    horizon_s: float,
    rate_per_hour: float,
    rng: np.random.Generator,
    mean_duration_s: float = 900.0,
) -> list[TrafficEvent]:
    """Poisson process with `rate_per_hour`; kind uniform; severity ~ Beta(2, 2);
    duration ~ Exp(mean_duration_s); 1-3 contiguous edges each
    (spec: scenario generation, stochastic events)."""
    raise NotImplementedError


def load_events(path: str) -> list[TrafficEvent]:
    """Read a scripted event list from data/scenarios/*.json (spec: scenario replay)."""
    raise NotImplementedError
