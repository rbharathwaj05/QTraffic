"""Traffic event model: what happens, where, when, how bad."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
        """SIM seconds. Every event time in this system is sim time [doc2 23]."""
        return self.t_start + self.duration_s

    def active_at(self, t_sim: float) -> bool:
        """[t_start, t_end): active from its start until, not including, its end."""
        return self.t_start <= t_sim < self.t_end


def severity_to_factor(severity: float, kind: EventKind) -> float:
    """Travel-time multiplier f = 1 / (1 - severity) for congestion/accident/weather,
    inf for closure (spec: event -> edge factor).

    This is `congestion.rho_to_factor` with severity read as the event's rho: a 0.70
    congestion event means rho_e = 0.70, i.e. the edge runs at 30 % of normal speed and
    takes 1/0.30 = 3.33x as long [SPEC 7.2]. severity 1 (or any CLOSURE) is infinite
    time, the encoding of an unusable edge -- not rho = 1, which the model excludes.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    if kind is EventKind.CLOSURE or severity >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - severity)


def generate_events(
    n_edges: int,
    horizon_s: float,
    rate_per_hour: float,
    rng: np.random.Generator,
    mean_duration_s: float = 900.0,
) -> list[TrafficEvent]:
    """Poisson process with `rate_per_hour`; kind uniform; severity ~ Beta(2, 2);
    duration ~ Exp(mean_duration_s); 1-3 contiguous edges each
    (spec: scenario generation, stochastic events).

    Edge ids are drawn as a contiguous run only in id space -- the graph adjacency is not
    consulted, so "contiguous" here means "a short run of neighbouring edge ids", which is
    all a synthetic scenario needs. Fully seeded through `rng` [CLAUDE.md invariant].
    """
    n = int(rng.poisson(rate_per_hour * horizon_s / 3600.0))
    if n == 0 or n_edges == 0:
        return []
    kinds = list(EventKind)
    starts = np.sort(rng.uniform(0.0, horizon_s, n))
    runs = rng.integers(1, 4, n)
    firsts = rng.integers(0, n_edges, n)
    severities = rng.beta(2.0, 2.0, n)
    durations = rng.exponential(mean_duration_s, n)
    picks = rng.integers(0, len(kinds), n)
    return [
        TrafficEvent(
            id=i,
            kind=kinds[picks[i]],
            edge_ids=np.arange(firsts[i], min(firsts[i] + runs[i], n_edges), dtype=np.int64),
            t_start=float(starts[i]),
            duration_s=float(durations[i]),
            severity=float(severities[i]),
        )
        for i in range(n)
    ]


def load_events(path: str) -> list[TrafficEvent]:
    """Read a scripted event list from data/scenarios/*.json (spec: scenario replay).

    One JSON object per event: {id, kind, edge_ids, t_start, duration_s, severity}.
    """
    raw = json.loads(Path(path).read_text())
    return [
        TrafficEvent(
            id=int(d["id"]),
            kind=EventKind(d["kind"]),
            edge_ids=np.asarray(d["edge_ids"], dtype=np.int64),
            t_start=float(d["t_start"]),
            duration_s=float(d["duration_s"]),
            severity=float(d["severity"]),
        )
        for d in raw
    ]


def save_events(events: list[TrafficEvent], path: str) -> None:
    """Inverse of `load_events`, so a generated event script is replayable."""
    Path(path).write_text(
        json.dumps(
            [
                {
                    "id": e.id,
                    "kind": e.kind.value,
                    "edge_ids": [int(x) for x in np.asarray(e.edge_ids).ravel()],
                    "t_start": e.t_start,
                    "duration_s": e.duration_s,
                    "severity": e.severity,
                }
                for e in events
            ],
            indent=2,
        )
    )
