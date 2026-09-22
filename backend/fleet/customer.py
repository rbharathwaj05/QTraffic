"""Customer entity: the demand point every downstream module (fitness, constraints,
route_manager, API) reads. Plain dataclass, JSON via `to_dict` / `from_dict`."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Customer:
    customer_id: str
    lat: float
    lon: float
    node_id: int  # OSM node id in the city graph (snapped, spec v2 doc 6)
    demand: int  # units, against Vehicle.capacity
    service_time: int  # s at the stop
    time_window_start: float  # sim-clock s
    time_window_end: float  # sim-clock s, > start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Customer:
        return cls(**d)
