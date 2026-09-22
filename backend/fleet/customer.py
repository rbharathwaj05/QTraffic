"""Customer entity: the demand point every downstream module (fitness, constraints,
route_manager, API) reads. Plain dataclass, JSON via `to_dict` / `from_dict`."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)  # frozen: a customer never changes once a scenario is generated
class Customer:
    customer_id: str  # "c000", "c001", ... as written by fleet.scenario
    lat: float  # coordinates of the snapped graph node, not the original random point
    lon: float
    node_id: int  # OSM node id in the city graph (snapped, spec v2 doc 6)
    demand: int  # units, against Vehicle.capacity
    service_time: int  # s at the stop
    time_window_start: float  # sim-clock s
    time_window_end: float  # sim-clock s, > start

    # JSON round-trip helpers; field names are the JSON keys in customers.json.
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Customer:
        return cls(**d)
