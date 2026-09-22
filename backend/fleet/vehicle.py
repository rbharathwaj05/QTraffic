"""Vehicle entity: capacity Q_v, shift end H_v, and the mutable plan slice that
`route_manager` / `state` update as the simulation advances."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

# Lifecycle a vehicle moves through during a simulated shift (driven by later phases).
Status = Literal["idle", "en_route", "servicing", "returning", "done"]


@dataclass  # NOT frozen: status / position / route are updated as the sim advances
class Vehicle:
    vehicle_id: str  # "v000", "v001", ...
    capacity: int  # Q_v
    current_node: int  # OSM node id; depot at scenario start
    shift_end: float  # H_v, sim-clock s
    status: Status = "idle"
    assigned_customers: list[str] = field(default_factory=list)
    remaining_route: list[str] = field(default_factory=list)  # customer ids, in order

    # JSON round-trip helpers; `field(default_factory=list)` keeps lists per-instance.
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Vehicle:
        return cls(**d)
