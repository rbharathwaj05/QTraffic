"""Vehicle entity: capacity Q_v, shift end H_v, and the mutable plan slice that
`route_manager` / `state` update as the simulation advances."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Status = Literal["idle", "en_route", "servicing", "returning", "done"]


@dataclass
class Vehicle:
    vehicle_id: str
    capacity: int  # Q_v
    current_node: int  # OSM node id; depot at scenario start
    shift_end: float  # H_v, sim-clock s
    status: Status = "idle"
    assigned_customers: list[str] = field(default_factory=list)
    remaining_route: list[str] = field(default_factory=list)  # customer ids, in order

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Vehicle:
        return cls(**d)
