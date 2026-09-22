"""Mutable fleet state: what each vehicle is driving right now, and how far along it is.

This is the live counterpart of the immutable `encoding.FleetRoute` a solver produces:
the routes plus a per-vehicle progress pointer and the per-vehicle cooldown clock the
re-plan controller reads [SPEC 9.3 point 3 / v4 47]. All times are SIM seconds.

NON-WRITE-BACK [SPEC 8.3] is unaffected: this module holds DECODED routes (report /
deployment artifacts). No key vector is stored here and none is ever derived back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.fleet import route_manager as rm
from backend.optimization.encoding import FleetRoute


@dataclass
class FleetState:
    """Deployed plan + progress. `position[v]` indexes into `routes[v]`."""

    routes: list[np.ndarray]
    assignment: np.ndarray  # customer -> vehicle, as in FleetRoute
    position: np.ndarray  # (M_veh,) int: index of the stop each vehicle is AT
    last_reopt: np.ndarray  # (M_veh,) sim s of each vehicle's last re-plan, -inf = never
    served: set[int] = field(default_factory=set)  # matrix indices already delivered

    @classmethod
    def from_fleet_route(cls, fleet: FleetRoute, t_sim: float = -np.inf) -> FleetState:
        """Deploy a solver result: every vehicle at its depot start, nothing served yet."""
        m_veh = len(fleet.routes)
        return cls(
            routes=[np.asarray(r, dtype=np.int64) for r in fleet.routes],
            assignment=np.asarray(fleet.assignment).copy(),
            position=np.zeros(m_veh, dtype=np.int64),
            last_reopt=np.full(m_veh, t_sim, dtype=float),
        )

    def to_fleet_route(self) -> FleetRoute:
        """Current plan as a `FleetRoute` (travelled prefixes included) for fitness's R'."""
        return FleetRoute([r.copy() for r in self.routes], self.assignment.copy())

    # -- progress -------------------------------------------------------------------
    def advance(self, v: int, stops: int = 1) -> int:
        """Move vehicle `v` forward by `stops` stops, marking each customer served."""
        end = len(self.routes[v]) - 1
        for _ in range(max(0, stops)):
            if self.position[v] >= end:
                break
            self.position[v] += 1
            node = int(self.routes[v][self.position[v]])
            if node != 0:
                self.served.add(node)
        return int(self.position[v])

    # -- the remaining view every Phase 9 computation uses [SPEC 9.3 point 1] --------
    def remaining(self, v: int) -> np.ndarray:
        return rm.remaining_route(self.routes[v], int(self.position[v]))

    def remaining_legs(self, v: int) -> list[tuple[int, int]]:
        return rm.remaining_legs(self.routes[v], int(self.position[v]))

    def remaining_customers(self, v: int) -> np.ndarray:
        return rm.remaining_customers(self.routes[v], int(self.position[v]))

    @property
    def n_vehicles(self) -> int:
        return len(self.routes)

    def vehicles(self) -> range:
        return range(self.n_vehicles)

    # -- re-plan bookkeeping [v4 47] -------------------------------------------------
    def record_replan(self, vehicles, t_sim: float) -> None:
        """Stamp the cooldown clock of every re-planned vehicle."""
        for v in vehicles:
            self.last_reopt[int(v)] = float(t_sim)

    def apply_replan(self, v: int, optimized: np.ndarray, t_sim: float) -> None:
        """Splice a fresh suffix onto the frozen prefix [SPEC 9.3 point 1] and stamp the
        cooldown. The vehicle's position does not move: it is still at the same stop, the
        plan ahead of it changed."""
        travelled = rm.travelled_route(self.routes[v], int(self.position[v]))
        self.routes[v] = rm.splice(travelled, optimized)
        self.last_reopt[v] = float(t_sim)
