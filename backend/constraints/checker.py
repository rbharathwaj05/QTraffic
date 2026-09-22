"""Constraint checks in FIXED order [SPEC 8.4 -- order is load-bearing, never reorder]:

    1. capacity      (fleet-level, coarse)           [SPEC 7.3 capacity]
    2. coverage      (every customer exactly once)   [SPEC 7.3 coverage]
    3. time windows  (sequence-level, fine)          [SPEC 7.3 A_k = max(e_k, A_prev + tau)]
    4. availability  (depot return <= H_v)           [SPEC 7.3]
    5. depot         (route starts and ends at 0)
    6. connectivity  (every leg routable at the current traffic_version)

Each check yields structured `Violation`s (kind, vehicle, customers, magnitude), never
a bare bool: the repair operator acts on the FIRST violation in this order. Magnitudes
are dimensionless so `ViolationReport.total` is the residual P of F_eval = F + w_p P
[v4 22]; w_p itself is applied in `optimization.fitness.combine`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.constraints.feasibility import Problem, flatten, fleet_arrivals
from backend.optimization.encoding import FleetRoute

ORDER = ("capacity", "coverage", "time_window", "availability", "depot", "connectivity")


@dataclass(frozen=True)
class Violation:
    kind: str  # one of ORDER
    vehicle: int  # -1 when not vehicle-specific (coverage)
    customers: tuple[int, ...]  # matrix indices 1..N involved
    magnitude: float  # dimensionless contribution to P


@dataclass
class ViolationReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def first(self) -> Violation | None:
        """First violation in fixed order -- the one repair acts on [SPEC 8.4]."""
        return self.violations[0] if self.violations else None

    @property
    def total(self) -> float:
        """Residual violation magnitude P consumed by the penalty term [v4 22]."""
        return float(sum(v.magnitude for v in self.violations))


def check_capacity(fleet: FleetRoute, prob: Problem) -> list[Violation]:
    """Per-vehicle excess max(0, load_v - rho_max Q_v) / (rho_max Q_v) [SPEC 7.3]."""
    flat, ptr = flatten(fleet.routes)
    # reduceat sums demand over each CSR segment = total load per vehicle, one call.
    # Depot entries contribute 0 so the [0, ..., 0] wrapping does not matter.
    load = np.add.reduceat(prob.demand[flat], ptr[:-1])
    limit = prob.rho_max * prob.capacity
    excess = np.maximum(0.0, load - limit)
    return [
        Violation(
            "capacity", v, tuple(int(c) for c in fleet.routes[v][1:-1]), float(excess[v] / limit[v])
        )
        for v in np.flatnonzero(excess)
    ]


def check_coverage(fleet: FleetRoute, prob: Problem) -> list[Violation]:
    """Unserved and duplicated customers, magnitude 1 per missing / extra visit
    [SPEC 7.3 coverage]. Holds after the Phase 4 decode; kept as a defensive check."""
    # Visit count per customer id; [1:] drops the depot slot so counts[k] is customer k+1.
    counts = np.bincount(
        np.concatenate([r[1:-1] for r in fleet.routes]).astype(np.int64),
        minlength=prob.n_customers + 1,
    )[1:]
    out = []
    missing = tuple(int(c) for c in np.flatnonzero(counts == 0) + 1)  # +1: back to 1..N
    if missing:
        out.append(Violation("coverage", -1, missing, float(len(missing))))
    dup = np.flatnonzero(counts > 1) + 1
    if dup.size:
        extra = float(counts[dup - 1].sum() - dup.size)
        out.append(Violation("coverage", -1, tuple(int(c) for c in dup), extra))
    return out


def check_time_windows(fleet: FleetRoute, prob: Problem) -> list[Violation]:
    """Late customers per route, magnitude sum(A_k - l_k) / H_v [SPEC 7.3 time window].
    Non-finite arrivals (unroutable leg upstream) are connectivity's business."""
    flat, ptr, A = fleet_arrivals(fleet.routes, prob)
    late = np.maximum(0.0, np.where(np.isfinite(A), A, 0.0) - prob.windows[flat, 1])
    late[flat == 0] = 0.0  # depot ends never late (window [0, inf))
    owner = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))  # vehicle id of each stop
    per_v = np.bincount(owner, weights=late, minlength=len(ptr) - 1)  # total lateness / vehicle
    return [
        Violation(
            "time_window",
            int(v),
            tuple(int(c) for c in flat[(owner == v) & (late > 0)]),
            float(per_v[v] / prob.shift_end[v]),
        )
        for v in np.flatnonzero(per_v)
    ]


def check_availability(fleet: FleetRoute, prob: Problem) -> list[Violation]:
    """Depot return after H_v, magnitude (A_end - H_v) / H_v [SPEC 7.3 availability].
    Non-finite return times are connectivity's business, not reported here."""
    flat, ptr, A = fleet_arrivals(fleet.routes, prob)
    # ptr[v+1] - 1 is the last element (closing depot) of route v: its arrival = return time.
    over = A[ptr[1:] - 1] - prob.shift_end
    return [
        Violation(  # customers = the final stop before returning (ptr[v+1] - 2)
            "availability", int(v), (int(flat[ptr[v + 1] - 2]),), float(over[v] / prob.shift_end[v])
        )
        for v in np.flatnonzero(np.isfinite(over) & (over > 0))
    ]


def check_depot(fleet: FleetRoute, prob: Problem) -> list[Violation]:
    """Every route is [0, ..., 0] with no interior depot visit."""
    return [
        Violation("depot", v, (), 1.0)
        for v, r in enumerate(fleet.routes)
        if len(r) < 2 or r[0] != 0 or r[-1] != 0 or (r[1:-1] == 0).any()
    ]


def check_connectivity(fleet: FleetRoute, prob: Problem) -> list[Violation]:
    """Legs whose effective duration is non-finite (no path in G at this traffic_version),
    magnitude 1 per broken leg; customers = far end of each broken leg."""
    flat, ptr = flatten(fleet.routes)
    # Every consecutive pair in the flat array, including the fake "end of route v ->
    # start of route v+1" joins, which are masked out next.
    leg_ok = np.isfinite(prob.duration[flat[:-1], flat[1:]])
    leg_ok[ptr[1:-1] - 1] = True  # joins between consecutive routes are not legs
    if leg_ok.all():
        return []
    owner = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))[1:]  # vehicle of leg's far end
    bad = np.flatnonzero(~leg_ok)
    return [
        Violation(
            "connectivity", int(v), tuple(int(c) for c in flat[bad[owner[bad] == v] + 1]), float(k)
        )
        for v, k in zip(*np.unique(owner[bad], return_counts=True))
    ]


# Tuple order == ORDER; check_all relies on this to emit violations in spec order.
CHECKS = (
    check_capacity,
    check_coverage,
    check_time_windows,
    check_availability,
    check_depot,
    check_connectivity,
)


def check_all(fleet: FleetRoute, prob: Problem) -> ViolationReport:
    """Run every check in `ORDER` and bundle the results [SPEC 8.4]."""
    return ViolationReport([v for chk in CHECKS for v in chk(fleet, prob)])
