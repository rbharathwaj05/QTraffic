"""Hard-feasibility predicates shared by the checker and the repair loop [SPEC 7.3].

Index convention (same as `optimization.encoding` routes): matrix index 0 = depot,
customers 1..N. Every per-node array here is (N+1,) so `arr[route]` works directly;
depot rows are demand 0, service 0, window [0, inf).

Sequential recurrences (arrival times, insertion scans) are Numba-jitted; everything
else is NumPy. No networkx / OSRM call exists in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Problem:
    """Frozen per-scenario constraint data at one traffic_version."""

    duration: np.ndarray  # (N+1, N+1) effective travel time c_ij, s; non-finite = no path
    demand: np.ndarray  # (N+1,) q_j, demand[0] = 0
    service: np.ndarray  # (N+1,) s_j, service[0] = 0
    windows: np.ndarray  # (N+1, 2) [e_j, l_j]; depot [0, inf)
    capacity: np.ndarray  # (M_veh,) Q_v
    shift_end: np.ndarray  # (M_veh,) H_v
    t_start: np.ndarray  # (M_veh,) depot departure time per vehicle
    rho_max: float  # max load ratio [SPEC 7.3]

    @property
    def n_customers(self) -> int:
        return len(self.demand) - 1

    @classmethod
    def from_scenario(cls, customers, vehicles, duration: np.ndarray, rho_max: float) -> Problem:
        """Build from `fleet.customer.Customer` / `fleet.vehicle.Vehicle` lists (depot
        prepended at index 0, matching the cost matrix layout)."""
        z = np.zeros(1)  # depot row prepended to every per-node array
        return cls(
            duration=np.asarray(duration, float),
            demand=np.concatenate([z, [c.demand for c in customers]]),
            service=np.concatenate([z, [c.service_time for c in customers]]),
            windows=np.vstack(
                [[0.0, np.inf], [[c.time_window_start, c.time_window_end] for c in customers]]
            ),
            capacity=np.array([v.capacity for v in vehicles], float),
            shift_end=np.array([v.shift_end for v in vehicles], float),
            t_start=np.zeros(len(vehicles)),  # all vehicles leave the depot at t = 0
            rho_max=rho_max,
        )


# -- capacity ---------------------------------------------------------------------
def load_ratio(route: np.ndarray, demand: np.ndarray, capacity: float) -> float:
    """rho = sum(demand[route]) / Q_v [SPEC 7.3 capacity constraint]."""
    return float(demand[route].sum() / capacity)


def is_capacity_feasible(
    route: np.ndarray, demand: np.ndarray, capacity: float, rho_max: float
) -> bool:
    """True iff load_ratio <= rho_max [SPEC 7.3, rho_max = 0.95]."""
    return load_ratio(route, demand, capacity) <= rho_max


# -- time --------------------------------------------------------------------------
@njit(cache=True)  # cache=True: compiled machine code persists across processes
def _arrivals(route, duration, service, tw_start, t_start):
    # [SPEC 7.3] A_k = max(e_k, A_{k-1} + s_{k-1} + c_{k-1,k}); waiting allowed when early
    # Inherently sequential (each A_k depends on A_{k-1}), hence a jitted loop, not NumPy.
    A = np.empty(len(route))
    A[0] = t_start
    for k in range(1, len(route)):
        i, j = route[k - 1], route[k]
        # leave i after service, travel i->j, wait if arriving before j's window opens
        A[k] = max(A[k - 1] + service[i] + duration[i, j], tw_start[j])
    return A


def arrival_times(
    route: np.ndarray,
    duration: np.ndarray,
    service: np.ndarray,
    windows: np.ndarray,
    t_start: float,
) -> np.ndarray:
    """A_k along `route` (depot at both ends), A_0 = t_start [SPEC 7.3 arrival recurrence].
    A[-1] is the depot return time used by the availability check."""
    return _arrivals(np.asarray(route, np.int64), duration, service, windows[:, 0], float(t_start))


def lateness(
    route: np.ndarray,
    duration: np.ndarray,
    service: np.ndarray,
    windows: np.ndarray,
    t_start: float,
) -> np.ndarray:
    """Per-stop max(0, A_k - l_k), 0 at the depot ends [SPEC 7.3 time-window violation].
    Non-finite arrivals (unroutable leg upstream) count 0 here: connectivity owns them."""
    A = arrival_times(route, duration, service, windows, t_start)
    # inf arrival -> treated as 0 here so it does not register as infinite lateness.
    return np.maximum(0.0, np.where(np.isfinite(A), A, 0.0) - windows[route, 1])


def is_time_feasible(
    route: np.ndarray,
    duration: np.ndarray,
    service: np.ndarray,
    windows: np.ndarray,
    t_start: float,
) -> bool:
    """True iff every A_k <= l_k [SPEC 7.3 time-window constraint]."""
    return not lateness(route, duration, service, windows, t_start).any()


def route_end(
    route: np.ndarray,
    duration: np.ndarray,
    service: np.ndarray,
    windows: np.ndarray,
    t_start: float,
) -> float:
    """Depot return time A_last [SPEC 7.3 vehicle availability: route_end <= H_v]."""
    return float(arrival_times(route, duration, service, windows, t_start)[-1])


@njit(cache=True)
def _fleet_arrivals(flat, ptr, duration, service, tw_start, t_start):
    # every route of a fleet in one call: flat = concatenated [0, ..., 0] routes,
    # ptr[v]:ptr[v+1] = route v (CSR); loops over vehicles AND stops inside numba
    A = np.empty(len(flat))
    for v in range(len(ptr) - 1):
        A[ptr[v] : ptr[v + 1]] = _arrivals(
            flat[ptr[v] : ptr[v + 1]], duration, service, tw_start, t_start[v]
        )
    return A


def flatten(routes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(flat, ptr) CSR view of a route list, depot ends included."""
    # Same CSR idea as path_index.csr(): ptr[v]:ptr[v+1] slices route v out of `flat`.
    ptr = np.zeros(len(routes) + 1, np.int64)
    np.cumsum([len(r) for r in routes], out=ptr[1:])
    return np.concatenate(routes).astype(np.int64), ptr


def fleet_arrivals(
    routes: list[np.ndarray], prob: Problem
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(flat, ptr, A): arrival time at every stop of every route in one jitted pass."""
    flat, ptr = flatten(routes)
    A = _fleet_arrivals(flat, ptr, prob.duration, prob.service, prob.windows[:, 0], prob.t_start)
    return flat, ptr, A


# -- insertion --------------------------------------------------------------------
@njit(cache=True)
def _insert_scan(route, cust, duration, service, tw_start, tw_end, t_start, shift_end):
    """(late, cost) per insertion slot p in 1..L-1 (cust placed before route[p]):
    late = total lateness of the new route (inf if a leg is unroutable),
    cost = c_{prev,cust} + c_{cust,next} - c_{prev,next}  [SPEC 8.2 insertion cost]."""
    L = len(route)
    late = np.empty(L - 1)
    cost = np.empty(L - 1)
    new = np.empty(L + 1, np.int64)  # scratch buffer for the candidate route, reused
    for p in range(1, L):  # try every interior slot
        # Build route-with-cust-at-p without allocating: copy prefix, cust, suffix.
        new[:p] = route[:p]
        new[p] = cust
        new[p + 1 :] = route[p:]
        A = _arrivals(new, duration, service, tw_start, t_start)
        tot = 0.0
        for k in range(1, L):  # sum lateness over all customer stops of the new route
            tot += max(0.0, A[k] - tw_end[new[k]])
        tot += max(0.0, A[L] - shift_end)  # availability folded in as depot lateness
        late[p - 1] = tot if np.isfinite(A[L]) else np.inf  # inf: some leg is unroutable
        a, b = route[p - 1], route[p]
        cost[p - 1] = duration[a, cust] + duration[cust, b] - duration[a, b]  # detour cost
    return late, cost


def insertion_scan(
    route: np.ndarray, cust: int, prob: Problem, v: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """(late, cost, cap_ok) over every slot of vehicle `v`'s `route`; slot p means
    'insert before route[p]', p = 1..L-1. `cap_ok` is the scalar capacity test."""
    late, cost = _insert_scan(
        np.asarray(route, np.int64),
        cust,
        prob.duration,
        prob.service,
        prob.windows[:, 0],
        prob.windows[:, 1],
        float(prob.t_start[v]),
        float(prob.shift_end[v]),
    )
    # Capacity does not depend on where in the route the customer goes: one scalar.
    cap_ok = (prob.demand[route].sum() + prob.demand[cust]) <= prob.rho_max * prob.capacity[v]
    return late, cost, bool(cap_ok)


def can_insert(route: np.ndarray, pos: int, cust: int, prob: Problem, v: int) -> bool:
    """Capacity + time-window + availability feasibility of inserting `cust` before
    `route[pos]` on vehicle `v` [SPEC 8.2 feasible insertion test]."""
    late, _, cap_ok = insertion_scan(route, cust, prob, v)
    return cap_ok and late[pos - 1] == 0.0  # slot p is stored at index p - 1
