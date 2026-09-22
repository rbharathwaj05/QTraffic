"""Hard-feasibility predicates used by both the checker and the repair loop."""

from __future__ import annotations

import numpy as np


def load_ratio(route: np.ndarray, demand: np.ndarray, capacity: float) -> float:
    """rho = sum(demand[route]) / capacity (spec: capacity constraint)."""
    raise NotImplementedError


def is_capacity_feasible(
    route: np.ndarray, demand: np.ndarray, capacity: float, rho_max: float
) -> bool:
    """True iff load_ratio <= rho_max (spec: capacity, rho_max = 0.95)."""
    raise NotImplementedError


def arrival_times(
    route: np.ndarray, duration: np.ndarray, service_time: np.ndarray, t_start: float
) -> np.ndarray:
    """a_k = max(a_{k-1} + s_{k-1} + c_{k-1,k}, e_k) along `route` starting from the depot
    at `t_start`; waiting is allowed when early (spec: time-window semantics)."""
    raise NotImplementedError


def is_time_feasible(
    route: np.ndarray,
    duration: np.ndarray,
    service_time: np.ndarray,
    windows: np.ndarray,
    t_start: float,
) -> bool:
    """True iff every a_k <= l_k where windows[:, 1] = l (spec: time-window constraint)."""
    raise NotImplementedError


def can_insert(
    route: np.ndarray,
    pos: int,
    customer: int,
    duration: np.ndarray,
    service_time: np.ndarray,
    windows: np.ndarray,
    demand: np.ndarray,
    capacity: float,
    rho_max: float,
    t_start: float,
) -> bool:
    """Capacity + time-window feasibility of inserting `customer` at `pos` in `route`
    (spec: repair, feasible insertion test)."""
    raise NotImplementedError
