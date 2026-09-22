"""Greedy repair loop that moves violating customers to feasible positions."""

from __future__ import annotations

import numpy as np

from backend.constraints.checker import ViolationReport


def repair(
    routes: list[np.ndarray],
    demand: np.ndarray,
    capacity: np.ndarray,
    duration: np.ndarray,
    service_time: np.ndarray,
    windows: np.ndarray,
    t_start: np.ndarray,
    rho_max: float,
    max_iterations: int,
) -> tuple[list[np.ndarray], ViolationReport]:
    """Iteratively (<= `max_iterations`, spec default 8) eject the customer with the
    largest violation contribution and re-insert it at the cheapest feasible
    (route, pos) via `feasibility.can_insert`; stop early when the report is clean.
    Residual violations are returned for penalty weighting by w_p (spec: repair).
    """
    raise NotImplementedError


def worst_violator(report: ViolationReport, routes: list[np.ndarray]) -> tuple[int, int]:
    """(route_idx, position) of the customer contributing most to `report.total`
    (spec: repair, ejection choice)."""
    raise NotImplementedError


def cheapest_feasible_insertion(
    routes: list[np.ndarray],
    customer: int,
    duration: np.ndarray,
    service_time: np.ndarray,
    windows: np.ndarray,
    demand: np.ndarray,
    capacity: np.ndarray,
    rho_max: float,
    t_start: np.ndarray,
) -> tuple[int, int] | None:
    """argmin over feasible (route, pos) of insertion cost
    c_{prev,cust} + c_{cust,next} - c_{prev,next}; None if nowhere feasible
    (spec: repair, re-insertion)."""
    raise NotImplementedError


def residual_penalty(report: ViolationReport, w_p: float) -> float:
    """w_p * report.total, added to the fitness when repair cannot fully fix a particle
    (spec: repair, residual penalty)."""
    raise NotImplementedError
