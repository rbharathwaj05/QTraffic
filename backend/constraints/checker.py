"""Quantify constraint violations for a decoded solution. Returns magnitudes, not booleans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ViolationReport:
    capacity_excess: np.ndarray  # per vehicle, units of demand above rho_max*capacity
    time_window_late: np.ndarray  # per customer, seconds after l_k (0 if on time)
    unserved: np.ndarray  # customer ids not present in any route
    duplicated: np.ndarray  # customer ids present more than once

    @property
    def total(self) -> float:
        """Scalar violation magnitude V consumed by the penalty term (spec: penalty term)."""
        raise NotImplementedError


def check_capacity(
    routes: list[np.ndarray], demand: np.ndarray, capacity: np.ndarray, rho_max: float
) -> np.ndarray:
    """Per-vehicle excess max(0, load_k - rho_max * Q_k) (spec: capacity violation)."""
    raise NotImplementedError


def check_time_windows(
    routes: list[np.ndarray],
    duration: np.ndarray,
    service_time: np.ndarray,
    windows: np.ndarray,
    t_start: np.ndarray,
) -> np.ndarray:
    """Per-customer lateness max(0, a_k - l_k) (spec: time-window violation)."""
    raise NotImplementedError


def check_coverage(routes: list[np.ndarray], n_customers: int) -> tuple[np.ndarray, np.ndarray]:
    """(unserved, duplicated) customer ids (spec: assignment validity)."""
    raise NotImplementedError


def check_all(
    routes: list[np.ndarray],
    demand: np.ndarray,
    capacity: np.ndarray,
    duration: np.ndarray,
    service_time: np.ndarray,
    windows: np.ndarray,
    t_start: np.ndarray,
    rho_max: float,
) -> ViolationReport:
    """Run every checker and bundle results (spec: constraint handling)."""
    raise NotImplementedError
