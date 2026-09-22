"""Weighted multi-objective fitness. Lower is better."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import QTrafficConfig


@dataclass
class ProblemContext:
    """Static-per-call inputs the fitness needs. Built once per optimisation call."""

    duration: np.ndarray  # (n, n) effective travel time, index 0 = depot
    distance: np.ndarray  # (n, n) metres
    demand: np.ndarray  # (N+1,), demand[0] = 0
    capacity: np.ndarray  # (K,)
    service_time: np.ndarray  # (N+1,) seconds
    windows: np.ndarray  # (N+1, 2) [e, l] seconds of sim time
    t_start: np.ndarray  # (K,) vehicle availability time
    incumbent: list[np.ndarray] | None  # previous plan for route-change penalty
    norm: dict[str, float]  # normalisers for each term, see `normalisers`


def total_time(routes: list[np.ndarray], duration: np.ndarray, service_time: np.ndarray) -> float:
    """T = sum_k sum_{(i,j) in route_k} c_ij + sum_i s_i (spec: fitness, time term)."""
    raise NotImplementedError


def total_distance(routes: list[np.ndarray], distance: np.ndarray) -> float:
    """D = sum_k sum_{(i,j) in route_k} d_ij (spec: fitness, distance term)."""
    raise NotImplementedError


def route_change(
    new: list[np.ndarray], old: list[np.ndarray], n_customers: int, eta_a: float, eta_o: float
) -> float:
    """R = eta_a * (#customers whose vehicle changed / N)
         + eta_o * (#adjacent pairs in `old` not preserved in `new` / N)
    (spec: route-change penalty, assignment vs ordering split)."""
    raise NotImplementedError


def normalisers(ctx: ProblemContext) -> dict[str, float]:
    """Scale factors so each term is O(1): T_ref, D_ref from a nearest-neighbour
    construction, V_ref = total demand (spec: fitness normalisation)."""
    raise NotImplementedError


def fitness(x: np.ndarray, ctx: ProblemContext, cfg: QTrafficConfig) -> float:
    """F(x) = w_t T/T_ref + w_d D/D_ref + w_c V/V_ref + w_r R, after decode -> repair
    (spec: fitness function). V is the residual violation after repair."""
    raise NotImplementedError


def fitness_batch(X: np.ndarray, ctx: ProblemContext, cfg: QTrafficConfig) -> np.ndarray:
    """Vectorised `fitness` over swarm matrix X of shape (M, D) (spec: performance)."""
    raise NotImplementedError
