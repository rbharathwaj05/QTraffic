"""Ant colony baseline. Constructs routes directly (not via random keys)."""

from __future__ import annotations

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization.fitness import ProblemContext
from backend.optimization.qpso import SwarmResult


class ACO:
    """Max-Min Ant System for CVRPTW (spec: benchmark baselines)."""

    def __init__(
        self,
        cfg: QTrafficConfig,
        ctx: ProblemContext,
        rng: np.random.Generator,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.9,
    ) -> None:
        self.cfg, self.ctx, self.rng = cfg, ctx, rng
        self.alpha, self.beta, self.rho, self.q0 = alpha, beta, rho, q0
        self.tau: np.ndarray | None = None  # (n, n) pheromone

    def initialize(self) -> None:
        """tau = tau_max everywhere; eta_ij = 1 / c_ij (spec: benchmark baselines)."""
        raise NotImplementedError

    def construct(self) -> list[np.ndarray]:
        """One ant builds K routes: next customer j chosen by pseudo-random proportional
        rule argmax tau^alpha eta^beta with prob q0, else roulette; respects capacity
        and time windows via `constraints.feasibility.can_insert`."""
        raise NotImplementedError

    def evaporate_and_deposit(self, best_routes: list[np.ndarray], best_f: float) -> None:
        """tau = (1 - rho) tau; tau_ij += 1/best_f on best-so-far edges;
        clamp to [tau_min, tau_max]."""
        raise NotImplementedError

    def run(self, T: int | None = None, time_budget_s: float | None = None) -> SwarmResult:
        """cfg.M ants per iteration; same termination contract as `QPSO.run`.
        `gbest` is returned as the encoded particle of the best routes for uniformity."""
        raise NotImplementedError
