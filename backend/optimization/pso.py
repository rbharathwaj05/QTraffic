"""Classical inertia-weight PSO baseline for benchmarking against QPSO/EB-QPSO."""

from __future__ import annotations

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization.qpso import FitnessFn, SwarmResult


class PSO:
    """Standard PSO (spec: benchmark baselines)."""

    def __init__(
        self,
        cfg: QTrafficConfig,
        fitness_fn: FitnessFn,
        dim: int,
        rng: np.random.Generator,
        w: float = 0.7298,
        c1: float = 1.4962,
        c2: float = 1.4962,
    ) -> None:
        self.cfg, self.fitness_fn, self.dim, self.rng = cfg, fitness_fn, dim, rng
        self.w, self.c1, self.c2 = w, c1, c2  # Clerc constriction defaults

    def initialize(self) -> None:
        """Uniform X in [0,1]^D, V = 0, evaluate, set pbest/gbest (spec: benchmark baselines)."""
        raise NotImplementedError

    def step(self) -> float:
        """v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x); x = clip(x + v); evaluate."""
        raise NotImplementedError

    def run(self, T: int | None = None, time_budget_s: float | None = None) -> SwarmResult:
        """Same termination contract as `QPSO.run` so benchmark.py can treat them alike."""
        raise NotImplementedError
