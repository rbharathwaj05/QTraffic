"""Genetic algorithm baseline operating on the same random-key encoding."""

# STATUS (through Phase 6): contract only. Signatures + docstrings define the formulas;
# every body raises NotImplementedError until its phase lands. Tests for this module skip.

from __future__ import annotations

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization.qpso import FitnessFn, SwarmResult


class GA:
    """Real-coded GA (spec: benchmark baselines)."""

    def __init__(
        self,
        cfg: QTrafficConfig,
        fitness_fn: FitnessFn,
        dim: int,
        rng: np.random.Generator,
        p_crossover: float = 0.9,
        p_mutation: float = 0.05,
        tournament_k: int = 3,
    ) -> None:
        self.cfg, self.fitness_fn, self.dim, self.rng = cfg, fitness_fn, dim, rng
        self.p_crossover, self.p_mutation, self.tournament_k = p_crossover, p_mutation, tournament_k

    def initialize(self) -> None:
        """Uniform population of size cfg.M in [0,1]^D, evaluate."""
        raise NotImplementedError

    def select(self) -> np.ndarray:
        """Tournament selection of size k, returns parent indices (M,)."""
        raise NotImplementedError

    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uniform crossover with probability p_crossover, else copies."""
        raise NotImplementedError

    def mutate(self, x: np.ndarray) -> np.ndarray:
        """Per-gene resample U(0,1) with probability p_mutation."""
        raise NotImplementedError

    def step(self) -> float:
        """One generation with elitism of 1; returns best fitness."""
        raise NotImplementedError

    def run(self, T: int | None = None, time_budget_s: float | None = None) -> SwarmResult:
        """Same termination contract as `QPSO.run`."""
        raise NotImplementedError
