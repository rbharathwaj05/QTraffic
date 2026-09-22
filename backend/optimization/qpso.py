"""Quantum-behaved PSO (Sun et al.) baseline. EB-QPSO subclasses this."""

# STATUS (through Phase 6): contract only. Signatures + docstrings define the formulas;
# every body raises NotImplementedError until its phase lands. Tests for this module skip.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from backend.config import QTrafficConfig

FitnessFn = Callable[[np.ndarray], np.ndarray]  # (M, D) -> (M,)


@dataclass
class SwarmResult:
    gbest: np.ndarray
    gbest_fitness: float
    history: np.ndarray  # gbest_fitness per iteration
    iterations: int
    elapsed_s: float
    stopped_by: str  # "budget" | "stagnation" | "time"


class QPSO:
    """Canonical QPSO (spec: QPSO core)."""

    def __init__(
        self, cfg: QTrafficConfig, fitness_fn: FitnessFn, dim: int, rng: np.random.Generator
    ) -> None:
        self.cfg = cfg
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.rng = rng
        self.X: np.ndarray | None = None  # (M, D)
        self.pbest: np.ndarray | None = None  # (M, D)
        self.pbest_f: np.ndarray | None = None  # (M,)
        self.gbest: np.ndarray | None = None
        self.gbest_f: float = np.inf

    def initialize(self, seed_particles: np.ndarray | None = None) -> None:
        """Fill X with M uniform particles; if `seed_particles` given, they replace the
        first rows (spec: initialisation / warm start hook). Evaluate, set pbest/gbest."""
        raise NotImplementedError

    def alpha(self, t: int, T: int) -> float:
        """Linear contraction-expansion schedule
        alpha(t) = alpha_max - (alpha_max - alpha_min) * t / T (spec: QPSO, alpha schedule)."""
        raise NotImplementedError

    def mean_best(self) -> np.ndarray:
        """mbest = (1/M) sum_i pbest_i (spec: QPSO, mean best position)."""
        raise NotImplementedError

    def update_positions(self, alpha: float) -> None:
        """For each particle i and dimension d:
        phi ~ U(0,1); p = phi * pbest_id + (1-phi) * gbest_d; u ~ U(0,1)
        x_id = p +/- alpha * |mbest_d - x_id| * ln(1/u), sign chosen with prob 0.5
        (spec: QPSO, position update). Then clip to [0, 1]."""
        raise NotImplementedError

    def step(self, t: int, T: int) -> float:
        """One iteration: update positions, evaluate, refresh pbest/gbest; return gbest_f."""
        raise NotImplementedError

    def stagnated(self, history: list[float]) -> bool:
        """True if relative improvement over the last `cfg.patience` iterations
        < `cfg.epsilon_stag` (spec: stagnation detection)."""
        raise NotImplementedError

    def run(self, T: int | None = None, time_budget_s: float | None = None) -> SwarmResult:
        """Iterate until T (default cfg.T_iter_max), wall-clock `time_budget_s`, or
        stagnation after at least cfg.T_iter_min iterations (spec: termination)."""
        raise NotImplementedError
