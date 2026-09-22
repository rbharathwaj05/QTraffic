"""Head-to-head comparison harness: EB-QPSO vs QPSO vs PSO vs GA vs ACO vs OR-Tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import QTrafficConfig
from backend.optimization.fitness import ProblemContext


@dataclass
class BenchmarkRow:
    algorithm: str
    scenario: str
    seed: int
    best_fitness: float
    iterations: int
    elapsed_s: float
    feasible: bool


def load_scenario(path: Path) -> ProblemContext:
    """Read a JSON scenario from data/scenarios into a ProblemContext (spec: benchmarks, inputs)."""
    raise NotImplementedError


def run_one(algorithm: str, ctx: ProblemContext, cfg: QTrafficConfig, seed: int) -> BenchmarkRow:
    """Instantiate `algorithm` by name, run with cfg.T_iter_max, record result
    (spec: benchmarks, protocol)."""
    raise NotImplementedError


def ortools_reference(ctx: ProblemContext, time_limit_s: float) -> BenchmarkRow:
    """OR-Tools routing solver on the same instance as an optimality reference
    (spec: benchmarks, reference solution)."""
    raise NotImplementedError


def run_suite(
    algorithms: list[str], scenarios: list[Path], seeds: list[int], cfg: QTrafficConfig
) -> list[BenchmarkRow]:
    """Cartesian product of algorithms x scenarios x seeds (spec: benchmarks, protocol)."""
    raise NotImplementedError


def summarize(rows: list[BenchmarkRow]) -> dict[tuple[str, str], dict[str, float]]:
    """Per (algorithm, scenario): mean, std, min best_fitness, mean elapsed, gap to OR-Tools
    (spec: benchmarks, reporting)."""
    raise NotImplementedError


def wilcoxon(rows: list[BenchmarkRow], a: str, b: str) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test on best_fitness of `a` vs `b` across seeds and
    scenarios; returns {statistic, p_value} (spec: benchmarks, significance)."""
    raise NotImplementedError


def write_csv(rows: list[BenchmarkRow], out: Path) -> None:
    """Dump rows to benchmarks/<name>.csv (spec: benchmarks, reporting)."""
    raise NotImplementedError
