"""Head-to-head comparison harness: EB-QPSO vs QPSO vs PSO vs GA vs ACO vs OR-Tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.feasibility import Problem
from backend.fleet import scenario as scenario_io
from backend.optimization.fitness import (
    Bounds,
    ProblemContext,
    TrafficState,
    compute_normalization_bounds,
)
from backend.road.build_scenario import SCENARIO_DIR
from backend.road.cost_matrix import CostMatrix
from backend.road.geometry import haversine_matrix


@dataclass
class BenchmarkRow:
    algorithm: str
    scenario: str
    seed: int
    best_fitness: float
    iterations: int
    elapsed_s: float
    feasible: bool


def traffic_state(name: str, lat: np.ndarray, lon: np.ndarray, cfg: QTrafficConfig) -> TrafficState:
    """Matrices for scenario `name`: the persisted OSRM `matrices.npz` when
    `road.build_scenario` has been run, else a haversine fallback with
    `cfg.fallback_speed_mps` so a scenario is runnable offline (documented in config;
    real distances come from OSRM)."""
    npz = SCENARIO_DIR / name / "matrices.npz"
    if npz.exists():
        return TrafficState.from_cost_matrix(CostMatrix.load(npz))
    D = haversine_matrix(lat, lon)
    return TrafficState(D / cfg.fallback_speed_mps, D, np.ones_like(D))


def load_scenario(
    name: str, cfg: QTrafficConfig, rng: np.random.Generator, bounds: Bounds | None = None
) -> tuple[ProblemContext, Problem]:
    """Read data/scenarios/<name> into the pair every optimiser needs: the fitness
    `ProblemContext` (Phase 5) and the constraint `Problem` (Phase 6).

    The Phase 0 contract returned only a ProblemContext; constraints were not designed
    yet and repair needs demands, windows, capacities and shift ends, so the checked
    Problem is returned alongside instead of being rebuilt per call. Normalisation
    bounds are computed ONCE here and frozen [SPEC 7.2].
    """
    customers, vehicles, depot = scenario_io.load(name)
    lat = np.array([depot["lat"], *[c.lat for c in customers]])
    lon = np.array([depot["lon"], *[c.lon for c in customers]])
    traffic = traffic_state(name, lat, lon, cfg)
    n, n_veh = len(customers), len(vehicles)
    ctx = ProblemContext(
        traffic=traffic,
        bounds=bounds or compute_normalization_bounds(traffic, n, n_veh, rng),
        n_customers=n,
        n_vehicles=n_veh,
    )
    # the checks read the SAME c_ij matrix fitness reads [SPEC 10.2]
    prob = Problem.from_scenario(customers, vehicles, traffic.duration, cfg.rho_max)
    return ctx, prob


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
