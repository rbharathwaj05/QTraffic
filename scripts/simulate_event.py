"""Phase 9 exit condition: inject one congestion event and show what the controller does.

Usage: python scripts/simulate_event.py [--scenario S2] [--severity 0.7] [--budget 3]

Applies a synthetic congestion event to one edge, then prints, per affected vehicle:
Delta_A [SPEC 9.2 / v4 44], the hysteresis decision [v4 45-47] and A(e) [SPEC 10.3].
The cost matrix is updated through `PathIndex.invalidate_edge` only -- the script reports
how many OD pairs were touched versus the full N^2, so "no full rebuild" is visible, not
claimed.

ROAD LAYER: when `data/scenarios/<id>/{matrices.npz,path_index.pkl}` exist (built by
`road.build_scenario` against OSRM) they are used as-is. Offline, the script falls back to
a synthetic grid road layer -- haversine distances and a G x G cell grid whose cells are
"edges", each OD path being the cells its straight line crosses. Same fallback spirit as
`benchmark.traffic_state`; it exercises the exact same code path, with made-up geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.feasibility import Problem
from backend.fleet import scenario as scenario_io
from backend.fleet.state import FleetState
from backend.optimization.encoding import dim
from backend.optimization.fitness import (
    ProblemContext,
    TrafficState,
    compute_normalization_bounds,
)
from backend.optimization.qpso import QPSO, Evaluator
from backend.road.build_scenario import SCENARIO_DIR
from backend.road.cost_matrix import CostMatrix, TrafficVersion
from backend.road.geometry import haversine_matrix
from backend.road.path_index import PathIndex
from backend.simulation.clock import SimClock
from backend.traffic import congestion
from backend.traffic.controller import (
    ReplanController,
    affected_customers,
    affected_vehicles,
    degradation,
)
from backend.traffic.events import EventKind, TrafficEvent
from backend.traffic.simulator import TrafficSimulator

GRID = 12  # cells per axis in the offline fallback road layer


def grid_road_layer(lat: np.ndarray, lon: np.ndarray, cfg: QTrafficConfig):
    """Offline stand-in for OSRM + PathIndex: a GRID x GRID cell grid over the bbox.

    ponytail: synthetic geometry, not a road network. Each cell is an "edge"; path(i, j) is
    the set of cells its straight line crosses, sampled densely enough that no cell is
    skipped. Real runs load the OSRM-built artifacts instead and nothing downstream knows
    the difference.
    """
    n = len(lat)
    D = haversine_matrix(lat, lon)
    T_base = D / cfg.fallback_speed_mps
    n_edges = GRID * GRID

    def cells(i: int, j: int) -> np.ndarray:
        steps = max(2 * GRID, 2)
        y = np.linspace(lat[i], lat[j], steps)
        x = np.linspace(lon[i], lon[j], steps)
        gy = np.clip(((y - lat.min()) / (np.ptp(lat) or 1.0) * (GRID - 1)).astype(int), 0, GRID - 1)
        gx = np.clip(((x - lon.min()) / (np.ptp(lon) or 1.0) * (GRID - 1)).astype(int), 0, GRID - 1)
        return np.unique(gy * GRID + gx)

    index = PathIndex()
    index.n = n
    for i in range(n):
        for j in range(n):
            e = np.empty(0, dtype=np.int64) if i == j else cells(i, j)
            index.pair_to_edges[(i, j)] = e
            for k in e:
                index.edge_to_pairs.setdefault(int(k), set()).add((i, j))
    edge_t0 = np.ones(n_edges)  # every cell weighs the same in the inflation average
    matrix = CostMatrix(T_base, D, np.ones_like(T_base), edge_t0, TrafficVersion("offline"))
    return matrix, index


def load_road_layer(scenario_id: str, lat, lon, cfg):
    d = SCENARIO_DIR / scenario_id
    if (d / "matrices.npz").exists() and (d / "path_index.pkl").exists():
        m = CostMatrix.load(d / "matrices.npz", TrafficVersion(scenario_id))
        return m, PathIndex.load(d / "path_index.pkl"), "OSRM artifacts"
    m, idx = grid_road_layer(lat, lon, cfg)
    return m, idx, f"offline {GRID}x{GRID} grid fallback"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="S2")
    p.add_argument("--severity", type=float, default=0.7, help="rho of the injected event")
    p.add_argument("--budget", type=float, default=3.0, help="solver wall-clock seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--edge", type=int, default=None, help="edge id; default = most used")
    p.add_argument("--at", type=float, default=300.0, help="sim seconds at which the event lands")
    a = p.parse_args()

    cfg = QTrafficConfig()
    rng = np.random.default_rng(a.seed)
    customers, vehicles, depot = scenario_io.load(a.scenario)
    lat = np.array([depot["lat"], *[c.lat for c in customers]])
    lon = np.array([depot["lon"], *[c.lon for c in customers]])
    matrix, index, source = load_road_layer(a.scenario, lat, lon, cfg)

    traffic_before = TrafficState.from_cost_matrix(matrix)
    n, n_veh = len(customers), len(vehicles)
    bounds = compute_normalization_bounds(traffic_before, n, n_veh, rng)
    ctx = ProblemContext(traffic_before, bounds, n, n_veh)
    prob = Problem.from_scenario(customers, vehicles, traffic_before.duration, cfg.rho_max)

    # a real deployed plan to degrade
    opt = QPSO(cfg, Evaluator(ctx, prob, cfg), dim(n), rng)
    res = opt.run(time_budget_s=a.budget)
    state = FleetState.from_fleet_route(res.fleet, t_sim=0.0)
    for v in state.vehicles():  # put every vehicle one stop into its route
        state.advance(v)

    # pick the edge the fleet actually uses most -- an event nobody drives through is a
    # correct but boring demo
    in_use = [leg for v in state.vehicles() for leg in state.remaining_legs(v)]
    usage: dict[int, int] = {}
    for i, j in in_use:
        for e in index.edges_on(i, j):
            usage[int(e)] = usage.get(int(e), 0) + 1
    edge = a.edge if a.edge is not None else (max(usage, key=usage.get) if usage else 0)

    n_edges = len(matrix.edge_t0)
    sim = TrafficSimulator(n_edges, np.zeros(n_edges), np.ones(n_edges), [], index, matrix, cfg)
    event = TrafficEvent(
        id=1,
        kind=EventKind.CONGESTION,
        edge_ids=np.array([edge], dtype=np.int64),
        t_start=0.0,
        duration_s=1800.0,
        severity=a.severity,
    )
    # sim clock first: the plan was deployed at t=0, so run past T_cool before the event
    # lands, otherwise the only thing the demo shows is the cooldown holding [v4 47]
    clock = SimClock(speed=60)
    clock.advance_sim(a.at)
    sim.t = clock.now()
    sim.inject(event)
    version_before = matrix.version.value
    changed = sim.step(1.0)  # activates the event and pushes the scoped update
    traffic_after = TrafficState.from_cost_matrix(matrix)

    affected = sorted(affected_vehicles(state, index, [edge]))
    delta_fleet = degradation(state, state.vehicles(), traffic_before, traffic_after, bounds, cfg)
    ctrl = ReplanController(cfg)
    decision = ctrl.decide(sim.t, delta_fleet, affected, n_veh, state)

    print(
        f"scenario={a.scenario} N={n} M_veh={n_veh} road layer: {source}\n"
        f"clock: t = {sim.t:.0f}s sim (speed {clock.speed}x), T_cool = {cfg.T_cool:.0f}s\n"
        f"event: edge {edge}, severity {a.severity} -> factor "
        f"{congestion.rho_to_factor(a.severity):.2f}x, active {event.duration_s:.0f}s sim\n"
        f"cost matrix: {len(changed)} of {(n + 1) ** 2} OD pairs recomputed "
        f"({100 * len(changed) / (n + 1) ** 2:.1f}%), traffic_version "
        f"{version_before} -> {matrix.version.value}   [no full rebuild]\n"
        f"A(e) = {affected}  ({len(affected)}/{n_veh} vehicles), "
        f"|C_A| = {len(affected_customers(state, affected))} unserved customers\n"
        f"Delta_fleet = {delta_fleet:+.4f}  -> {decision.scope.value.upper()} "
        f"({decision.reason}, override={decision.override})"
    )
    print("\nper affected vehicle:")
    for v in affected:
        d_v = degradation(state, [v], traffic_before, traffic_after, bounds, cfg)
        cust = affected_customers(state, [v])
        print(
            f"  vehicle {v:>2}: Delta_A = {d_v:+.4f}  remaining stops = "
            f"{len(state.remaining_customers(v))}  C_v_remaining = {sorted(cust)}"
        )
    if not affected:
        print("  (none: no vehicle's remaining path uses that edge)")


if __name__ == "__main__":
    Path("benchmarks").mkdir(exist_ok=True)
    main()
