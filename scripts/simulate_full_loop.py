"""Phase 10 exit condition: run the full reactive loop over a scripted event sequence.

Usage: python scripts/simulate_full_loop.py [--scenario S3] [--minutes 30] [--seed 0]

Runs N simulated minutes with a mix of noisy and severe events and prints, per event, the
T_response breakdown [SPEC 9.1] plus a closing "reroutes issued / unnecessary" summary.

SPEC 15 applies to how this output is read: it is NOT a claim that the system responds in
real time. It is the measurement -- T_detection, T_optimization, T_deployment and their
sum, per event, against the configured target.

Road layer: the OSRM-built artifacts when `data/scenarios/<id>/{matrices.npz,
path_index.pkl}` exist, else the same synthetic grid fallback `simulate_event.py` uses.
"""

from __future__ import annotations

import argparse

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.feasibility import Problem
from backend.fleet import scenario as scenario_io
from backend.fleet.state import FleetState
from backend.optimization.encoding import dim
from backend.optimization.fitness import ProblemContext, TrafficState, compute_normalization_bounds
from backend.optimization.qpso import QPSO, Evaluator
from backend.simulation.clock import SimClock
from backend.traffic.controller import ReplanController
from backend.traffic.events import EventKind, TrafficEvent
from backend.traffic.reactive import Fallback, ReactiveLoop
from backend.traffic.simulator import TrafficSimulator
from scripts.simulate_event import load_road_layer

COLUMNS = [
    ("event", 6),
    ("t_sim", 7),
    ("delta", 8),
    ("scope", 6),
    ("n_affected", 10),
    ("T_detection", 11),
    ("T_optimization", 14),
    ("T_deployment", 12),
    ("T_response", 10),
    ("fallback", 8),
    ("deployed", 8),
    ("I_net", 8),
    ("S", 7),
    ("N_changes", 9),
    ("RepairDistance", 15),
    ("RepairIterations", 17),
    ("gate", 6),
    ("feasible", 8),
    ("was_feasible", 12),
]


def event_script(n_edges: int, busiest: list[int], minutes: float) -> list[TrafficEvent]:
    """Three noisy events and two severe ones, spread over the run [SPEC 9.2 ladder].

    Severity alone does not decide Delta: an event on ONE edge of a path contributes only
    its share of that path's weighted average, so a single-edge 0.95 event can still land
    in the noise band. The severe events therefore hit the fleet's busiest `severe_edges`
    at once -- which is also what a real incident looks like, a corridor rather than a
    metre of tarmac -- so the run reliably exercises the trigger branch as well as the
    ignore branch instead of depending on which initial plan the budgeted solve produced.
    """
    horizon = minutes * 60.0
    noise = (busiest[3:] + [0, 1, 2, 3, 4])[:3]
    severe = np.array(busiest[:6] or [0], dtype=np.int64)
    plan = [
        (0.15, np.array([noise[0] % max(n_edges, 1)], dtype=np.int64)),
        (0.20, np.array([noise[1] % max(n_edges, 1)], dtype=np.int64)),
        (0.95, severe),
        (0.18, np.array([noise[2] % max(n_edges, 1)], dtype=np.int64)),
        (0.90, severe),
    ]
    return [
        TrafficEvent(
            id=i,
            kind=EventKind.CONGESTION,
            edge_ids=edges,
            t_start=horizon * (i + 1) / (len(plan) + 1),
            duration_s=600.0,
            severity=sev,
        )
        for i, (sev, edges) in enumerate(plan)
    ]


def _gate_result(entry) -> str:
    """What the switching gate said, as one word for the table [SPEC 9.3 point 4]."""
    if entry.scope == "none":
        return "-"
    if not entry.swarm_in_budget:
        return "late"  # the swarm missed its box, so the gate never saw it
    return "pass" if entry.deployed is Fallback.SWARM else "hold"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="S3")
    p.add_argument("--minutes", type=float, default=30.0, help="simulated minutes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=float, default=8.0, help="initial-plan solver seconds")
    a = p.parse_args()

    cfg = QTrafficConfig()
    rng = np.random.default_rng(a.seed)
    customers, vehicles, depot = scenario_io.load(a.scenario)
    lat = np.array([depot["lat"], *[c.lat for c in customers]])
    lon = np.array([depot["lon"], *[c.lon for c in customers]])
    matrix, index, source = load_road_layer(a.scenario, lat, lon, cfg)

    traffic = TrafficState.from_cost_matrix(matrix)
    n, n_veh = len(customers), len(vehicles)
    bounds = compute_normalization_bounds(traffic, n, n_veh, rng)
    ctx = ProblemContext(traffic, bounds, n, n_veh)
    prob = Problem.from_scenario(customers, vehicles, traffic.duration, cfg.rho_max)

    # the plan the day starts with
    opt = QPSO(cfg, Evaluator(ctx, prob, cfg), dim(n), rng)
    initial = opt.run(time_budget_s=a.budget)
    state = FleetState.from_fleet_route(initial.fleet, t_sim=-np.inf)
    for v in state.vehicles():
        state.advance(v)  # everyone is one stop in when the day's traffic starts
    before_routes = {v: state.routes[v].copy() for v in state.vehicles()}

    n_edges = len(matrix.edge_t0)
    usage: dict[int, int] = {}
    for v in state.vehicles():
        for i, j in state.remaining_legs(v):
            for e in index.edges_on(i, j):
                usage[int(e)] = usage.get(int(e), 0) + 1
    busiest = sorted(usage, key=usage.get, reverse=True)[:10]

    events = event_script(n_edges, busiest, a.minutes)
    sim = TrafficSimulator(n_edges, np.zeros(n_edges), np.ones(n_edges), events, index, matrix, cfg)
    loop = ReactiveLoop(
        cfg=cfg,
        state=state,
        prob=prob,
        ctx=ctx,
        index=index,
        controller=ReplanController(cfg),
        rng=rng,
    )

    clock = SimClock(speed=60)
    step_s = 30.0
    print(
        f"scenario={a.scenario} N={n} M_veh={n_veh} road layer: {source}\n"
        f"initial plan: F={initial.gbest_fitness:.4f} feasible={initial.feasible} "
        f"residual={initial.residual:.4g} (solved in {initial.elapsed_s:.1f}s)\n"
        f"{a.minutes:.0f} simulated minutes, {len(events)} scheduled events, "
        f"clock {clock.speed}x, step {step_s:.0f}s sim\n"
        f"budgets: local {cfg.T_response_local}s fleet {cfg.T_response_fleet}s "
        f"(targets, measured per event [SPEC 15])\n"
    )
    print("".join(name.rjust(w) for name, w in COLUMNS))

    while clock.now() < a.minutes * 60.0:
        t = clock.advance_sim(step_s)
        before = TrafficState.from_cost_matrix(matrix)
        changed = sim.step(step_s)
        if len(changed) == 0:
            continue
        batch = sim.take_batch(force=True)  # debounce window handled by the simulator
        if not batch:
            continue
        after = TrafficState.from_cost_matrix(matrix)
        entry = loop.on_event(t, batch, before, after)
        row = entry.as_row()
        row["event"] = len(loop.log)
        row["gate"] = _gate_result(entry)
        print("".join(str(row[name]).rjust(w) for name, w in COLUMNS))

    s = loop.summary()
    untouched = [v for v in state.vehicles() if np.array_equal(state.routes[v], before_routes[v])]
    print(
        f"\nevents seen: {s['events']}   triggered: {s['triggered']}\n"
        f"reroutes issued: {s['reroutes_issued']}, unnecessary: {s['unnecessary']} "
        f"(swarm finished in budget, switching gate declined)\n"
        f"max T_response: {s['max_T_response']}s against the applicable target "
        f"(local {cfg.T_response_local}s / fleet {cfg.T_response_fleet}s)\n"
        f"vehicles never touched: {len(untouched)}/{n_veh} "
        f"(scoping: an unaffected vehicle's route is byte-identical)"
    )
    levels = {lvl.name: sum(1 for e in loop.log if e.fallback is lvl) for lvl in Fallback}
    print(f"fallback levels used: {levels}")


if __name__ == "__main__":
    main()
