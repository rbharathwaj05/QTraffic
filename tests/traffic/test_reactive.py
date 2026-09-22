"""Phase 10: the reactive loop end to end [SPEC 9.1-9.5, v4 20-21 / 50-58]."""

import numpy as np
import pytest

from backend.config import QTrafficConfig
from backend.constraints.checker import check_all
from backend.constraints.feasibility import Problem
from backend.fleet import scenario as scenario_io
from backend.fleet.state import FleetState
from backend.optimization import local_scope
from backend.optimization.encoding import FleetRoute, dim
from backend.optimization.fitness import (
    ProblemContext,
    TrafficState,
    compute_normalization_bounds,
)
from backend.optimization.qpso import QPSO, Evaluator
from backend.road.path_index import PathIndex
from backend.traffic import reactive as mod
from backend.traffic.controller import ReplanController, Scope
from backend.traffic.reactive import Fallback, ReactiveLoop

S2_BUDGET = 4.0


# -- a small scenario with a CONTROLLED path index -----------------------------------------
def s2_world(seed=0, budget=S2_BUDGET):
    """S2 customers/vehicles with a hand-built index, so "affected" is something the test
    chooses rather than something the offline grid fallback decides.

    The grid fallback in `simulate_event.py` routes every pair through the same central
    cells, which would make every vehicle affected by every event and turn the scoping
    assertions vacuous. Here edge 1000 is placed on exactly one vehicle's legs.
    """
    cfg = QTrafficConfig()
    rng = np.random.default_rng(seed)
    customers, vehicles, depot = scenario_io.load("S2")
    lat = np.array([depot["lat"], *[c.lat for c in customers]])
    lon = np.array([depot["lon"], *[c.lon for c in customers]])
    d = np.hypot(lat[:, None] - lat, lon[:, None] - lon) * 111_000.0
    traffic = TrafficState(d / cfg.fallback_speed_mps, d, np.ones_like(d))
    n, n_veh = len(customers), len(vehicles)
    bounds = compute_normalization_bounds(traffic, n, n_veh, rng)
    ctx = ProblemContext(traffic, bounds, n, n_veh)
    prob = Problem.from_scenario(customers, vehicles, traffic.duration, cfg.rho_max)

    res = QPSO(cfg, Evaluator(ctx, prob, cfg), dim(n), rng).run(time_budget_s=budget)
    state = FleetState.from_fleet_route(res.fleet, t_sim=-np.inf)
    for v in state.vehicles():
        state.advance(v)
    return cfg, rng, state, prob, ctx, traffic


def index_for(state, victim: int, edge: int = 1000) -> PathIndex:
    """Every leg gets its own private edge; `victim`'s legs additionally carry `edge`."""
    idx = PathIndex()
    idx.n = state.n_vehicles
    e = 0
    for v in state.vehicles():
        for leg in state.remaining_legs(v):
            edges = [e] + ([edge] if v == victim else [])
            idx.pair_to_edges[leg] = np.array(edges, dtype=np.int64)
            for k in edges:
                idx.edge_to_pairs.setdefault(int(k), set()).add(leg)
            e += 1
    return idx


def worsen(traffic: TrafficState, factor: float) -> TrafficState:
    return TrafficState(traffic.duration * factor, traffic.distance, traffic.congestion * factor)


def make_loop(cfg, state, prob, ctx, index, rng):
    return ReactiveLoop(
        cfg=cfg,
        state=state,
        prob=prob,
        ctx=ctx,
        index=index,
        controller=ReplanController(cfg),
        rng=rng,
    )


# -- 10.4 switching gate [SPEC 9.3 point 4 / v4 20-21] --------------------------------------
def test_switching_cost_formula_counts_neighbour_changes():
    """N_changes counts a customer whose (vehicle, pred, succ) triple moved [v4 21]."""
    cfg = QTrafficConfig()
    d = np.full((5, 5), 100.0)
    np.fill_diagonal(d, 0.0)
    traffic = TrafficState(d, d, np.ones_like(d))
    current = {0: np.array([0, 1, 2, 0]), 1: np.array([0, 3, 4, 0])}
    same = {0: np.array([0, 1, 2, 0]), 1: np.array([0, 3, 4, 0])}
    s, d_change, n_changes, backtrack = mod.switching_cost(current, same, traffic, cfg, 1000.0)
    assert (s, d_change, n_changes, backtrack) == (0.0, 0.0, 0, 0.0)

    swapped = {0: np.array([0, 2, 1, 0]), 1: np.array([0, 3, 4, 0])}
    _, _, n_changes, _ = mod.switching_cost(current, swapped, traffic, cfg, 1000.0)
    assert n_changes == 2  # both customers on vehicle 0 changed neighbours

    moved = {0: np.array([0, 1, 0]), 1: np.array([0, 2, 3, 4, 0])}
    _, _, n_changes, _ = mod.switching_cost(current, moved, traffic, cfg, 1000.0)
    assert n_changes >= 2  # customer 2 changed vehicle; 3 changed predecessor


def test_backtracking_is_charged():
    """D_backtrack: a new plan that re-drives a leg it already drove, reversed [v4 20]."""
    cfg = QTrafficConfig()
    d = np.full((4, 4), 100.0)
    np.fill_diagonal(d, 0.0)
    traffic = TrafficState(d, d, np.ones_like(d))
    current = {0: np.array([0, 1, 2, 0])}
    back = {0: np.array([0, 2, 1, 0])}  # drives 2->1, the reverse of the current 1->2
    _, _, _, d_backtrack = mod.switching_cost(current, back, traffic, cfg, 100.0)
    assert d_backtrack > 0.0


def test_worked_example_a_three_percent_gain_is_not_worth_switching():
    """[SPEC 9.3] "a 30-minute route replaced by a 29-minute route" must NOT be deployed
    once the switching cost is charged -- "The reroute is correctly not issued."

    Built literally: the incumbent takes 30 min, the candidate 29 min (a 3.3 % gain) with
    every customer's neighbours changed, and the gate declines.
    """
    cfg = QTrafficConfig()
    n = 7
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = 300.0  # 5 min a leg; a 6-leg route is 30 min
    traffic = TrafficState(d, d, np.ones_like(d))
    current = np.array([0, 1, 2, 3, 4, 5, 0])
    state = FleetState.from_fleet_route(FleetRoute([current], np.zeros(5, dtype=np.int64)))
    bounds = compute_normalization_bounds(traffic, n - 1, 1, np.random.default_rng(0))

    faster = d.copy()
    faster[0, 5] = faster[5, 4] = faster[4, 3] = 280.0  # 3*280 + 3*300 = 29 min
    traffic_new = TrafficState(faster, faster, np.ones_like(faster))
    candidate = {0: np.array([0, 5, 4, 3, 2, 1, 0])}

    cur_min = float(d[current[:-1], current[1:]].sum()) / 60.0
    new_min = float(faster[candidate[0][:-1], candidate[0][1:]].sum()) / 60.0
    assert cur_min == pytest.approx(30.0)
    assert new_min == pytest.approx(29.0)
    assert (cur_min - new_min) / cur_min < 0.05  # the ~3 % gain of the worked example

    check = mod.assess_switch(state, [0], candidate, traffic_new, bounds, cfg)
    assert check.n_changes == 5  # every customer's neighbours moved
    assert check.s > 0.0
    assert not check.deploy  # the reroute is correctly not issued
    assert check.i_net <= cfg.epsilon_switch


def test_a_large_gain_does_clear_the_gate():
    """The counterpart: the gate must not simply refuse everything.

    Both sides are measured on the SAME snapshot, so the candidate has to be genuinely
    cheaper as a PLAN -- here the incumbent walks an expensive chain and the candidate
    takes the cheap one, a ~10x improvement that easily outweighs S.
    """
    cfg = QTrafficConfig()
    n = 5
    d = np.full((n, n), 6000.0)  # expensive everywhere...
    np.fill_diagonal(d, 0.0)
    for i, j in [(0, 4), (4, 3), (3, 2), (2, 1), (1, 0)]:
        d[i, j] = 60.0  # ...except along one cheap chain, which only the candidate uses
    traffic = TrafficState(d, d, np.ones_like(d))
    state = FleetState.from_fleet_route(
        FleetRoute([np.array([0, 1, 2, 3, 4, 0])], np.zeros(4, dtype=np.int64))
    )
    bounds = compute_normalization_bounds(traffic, n - 1, 1, np.random.default_rng(0))
    check = mod.assess_switch(state, [0], {0: np.array([0, 4, 3, 2, 1, 0])}, traffic, bounds, cfg)
    assert check.f_new < check.f_current
    assert check.deploy and check.i_net > cfg.epsilon_switch


# -- 10.3 scoping [v4 50-55] ----------------------------------------------------------------
def test_local_scope_decodes_into_the_affected_vehicles_only():
    """[v4 55] "This prevents a local rerouting problem involving four vehicles from
    accidentally assigning customers to all 100 vehicles"."""
    cfg, rng, state, prob, ctx, traffic = s2_world()
    A = [1, 3]
    scope = local_scope.build_scope(state, prob, ctx, A)
    assert scope.n_vehicles == 2 and scope.ctx.n_vehicles == 2
    assert scope.vehicles == (1, 3)
    expected = {int(c) for v in A for c in state.remaining_customers(v)}
    assert set(scope.customers) == expected
    assert scope.prob.duration.shape == (len(expected) + 1, len(expected) + 1)

    routes, result, _ = __import__("backend.optimization.ebqpso", fromlist=["x"]).reoptimize_local(
        state, prob, ctx, A, cfg, rng, time_budget_s=1.0
    )
    assert set(routes) == {1, 3}  # nothing outside A can be produced at all
    served = {int(c) for r in routes.values() for c in r if int(c) != 0}
    assert served >= expected  # every affected customer is still covered


def test_local_routes_keep_the_travelled_prefix():
    """[SPEC 9.3 point 1] R_v_new = R_v_travelled (+) R_v_optimized."""
    cfg, rng, state, prob, ctx, _ = s2_world()
    state.advance(2, stops=2)
    prefix = state.routes[2][: int(state.position[2]) + 1].copy()
    scope = local_scope.build_scope(state, prob, ctx, [2])
    local = FleetRoute(
        [np.array([0, *range(1, scope.n_customers + 1), 0])],
        np.zeros(scope.n_customers, dtype=np.int64),
    )
    out = scope.to_global(local, state)
    assert np.array_equal(out[2][: len(prefix)], prefix)


def test_warm_start_is_the_incumbent_plus_jitter():
    """[v4 53-54] N_warm = warm_fraction * M seeds, the first of them exact."""
    cfg, rng, state, prob, ctx, _ = s2_world()
    scope = local_scope.build_scope(state, prob, ctx, [0, 1])
    seeds = local_scope.warm_start_particles(scope, cfg, rng)
    assert len(seeds) == int(round(cfg.warm_fraction * cfg.M))
    assert seeds.min() >= 0.0 and seeds.max() <= 1.0
    from backend.optimization.encoding import decode

    assert decode(seeds[:1], scope.n_vehicles)[0].routes  # row 0 decodes to a real plan
    assert not np.allclose(seeds[0], seeds[1])  # the rest are perturbed, not copies


# -- 10.1 detection does no graph work [SPEC 9.1] -------------------------------------------
def test_detection_makes_no_osrm_or_shortest_path_calls(monkeypatch):
    """T_detection is a cost-matrix lookup, never a fresh graph search. Asserted by
    counting calls into osrm_client and networkx.shortest_path, not by timing."""
    import networkx as nx

    from backend.road import osrm_client

    calls = {"osrm": 0, "nx": 0}

    class Boom(osrm_client.OSRMClient):
        def route(self, *a, **k):
            calls["osrm"] += 1
            raise AssertionError("detection called OSRM")

        def table(self, *a, **k):
            calls["osrm"] += 1
            raise AssertionError("detection called OSRM")

    monkeypatch.setattr(osrm_client, "OSRMClient", Boom)
    for name in ("shortest_path", "dijkstra_path", "astar_path"):
        if hasattr(nx, name):
            monkeypatch.setattr(nx, name, lambda *a, **k: calls.__setitem__("nx", calls["nx"] + 1))

    cfg, rng, state, prob, ctx, traffic = s2_world()
    loop = make_loop(cfg, state, prob, ctx, index_for(state, victim=0), rng)
    entry = loop.on_event(500.0, [1000], traffic, worsen(traffic, 1.5))
    assert calls == {"osrm": 0, "nx": 0}
    assert entry.t_detection > 0.0  # it was measured, not skipped


# -- 10.5 the loop, end to end --------------------------------------------------------------
def test_severe_event_deploys_a_fallback_then_evaluates_the_gate():
    """Correctness bar: a severe event on S2 must leave a feasible plan, log the swarm, run
    the gate, and leave every UNAFFECTED vehicle's route byte-identical."""
    cfg, rng, state, prob, ctx, traffic = s2_world()
    victim = 0
    index = index_for(state, victim=victim)
    before_routes = {v: state.routes[v].copy() for v in state.vehicles()}
    was_ok = check_all(state.to_fleet_route(), prob).ok

    loop = make_loop(cfg, state, prob, ctx, index, rng)
    entry = loop.on_event(500.0, [1000], traffic, worsen(traffic, 3.0))

    assert entry.delta > cfg.theta_override  # severe, per the bar
    assert entry.scope != Scope.NONE.value
    assert entry.affected == (victim,)  # path index, not proximity [SPEC 10.3]
    assert entry.t_detection > 0 and entry.t_optimization > 0 and entry.t_deployment > 0
    assert entry.t_response == pytest.approx(
        entry.t_detection + entry.t_optimization + entry.t_deployment
    )
    assert entry.b_available > 0  # the swarm was given a box, not an iteration count
    assert entry.deployed in (Fallback.GREEDY, Fallback.SWARM, Fallback.NONE)
    assert entry.feasible == was_ok or entry.feasible  # never made worse

    for v in state.vehicles():
        if v != victim:
            assert np.array_equal(
                state.routes[v], before_routes[v]
            ), f"vehicle {v} was not affected but its route changed -- scoping leaked"


def test_flickering_traffic_through_the_whole_loop_deploys_nothing():
    """Phase 9's pinned worked example, driven through the FULL loop rather than the
    controller alone: 10/11/9/12/8/13 % must issue zero reroutes [SPEC 9.2]."""
    cfg, rng, state, prob, ctx, traffic = s2_world()
    index = index_for(state, victim=0)
    before_routes = {v: state.routes[v].copy() for v in state.vehicles()}
    loop = make_loop(cfg, state, prob, ctx, index, rng)

    # factors chosen so Delta lands on the spec's flickering series, alternating either
    # side of theta_soft so the persistence rule is never satisfied
    for t, factor in enumerate([1.10, 1.11, 1.09, 1.12, 1.08, 1.13]):
        loop.on_event(100.0 * t, [1000], traffic, worsen(traffic, factor))

    assert all(e.deployed is Fallback.NONE for e in loop.log)
    assert loop.summary()["reroutes_issued"] == 0
    for v in state.vehicles():
        assert np.array_equal(state.routes[v], before_routes[v])


def test_every_response_component_is_logged_for_every_event():
    """[SPEC 15] the system never claims real time; it logs the three components and the
    total per event, which is what makes the claim checkable."""
    cfg, rng, state, prob, ctx, traffic = s2_world()
    loop = make_loop(cfg, state, prob, ctx, index_for(state, victim=1), rng)
    loop.on_event(10.0, [1000], traffic, worsen(traffic, 1.02))  # noise
    loop.on_event(600.0, [1000], traffic, worsen(traffic, 3.0))  # severe
    assert len(loop.log) == 2
    for e in loop.log:
        row = e.as_row()
        for key in ("T_detection", "T_optimization", "T_deployment", "T_response"):
            assert key in row and row[key] >= 0.0
        assert "RepairDistance" in row and "RepairIterations" in row
    assert loop.summary()["events"] == 2


def test_a_quiet_event_never_reaches_the_optimiser():
    cfg, rng, state, prob, ctx, traffic = s2_world()
    loop = make_loop(cfg, state, prob, ctx, index_for(state, victim=0), rng)
    entry = loop.on_event(10.0, [1000], traffic, worsen(traffic, 1.001))
    assert entry.scope == Scope.NONE.value
    assert entry.t_optimization == 0.0 and entry.deployed is Fallback.NONE


# -- 10.2 fallback ladder [doc2 31] ---------------------------------------------------------
def test_greedy_fallback_covers_every_remaining_customer_and_keeps_the_prefix():
    cfg, rng, state, prob, ctx, _ = s2_world()
    state.advance(0, stops=2)
    prefix = state.routes[0][: int(state.position[0]) + 1].copy()
    pending = {int(c) for c in state.remaining_customers(0)}
    route = mod.greedy_insertion_route(state, 0, prob, cfg)
    assert np.array_equal(route[: len(prefix)], prefix)
    assert {int(c) for c in route if int(c) != 0} >= pending
    assert route[0] == 0 and route[-1] == 0


def test_the_cache_serves_a_prior_route_for_the_same_vehicle_and_edge():
    cache = mod.RouteCache()
    assert cache.get(3, [77]) is None
    cache.put(3, [77], np.array([0, 5, 0]))
    assert np.array_equal(cache.get(3, [77]), np.array([0, 5, 0]))
    assert cache.get(4, [77]) is None  # a different vehicle is a different situation
    assert cache.get(3, [78]) is None  # and so is a different edge


def test_the_ladder_picks_its_level_by_the_budget_already_spent():
    cfg, rng, state, prob, ctx, _ = s2_world()
    loop = make_loop(cfg, state, prob, ctx, index_for(state, victim=0), rng)
    _, level = loop.fast_fallback([0], [1000], b_available=10.0, spent=0.05)
    assert level is Fallback.GREEDY  # nothing cached yet, so level 1 is unavailable

    loop.cache.put(0, [1000], state.routes[0])
    _, level = loop.fast_fallback([0], [1000], b_available=10.0, spent=0.05)
    assert level is Fallback.CACHED  # 5 % of the box spent: level 1

    _, level = loop.fast_fallback([0], [1000], b_available=10.0, spent=9.95)
    assert level is Fallback.NONE  # past fallback_l2_fraction: keep the current route
