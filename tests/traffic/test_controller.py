import numpy as np
import pytest

from backend.config import QTrafficConfig
from backend.fleet import route_manager as rm
from backend.fleet.state import FleetState
from backend.optimization.encoding import FleetRoute
from backend.optimization.fitness import Bounds, TrafficState, combine
from backend.road.path_index import PathIndex
from backend.traffic import controller as mod
from backend.traffic.controller import Decision, ReplanController, Scope
from tests.traffic.helpers import COORDS, cost_matrix, path_index

BOUNDS = Bounds(t=(0.0, 1000.0), d=(0.0, 10000.0), c=(0.0, 10000.0))


def three_vehicle_state():
    """A: 0->1->0 (uses edge 0), B: 0->2->0 (next to edge 0, does NOT use it),
    C: 0->3->0 (far away, DOES use edge 0). See tests/traffic/helpers.PATHS."""
    routes = [np.array([0, 1, 0]), np.array([0, 2, 0]), np.array([0, 3, 0])]
    return FleetState.from_fleet_route(FleetRoute(routes, np.array([0, 1, 2])), t_sim=-np.inf)


def traffic(factor_scale: float = 1.0) -> TrafficState:
    m = cost_matrix()
    return TrafficState(m.duration_s * factor_scale, m.distance_m, m.factor * factor_scale)


# -- 9.7 affected vehicles: path index, never proximity [SPEC 10.3] -----------------------
def test_only_vehicles_whose_path_uses_the_edge_are_affected():
    """Correctness bar: 3 vehicles, 1 closed edge, exactly the users come back."""
    state, idx = three_vehicle_state(), path_index()
    assert mod.affected_vehicles(state, idx, [0]) == {0, 2}
    assert mod.affected_vehicles(state, idx, [4]) == set()  # nobody drives edge 4
    assert mod.affected_vehicles(state, idx, [2]) == {1}


def test_geographic_proximity_would_give_a_different_wrong_answer():
    """[SPEC 10.3] proximity "over-counts ... under-counts" -- this is that example.

    Vehicle 1 (node 2) sits right beside edge 0 but is routed around it: proximity
    over-counts it. Vehicle 2 (node 3) is 70 km away but its path runs through edge 0:
    proximity under-counts it. The path index gets both right.
    """
    state, idx = three_vehicle_state(), path_index()
    edge_0_location = COORDS[0]  # edge 0 runs from node 0 towards node 1

    def by_proximity(radius: float) -> set[int]:
        served = {0: 1, 1: 2, 2: 3}  # vehicle -> the customer node it visits
        return {
            v
            for v, node in served.items()
            if np.linalg.norm(COORDS[node] - edge_0_location) <= radius
        }

    proximity = by_proximity(radius=2.0)
    by_index = mod.affected_vehicles(state, idx, [0])
    assert proximity == {0, 1}
    assert by_index == {0, 2}
    assert 1 in proximity and 1 not in by_index  # over-count
    assert 2 in by_index and 2 not in proximity  # under-count


def test_batched_edges_are_the_union_of_their_affected_sets():
    """A(E_event) = union over e of A(e) [v4 48]."""
    state, idx = three_vehicle_state(), path_index()
    assert mod.affected_vehicles(state, idx, [0, 2]) == {0, 1, 2}


def test_affected_customers_are_the_unserved_ones_only():
    """C_A = union of C_v_remaining [v4 49]: a delivered customer is never re-planned."""
    state, idx = three_vehicle_state(), path_index()
    assert mod.affected_customers(state, [0, 2]) == {1, 3}
    state.advance(0)  # vehicle 0 delivers customer 1
    assert mod.affected_customers(state, [0]) == set()
    assert mod.affected_vehicles(state, idx, [0]) == {0, 2}  # still driving edge 0 home


# -- 9.4 degradation [SPEC 9.2 / v4 44] ---------------------------------------------------
def test_degradation_is_the_relative_increase_of_the_remaining_plan():
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    before, after = traffic(1.0), traffic(1.4)
    # duration and congestion both scale by 1.4; distance is unchanged, so the fleet-level
    # increase sits between 0 and 0.4 and is strictly positive
    d = mod.degradation(state, state.vehicles(), before, after, BOUNDS, cfg)
    assert 0.0 < d < 0.4
    assert mod.degradation(state, state.vehicles(), before, before, BOUNDS, cfg) == 0.0


def test_degradation_ignores_the_travelled_prefix():
    """[SPEC 9.3 point 1] only the remaining suffix can still be made worse."""
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    before, after = traffic(1.0), traffic(2.0)
    full = mod.degradation(state, [0], before, after, BOUNDS, cfg)
    state.advance(0)  # half the route is now history
    partial_cost = mod.remaining_cost(state, [0], before, BOUNDS, cfg)
    assert partial_cost < mod.remaining_cost(three_vehicle_state(), [0], before, BOUNDS, cfg)
    assert mod.degradation(state, [0], before, after, BOUNDS, cfg) == pytest.approx(full)


def test_degradation_of_a_finished_vehicle_is_zero_not_a_division_error():
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    state.advance(0, stops=5)  # parked at the depot, nothing remaining
    assert mod.degradation(state, [0], traffic(1.0), traffic(3.0), BOUNDS, cfg) == 0.0


def test_degradation_is_scale_invariant_between_one_vehicle_and_the_fleet():
    """Same uniform slowdown -> same Delta whichever subset it is measured on."""
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    before, after = traffic(1.0), traffic(1.5)
    fleet = mod.degradation(state, state.vehicles(), before, after, BOUNDS, cfg)
    one = mod.degradation(state, [1], before, after, BOUNDS, cfg)
    assert one == pytest.approx(fleet)


# -- 9.5 hysteresis: the two worked examples, pinned [SPEC 9.2] ---------------------------
def test_worked_example_fluctuating_traffic_issues_zero_reroutes():
    """Spec worked example 1: readings 10 %, 11 %, 9 %, 12 %, 8 %, 13 % -> no reroute.

    Nothing ever exceeds theta_hard, and no two CONSECUTIVE readings stay above
    theta_soft, so the persistence requirement [v4 46] is never met.
    """
    state = three_vehicle_state()
    ctrl = ReplanController(QTrafficConfig())
    decisions = [
        ctrl.decide(t * 1000.0, d, [0], 3, state)
        for t, d in enumerate([0.10, 0.11, 0.09, 0.12, 0.08, 0.13])
    ]
    assert all(x.scope is Scope.NONE for x in decisions)
    assert ctrl.last_replan_t == -np.inf  # nothing fired


def test_worked_example_real_disruption_reroutes_immediately_through_cooldown():
    """Spec worked example 2: F 50 -> 70 is Delta = 40 % -> reroute now, cooldown ignored."""
    state = three_vehicle_state()
    state.last_reopt[:] = 1000.0  # every vehicle re-planned a second ago: cooldown ACTIVE
    ctrl = ReplanController(QTrafficConfig())
    delta = ctrl.cost_delta(70.0, 50.0)
    assert delta == pytest.approx(0.40)

    d = ctrl.decide(1001.0, delta, [0, 1, 2], 3, state)
    assert d.scope is Scope.FLEET and d.override is True
    assert d.reason == "severe override"
    assert not ctrl.cooldown_ok(state, [0], 1001.0)  # the cooldown really was active


def test_persistence_needs_two_consecutive_cycles_above_theta_soft():
    """[v4 46] one reading in the soft band arms; the second one fires."""
    state = three_vehicle_state()
    ctrl = ReplanController(QTrafficConfig())
    first = ctrl.decide(0.0, 0.12, [0], 3, state)
    assert first.scope is Scope.NONE and "armed 1/2" in first.reason
    second = ctrl.decide(1.0, 0.12, [0], 3, state)
    assert second.scope is Scope.LOCAL and second.override is False


def test_a_quiet_reading_disarms_the_persistence_counter():
    state = three_vehicle_state()
    ctrl = ReplanController(QTrafficConfig())
    ctrl.decide(0.0, 0.12, [0], 3, state)
    assert ctrl.decide(1.0, 0.05, [0], 3, state).scope is Scope.NONE
    assert ctrl.armed_cycles == 0
    assert ctrl.decide(2.0, 0.12, [0], 3, state).scope is Scope.NONE  # armed again, not fired


def test_hard_trigger_fires_on_a_single_reading():
    state = three_vehicle_state()
    d = ReplanController(QTrafficConfig()).decide(0.0, 0.20, [0], 3, state)
    assert d.scope is Scope.LOCAL and d.reason == "above theta_hard" and d.override is False


def test_exactly_theta_soft_is_noise_not_a_trigger():
    """The ladder is `Delta <= theta_soft -> ignore` [SPEC 9.2]."""
    state = three_vehicle_state()
    ctrl = ReplanController(QTrafficConfig())
    assert ctrl.decide(0.0, 0.10, [0], 3, state).scope is Scope.NONE
    assert ctrl.armed_cycles == 0


# -- 9.6 cooldown [v4 47] ------------------------------------------------------------------
def test_hard_trigger_is_held_while_the_cooldown_runs():
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    state.last_reopt[:] = 100.0
    ctrl = ReplanController(cfg)
    held = ctrl.decide(100.0 + cfg.T_cool - 1.0, 0.20, [0], 3, state)
    assert held.scope is Scope.NONE and "cooldown" in held.reason
    fired = ctrl.decide(100.0 + cfg.T_cool, 0.20, [0], 3, state)
    assert fired.scope is Scope.LOCAL


def test_cooldown_is_per_vehicle_and_in_sim_seconds():
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    state.last_reopt[:] = -np.inf
    state.last_reopt[1] = 100.0  # only vehicle 1 is cooling down
    ctrl = ReplanController(cfg)
    assert ctrl.cooldown_ok(state, [0, 2], 150.0)
    assert not ctrl.cooldown_ok(state, [1], 150.0)
    fired = ctrl.decide(150.0, 0.20, [0, 2], 3, state)
    assert fired.scope is Scope.FLEET  # 2 of 3 vehicles -> wider than LOCAL
    assert ctrl.decide(151.0, 0.20, [1], 3, state).scope is Scope.NONE  # still cooling


def test_record_starts_the_cooldown_of_the_replanned_vehicles_only():
    state = three_vehicle_state()
    ctrl = ReplanController(QTrafficConfig())
    ctrl.record(500.0, state, [0, 2])
    assert state.last_reopt[0] == 500.0 and state.last_reopt[2] == 500.0
    assert state.last_reopt[1] == -np.inf
    assert ctrl.armed_cycles == 0 and ctrl.last_replan_t == 500.0


# -- scope ---------------------------------------------------------------------------------
def test_scope_widens_to_the_fleet_when_most_vehicles_are_affected():
    state = three_vehicle_state()
    cfg = QTrafficConfig()
    assert ReplanController(cfg).decide(0.0, 0.20, [0], 3, state).scope is Scope.LOCAL
    assert ReplanController(cfg).decide(0.0, 0.20, [0, 1, 2], 3, state).scope is Scope.FLEET


def test_severe_override_is_always_fleet_wide():
    state = three_vehicle_state()
    d = ReplanController(QTrafficConfig()).decide(0.0, 0.9, [0], 3, state)
    assert d.scope is Scope.FLEET and d.override is True


def test_cost_delta_does_not_divide_by_zero():
    ctrl = ReplanController(QTrafficConfig())
    assert ctrl.cost_delta(50.0, 0.0) > 0  # finite, not a ZeroDivisionError
    assert ctrl.cost_delta(70.0, 50.0) == pytest.approx(0.4)


def test_decision_carries_the_vehicles_it_covers():
    state = three_vehicle_state()
    d = ReplanController(QTrafficConfig()).decide(0.0, 0.20, [2, 0], 3, state)
    assert isinstance(d, Decision) and d.vehicles == (2, 0)


# -- F3: Delta and the optimiser's F are ONE formula ---------------------------------------
def test_remaining_cost_is_fitness_combine_on_remaining_rebased_bounds():
    """No parallel objective lives in the controller: `remaining_cost` must equal a direct
    `fitness.combine` call on `bounds.for_remaining()` with R' = 0."""
    cfg = QTrafficConfig()
    state, tr = three_vehicle_state(), traffic(1.0)
    T = D = C = 0.0
    for v in state.vehicles():
        t, d, c = rm.route_cost_terms(
            state.routes[v], int(state.position[v]), tr.duration, tr.distance, tr.congestion
        )
        T, D, C = T + t, D + d, C + c
    direct = combine(
        np.array([T]), np.array([D]), np.array([C]), np.zeros(1), BOUNDS.for_remaining(), cfg
    )
    assert mod.remaining_cost(state, state.vehicles(), tr, BOUNDS, cfg) == pytest.approx(
        float(direct[0])
    )


def test_remaining_rebased_bounds_keep_the_scale_and_drop_only_the_offset():
    """[CLAUDE.md rule 5] with lo = 0 the min-max form IS the ratio form w_t T/T_ref + ...,
    which is why one definition of F now covers both the swarm and the degradation check."""
    cfg = QTrafficConfig()
    rb = BOUNDS.for_remaining()
    assert rb.t[0] == rb.d[0] == rb.c[0] == 0.0  # re-based at zero
    assert rb.t[1] == BOUNDS.t[1] - BOUNDS.t[0]  # same frozen spread
    T, D, C = 300.0, 4000.0, 5000.0
    ratio = cfg.w_t * T / rb.t[1] + cfg.w_d * D / rb.d[1] + cfg.w_c * C / rb.c[1]
    viacombine = combine(np.array([T]), np.array([D]), np.array([C]), np.zeros(1), rb, cfg)
    assert float(viacombine[0]) == pytest.approx(ratio)


def test_remaining_cost_is_never_negative_under_the_rebased_bounds():
    """The failure the rebasing fixes: a single vehicle's remaining legs sit far below a
    fleet-scale `lo`, so the un-rebased form could drive F_before below zero and flip the
    sign of Delta."""
    cfg = QTrafficConfig()
    state, tr = three_vehicle_state(), traffic(1.0)
    for v in state.vehicles():
        assert mod.remaining_cost(state, [v], tr, BOUNDS, cfg) >= 0.0
    raw = combine(  # the old, un-rebased call, for contrast
        np.array([30.0]), np.array([300.0]), np.array([300.0]), np.zeros(1), BOUNDS, cfg
    )
    assert float(raw[0]) < mod.remaining_cost(state, [0], tr, BOUNDS, cfg)


# -- F5: an edge only in the TRAVELLED prefix must not make a vehicle affected -------------
def test_a_vehicle_that_has_already_driven_the_edge_is_not_affected():
    """[SPEC 9.3 point 1] the frozen prefix cannot be degraded, so it cannot trigger.
    The other pinned tests happen to use vehicles that still drive the edge on the way
    home; this one constructs the case where the edge is genuinely behind the vehicle."""
    idx = PathIndex()
    idx.n = 3
    # edge 7 is used ONLY by the (0, 1) leg -- the first leg of the route below
    paths = {(0, 1): [7], (1, 0): [7], (1, 2): [8], (2, 1): [8], (2, 0): [9], (0, 2): [9]}
    for (i, j), e in paths.items():
        idx.pair_to_edges[(i, j)] = np.array(e, dtype=np.int64)
        for k in e:
            idx.edge_to_pairs.setdefault(int(k), set()).add((i, j))
    state = FleetState.from_fleet_route(FleetRoute([np.array([0, 1, 2, 0])], np.array([0, 0])))

    assert mod.affected_vehicles(state, idx, [7]) == {0}  # still ahead of the vehicle
    state.advance(0)  # drives 0 -> 1; edge 7 is now history
    assert state.remaining_legs(0) == [(1, 2), (2, 0)]
    assert mod.affected_vehicles(state, idx, [7]) == set()
    assert mod.affected_vehicles(state, idx, [8]) == {0}  # the leg still ahead does trigger


# -- F7: the other two ladder boundaries ---------------------------------------------------
def test_exactly_theta_hard_takes_the_persistence_branch_not_the_hard_trigger():
    """The ladder is `theta_soft < Delta <= theta_hard -> persistence` [SPEC 9.2], so the
    boundary value must arm rather than fire."""
    state = three_vehicle_state()
    ctrl = ReplanController(QTrafficConfig())
    first = ctrl.decide(0.0, QTrafficConfig().theta_hard, [0], 3, state)
    assert first.scope is Scope.NONE and "armed 1/2" in first.reason
    assert ctrl.decide(1.0, QTrafficConfig().theta_hard, [0], 3, state).scope is Scope.LOCAL


def test_exactly_theta_override_is_a_hard_trigger_not_a_severe_override():
    """`theta_hard < Delta <= theta_override -> hard trigger` [SPEC 9.2]: the boundary
    belongs to the hard branch, so the cooldown still applies to it."""
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    d = ReplanController(cfg).decide(0.0, cfg.theta_override, [0], 3, state)
    assert d.reason == "above theta_hard" and d.override is False

    state.last_reopt[:] = 0.0  # now cooling: a true override would ignore this, a hard one holds
    held = ReplanController(cfg).decide(1.0, cfg.theta_override, [0], 3, state)
    assert held.scope is Scope.NONE and "cooldown" in held.reason


# -- F4: the set-wide cooldown veto is deliberate, pinned as such ---------------------------
def test_a_single_cooling_vehicle_holds_the_whole_decision_by_design():
    """Deliberate reading of [v4 47]: the cooldown CLOCK is per vehicle, the GATE is
    set-wide. A LOCAL re-plan re-optimises its set together, so firing for a subset while
    one member is frozen is the churn `T_cool` exists to prevent. Pinned so this is not
    re-flagged as an oversight; if the spec says otherwise, this test changes with it."""
    cfg = QTrafficConfig()
    state = three_vehicle_state()
    state.last_reopt[:] = -np.inf
    state.last_reopt[1] = 1000.0  # only vehicle 1 is cooling
    ctrl = ReplanController(cfg)
    assert ctrl.cooldown_ok(state, [0, 2], 1001.0)  # the others are individually clear
    held = ctrl.decide(1001.0, 0.20, [0, 1], 3, state)
    assert held.scope is Scope.NONE and held.reason == "hard trigger held by cooldown"
    assert held.vehicles == (0, 1)  # the full set is reported, none of it fired
    # and it fires for the whole set once the last member clears
    fired = ctrl.decide(1000.0 + cfg.T_cool, 0.20, [0, 1], 3, state)
    assert fired.scope is Scope.FLEET  # 2 of 3 vehicles is past local_scope_fraction
    assert fired.vehicles == (0, 1)
