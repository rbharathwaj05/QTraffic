"""Phase 10 acceptance: the affected-subset scoping contract [SPEC v4 50-55]."""

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.feasibility import Problem
from backend.fleet.state import FleetState
from backend.optimization import ebqpso, local_scope
from backend.optimization.encoding import FleetRoute, decode
from backend.optimization.fitness import Bounds, ProblemContext, TrafficState


def big_world(n_vehicles=100, per_vehicle=3):
    """`n_vehicles` vehicles, three customers each: the v4 55 scale, literally."""
    n = n_vehicles * per_vehicle
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10_000, (n + 1, 2))
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    traffic = TrafficState(d, d, np.ones_like(d))
    prob = Problem(
        duration=d,
        demand=np.concatenate([[0.0], np.ones(n)]),
        service=np.zeros(n + 1),
        windows=np.vstack([[0.0, np.inf]] * (n + 1)),
        capacity=np.full(n_vehicles, 100.0),
        shift_end=np.full(n_vehicles, np.inf),
        t_start=np.zeros(n_vehicles),
        rho_max=0.95,
    )
    routes, assignment = [], np.zeros(n, dtype=np.int64)
    c = 1
    for v in range(n_vehicles):
        stops = list(range(c, c + per_vehicle))
        c += per_vehicle
        routes.append(np.array([0, *stops, 0], dtype=np.int64))
        for s in stops:
            assignment[s - 1] = v
    state = FleetState.from_fleet_route(FleetRoute(routes, assignment), t_sim=-np.inf)
    ctx = ProblemContext(traffic, Bounds((0.0, 1e6), (0.0, 1e6), (0.0, 1e6)), n, n_vehicles)
    return state, prob, ctx


# -- v4 55: a four-vehicle incident may not touch the other ninety-six ---------------------
def test_a_four_vehicle_incident_cannot_reach_the_other_ninety_six():
    """[v4 55] "This prevents a local rerouting problem involving four vehicles from
    accidentally assigning customers to all 100 vehicles"."""
    state, prob, ctx = big_world(n_vehicles=100)
    A = [7, 23, 55, 91]
    scope = local_scope.build_scope(state, prob, ctx, A)

    assert ctx.n_vehicles == 100  # the fleet really is 100 vehicles
    assert scope.n_vehicles == 4 == len(A)  # the subproblem is 4
    assert scope.ctx.n_vehicles == 4  # and the DECODER is told 4 [v4 55]
    assert scope.vehicles == (7, 23, 55, 91)

    expected = {int(c) for v in A for c in state.remaining_customers(v)}
    assert set(scope.customers) == expected and len(expected) == 12
    assert scope.prob.capacity.shape == (4,)
    assert scope.prob.duration.shape == (13, 13)  # depot + C_A only

    # exhaustive: no decode of ANY particle in this subproblem can name a 5th vehicle
    rng = np.random.default_rng(1)
    X = rng.random((200, 2 * scope.n_customers))
    for fleet in decode(X, scope.n_vehicles):
        assert len(fleet.routes) == 4
        assert set(np.unique(fleet.assignment)) <= {0, 1, 2, 3}


def test_local_vehicle_ids_remap_back_to_the_right_global_ids():
    state, prob, ctx = big_world(n_vehicles=100)
    A = [7, 23, 55, 91]
    scope = local_scope.build_scope(state, prob, ctx, A)
    # local vehicle k holds local customers k*3+1 .. k*3+3 in this construction
    local = FleetRoute(
        [np.array([0, 3 * k + 1, 3 * k + 2, 3 * k + 3, 0]) for k in range(4)],
        np.repeat(np.arange(4), 3),
    )
    out = scope.to_global(local, state)
    assert set(out) == {7, 23, 55, 91}  # global ids, not 0..3
    for k, v in enumerate(A):
        globals_on_route = [int(c) for c in out[v] if int(c) != 0]
        assert globals_on_route == [scope.customers[3 * k + i] for i in range(3)]
    # and every customer named is a real global matrix index inside C_A
    assert {int(c) for r in out.values() for c in r if int(c) != 0} == set(scope.customers)


def test_reoptimize_local_never_returns_a_vehicle_outside_A():
    state, prob, ctx = big_world(n_vehicles=100)
    cfg = QTrafficConfig(M=12, T_iter_min=2, T_iter_max=6)
    A = [7, 23, 55, 91]
    before = {v: state.routes[v].copy() for v in state.vehicles()}
    routes, result, scope = ebqpso.reoptimize_local(
        state, prob, ctx, A, cfg, np.random.default_rng(0), time_budget_s=1.0
    )
    assert set(routes) == set(A)
    assert result is not None and scope.n_vehicles == 4
    # the call itself deploys nothing: every global route is still untouched
    for v in state.vehicles():
        assert np.array_equal(state.routes[v], before[v])


# -- SPEC 9.3 point 1: the travelled prefix is frozen ---------------------------------------
def test_travelled_prefix_is_byte_identical_after_local_optimization():
    """R_new = R_travelled (+) R_optimized: the prefix is copied through unchanged."""
    state, prob, ctx = big_world(n_vehicles=20)
    A = [2, 5]
    for v in A:
        state.advance(v, stops=2)  # two stops already delivered
    prefixes = {v: state.routes[v][: int(state.position[v]) + 1].copy() for v in A}
    cfg = QTrafficConfig(M=10, T_iter_min=2, T_iter_max=4)
    routes, _, _ = ebqpso.reoptimize_local(state, prob, ctx, A, cfg, np.random.default_rng(0), 0.5)
    for v in A:
        p = prefixes[v]
        assert np.array_equal(routes[v][: len(p)], p), f"vehicle {v} prefix was rewritten"
        assert routes[v].dtype == p.dtype
        # and the already-served customers appear exactly once, still in the prefix
        served = [int(c) for c in p if int(c) != 0]
        tail = [int(c) for c in routes[v][len(p) :]]
        assert not set(served) & set(tail)


# -- v4 53-54 warm start --------------------------------------------------------------------
def test_warm_start_counts_are_the_spec_split_and_the_population_stays_M():
    """N_warm = 0.20 M perturbed incumbents, N_diverse = 0.80 M cold; |X| == M."""
    state, prob, ctx = big_world(n_vehicles=20)
    cfg = QTrafficConfig()
    assert (cfg.warm_fraction, cfg.diverse_fraction) == (0.20, 0.80)
    scope = local_scope.build_scope(state, prob, ctx, [1, 2, 3])
    rng = np.random.default_rng(0)
    seeds = local_scope.warm_start_particles(scope, cfg, rng)

    n_warm = int(round(cfg.warm_fraction * cfg.M))
    assert len(seeds) == n_warm == 10
    assert cfg.M - n_warm == int(round(cfg.diverse_fraction * cfg.M)) == 40

    from backend.optimization.qpso import QPSO, Evaluator

    opt = QPSO(cfg, Evaluator(scope.ctx, scope.prob, cfg), 2 * scope.n_customers, rng)
    opt.initialize(seed_particles=seeds)
    assert opt.state.X.shape == (cfg.M, 2 * scope.n_customers)  # population is exactly M
    assert np.allclose(opt.state.X[:n_warm], np.clip(seeds, 0, 1))  # warm rows are the seeds
    # the cold rows are not copies of the incumbent
    assert not np.allclose(opt.state.X[n_warm], opt.state.X[0])
    assert opt.state.X.min() >= 0.0 and opt.state.X.max() <= 1.0


def test_warm_particles_are_perturbed_variants_of_the_incumbent_not_random():
    """A warm particle must sit near X_current; a cold one must not [v4 53]."""
    state, prob, ctx = big_world(n_vehicles=20)
    cfg = QTrafficConfig()
    scope = local_scope.build_scope(state, prob, ctx, [1, 2, 3])
    seeds = local_scope.warm_start_particles(scope, cfg, np.random.default_rng(0))
    incumbent = seeds[0]
    warm_gap = np.abs(seeds[1:] - incumbent).mean()
    cold = np.random.default_rng(1).random((20, seeds.shape[1]))
    cold_gap = np.abs(cold - incumbent).mean()
    assert warm_gap < cold_gap  # warm is a neighbourhood, cold is the whole box
    assert decode(seeds[:1], scope.n_vehicles)[0].routes  # row 0 is a usable plan


def test_warm_start_is_reproducible_for_a_seed():
    """[CLAUDE.md] every stochastic component takes a Generator; same seed, same seeds."""
    state, prob, ctx = big_world(n_vehicles=20)
    cfg = QTrafficConfig()
    scope = local_scope.build_scope(state, prob, ctx, [1, 2])
    a = local_scope.warm_start_particles(scope, cfg, np.random.default_rng(7))
    b = local_scope.warm_start_particles(scope, cfg, np.random.default_rng(7))
    assert np.array_equal(a, b)


# -- SPEC 7.2: the frozen bounds stay frozen ------------------------------------------------
def test_local_bounds_are_derived_from_the_frozen_ones_not_recomputed():
    state, prob, ctx = big_world(n_vehicles=20)
    before = (ctx.bounds.t, ctx.bounds.d, ctx.bounds.c)
    scope = local_scope.build_scope(state, prob, ctx, [0, 1])
    assert (ctx.bounds.t, ctx.bounds.d, ctx.bounds.c) == before  # parent untouched
    k = scope.n_customers / ctx.n_customers
    assert scope.ctx.bounds.t[1] == ctx.bounds.t[1] * k  # pro-rated, not re-sampled
