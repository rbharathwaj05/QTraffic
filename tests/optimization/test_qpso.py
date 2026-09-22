import time

import numpy as np
import pytest

from backend.config import QTrafficConfig
from backend.constraints.checker import check_all
from backend.optimization import encoding as enc
from backend.optimization import qpso as mod
from backend.optimization.benchmark import load_scenario
from backend.optimization.fitness import Bounds, ProblemContext, TrafficState
from tests.constraints.helpers import fleet, make_problem

S1_BUDGET = 2.0  # s; the 5 s x 5 seeds run is the exit-condition script, not the suite
SEEDS = [0, 1, 2, 3, 4]


def toy(n=12, n_veh=3, seed=0):
    """Small synthetic instance: euclidean durations, wide windows, ample capacity."""
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(0, 1, n + 1), rng.uniform(0, 1, n + 1)
    D = np.hypot(lat[:, None] - lat, lon[:, None] - lon) * 1000.0
    prob = make_problem(
        rng.integers(1, 5, n),
        np.full(n_veh, 40.0),
        duration=D,
        windows=np.column_stack([np.zeros(n), np.full(n, 1e9)]),
        service=np.full(n, 30.0),
        shift_end=1e9,
        rho_max=0.95,
    )
    traffic = TrafficState(D, D, np.ones_like(D))
    ctx = ProblemContext(traffic, Bounds((0.0, 1e4), (0.0, 1e4), (0.0, 1e4)), n, n_veh)
    return ctx, prob


def optimizer(seed=0, **over):
    ctx, prob = toy()
    cfg = QTrafficConfig(**over)
    rng = np.random.default_rng(seed)
    return mod.QPSO(cfg, mod.Evaluator(ctx, prob, cfg), enc.dim(ctx.n_customers), rng), ctx, prob


# -- 7.2 local attractor: pinned spec worked example --------------------------------
def test_local_attractor_worked_example():
    """[SPEC 7.4] pbest = 0.30, gbest = 0.80, phi = 0.6 -> p = 0.6*0.3 + 0.4*0.8 = 0.50."""
    opt, _, _ = optimizer()
    opt.initialize()
    s = opt.state
    s.pbest[:] = 0.30
    s.gbest[:] = 0.80

    class Fixed:  # phi pinned to 0.6, everything else untouched
        def random(self, shape):
            return np.full(shape, 0.6)

    s.rng = Fixed()
    assert np.allclose(opt.local_attractor(), 0.50)


def test_local_attractor_is_per_dimension_and_fresh():
    opt, _, _ = optimizer()
    opt.initialize()
    p1, p2 = opt.local_attractor(), opt.local_attractor()
    assert p1.shape == opt.state.X.shape and not np.allclose(p1, p2)  # redrawn every call
    # phi in [0,1] -> p lies between pbest and gbest elementwise
    lo = np.minimum(opt.state.pbest, opt.state.gbest)
    hi = np.maximum(opt.state.pbest, opt.state.gbest)
    assert ((p1 >= lo - 1e-12) & (p1 <= hi + 1e-12)).all()


# -- 7.3 mbest ------------------------------------------------------------------------
def test_mbest_is_mean_of_raw_key_pbest():
    opt, _, _ = optimizer()
    opt.initialize()
    s = opt.state
    assert np.allclose(opt.mean_best(), s.pbest.mean(axis=0)) and opt.mean_best().shape == (24,)
    s.pbest[:] = np.linspace(0, 1, s.pbest.size).reshape(s.pbest.shape)
    assert np.allclose(opt.mean_best(), s.pbest.mean(axis=0))
    assert s.pbest.dtype == float and 0.0 <= s.pbest.min() and s.pbest.max() <= 1.0


# -- 7.4 position update --------------------------------------------------------------
def test_position_update_formula_and_clip():
    opt, _, _ = optimizer()
    opt.initialize()
    s = opt.state
    s.mbest = opt.mean_best()
    X0, alpha = s.X.copy(), 0.7
    rng_state = np.random.default_rng(99)
    s.rng = np.random.default_rng(99)
    opt.update_positions(alpha)
    # replay the exact draw order: phi, u, sign
    phi = rng_state.random(X0.shape)
    p = phi * s.pbest + (1 - phi) * s.gbest
    u = rng_state.random(X0.shape)
    sign = np.where(rng_state.random(X0.shape) < 0.5, -1.0, 1.0)
    want = np.clip(p + sign * alpha * np.abs(s.mbest - X0) * np.log(1 / u), 0, 1)
    assert np.allclose(s.X, want)
    assert s.X.min() >= 0.0 and s.X.max() <= 1.0  # [v4 26] clip


def test_step_is_unbounded_long_tail_not_truncated():
    """ln(1/u) must be able to produce large steps -- that is the escape mechanism."""
    opt, _, _ = optimizer()
    opt.initialize()
    s = opt.state
    s.mbest = np.full(s.X.shape[1], 0.5)
    s.X[:] = 0.5
    s.pbest[:] = 0.5
    s.gbest[:] = 0.5
    s.X[:] = 0.0  # |mbest - x| = 0.5 everywhere
    moved = []
    for _ in range(50):
        opt.update_positions(1.0)
        moved.append(np.abs(s.X - 0.5).max())
        s.X[:] = 0.0
    assert max(moved) >= 0.5 - 1e-9  # some draw saturates the clip: no truncation


# -- 7.5 alpha ------------------------------------------------------------------------
def test_alpha_schedule_endpoints_and_projection():
    opt, _, _ = optimizer()
    cfg = opt.cfg
    assert opt.alpha(0, 100) == cfg.alpha_max
    assert np.isclose(opt.alpha(50, 100), (cfg.alpha_max + cfg.alpha_min) / 2)
    assert np.isclose(opt.alpha(100, 100), cfg.alpha_min)
    assert opt.alpha(500, 100) == cfg.alpha_min  # clamped past the projection


def test_alpha_uses_projected_T_under_a_budget():
    opt, _, _ = optimizer(M=8)
    res = opt.run(T=1000, time_budget_s=0.6)
    T_proj = res.diagnostics["T_projected"]
    assert 0 < T_proj < 1000  # projected from timings, not the raw iteration cap
    assert res.diagnostics["alpha_final"] <= opt.cfg.alpha_max


# -- 7.6 stagnation / diversity -------------------------------------------------------
def test_stagnation_and_injection():
    opt, _, _ = optimizer(M=10)
    opt.initialize()
    p = opt.cfg.patience
    assert not opt.stagnated([1.0] * p)  # window not full yet
    assert opt.stagnated([1.0] * (p + 1))  # flat -> stagnated
    assert not opt.stagnated([1.0] * p + [0.5])  # big improvement -> not stagnated

    s = opt.state
    s.pbest_F = np.arange(10, dtype=float)
    before, keep = s.X.copy(), int(np.argmin(s.pbest_F))
    k = opt.inject_diversity()
    assert k == max(1, round(opt.cfg.r_inject * 10))
    assert np.array_equal(s.X[keep], before[keep])  # best particles untouched
    assert np.isinf(s.pbest_F[np.argsort(np.arange(10, dtype=float))[-k:]]).all()
    assert (s.X >= 0).all() and (s.X <= 1).all()


def test_diversity_metric():
    opt, _, _ = optimizer()
    opt.initialize()
    opt.state.X[:] = 0.5
    assert opt.diversity() == 0.0
    opt.state.X[:] = np.random.default_rng(0).random(opt.state.X.shape)
    assert opt.diversity() > 0.0


# -- 7.5 two-opt polish ----------------------------------------------------------------
def test_two_opt_improves_and_never_touches_keys():
    # 4 customers on a line at x = 1..4; the crossing order 1,3,2,4 must untangle
    d = np.abs(np.arange(5)[:, None] - np.arange(5)) * 100.0
    prob = make_problem([1, 1, 1, 1], [100], duration=d)
    f = fleet([1, 3, 2, 4])
    cfg = QTrafficConfig()
    g = mod.two_opt(f, prob, cfg)
    assert g.routes[0].tolist() == [0, 1, 2, 3, 4, 0]
    assert mod.route_duration(g.routes[0], prob) < mod.route_duration(f.routes[0], prob)
    assert f.routes[0].tolist() == [0, 1, 3, 2, 4, 0]  # input untouched
    # a move that would break a time window is rejected
    tight = make_problem([1, 1, 1, 1], [100], duration=d, windows=[[0, 1e9]] * 3 + [[0, 250]])
    assert mod.two_opt(fleet([1, 3, 2, 4]), tight, cfg).routes[0].tolist() == [0, 1, 3, 2, 4, 0]
    assert mod.two_opt(f, prob, QTrafficConfig(two_opt_max_passes=0)).routes[0].tolist() == [
        0,
        1,
        3,
        2,
        4,
        0,
    ]


# -- 8.3 non-write-back -----------------------------------------------------------------
def test_swarm_state_is_raw_keys_only():
    opt, ctx, prob = optimizer()
    res = opt.run(T=6)
    s = opt.state
    for name in ("X", "pbest", "gbest", "mbest"):
        v = getattr(s, name)
        assert isinstance(v, np.ndarray) and v.dtype == float
        assert v.min() >= 0.0 and v.max() <= 1.0  # still in [0,1]^2N
    assert np.array_equal(res.gbest, s.gbest)
    # the reported (repaired + polished) fleet is NOT the decode of gbest, and gbest
    # never learned about it
    g_before = s.gbest.copy()
    mod.two_opt(res.fleet, prob, opt.cfg)
    assert np.array_equal(s.gbest, g_before)
    # gbest keys still evaluate to gbest_fitness: nothing was written back
    assert np.isclose(opt.evaluate(s.gbest[None, :]).F[0], res.gbest_fitness)


def test_run_is_reproducible_for_a_seed():
    a = optimizer(seed=7)[0].run(T=8)
    b = optimizer(seed=7)[0].run(T=8)
    assert np.array_equal(a.history, b.history) and np.array_equal(a.gbest, b.gbest)


def test_gbest_is_monotone_non_increasing():
    res = optimizer(seed=3)[0].run(T=15)
    assert (np.diff(res.history) <= 1e-12).all()
    assert res.history[0] >= res.history[-1] == res.gbest_fitness
    assert len(res.wallclock) == len(res.history) == res.iterations + 1
    assert (np.diff(res.wallclock) >= 0).all()


# -- correctness bar: S1, 5 seeds -------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_s1_budgeted_run_is_monotone_feasible_and_on_time(seed):
    cfg = QTrafficConfig()
    rng = np.random.default_rng(seed)
    ctx, prob = load_scenario("S1", cfg, rng)
    opt = mod.QPSO(cfg, mod.Evaluator(ctx, prob, cfg), enc.dim(ctx.n_customers), rng)
    res = opt.run(time_budget_s=S1_BUDGET)
    assert (np.diff(res.history) <= 1e-12).all()  # monotone per seed
    assert res.elapsed_s <= S1_BUDGET * 1.1  # budget + 10 % tolerance
    assert res.iterations >= 1 and res.stopped_by in {"budget", "iterations", "stagnation"}
    assert check_all(res.fleet, prob).ok and res.feasible  # deployed route is feasible
    assert res.capped_out is False  # a capped gbest would be surfaced here, not hidden
    assert np.isfinite(res.gbest_fitness)


def test_s1_iteration_throughput_baseline():
    """Documented Phase 7 baseline future phases must not regress: M=50 on S1."""
    cfg = QTrafficConfig()
    rng = np.random.default_rng(0)
    ctx, prob = load_scenario("S1", cfg, rng)
    opt = mod.QPSO(cfg, mod.Evaluator(ctx, prob, cfg), enc.dim(ctx.n_customers), rng)
    t = time.perf_counter()
    res = opt.run(T=20)
    dt = time.perf_counter() - t
    rate = res.iterations / dt
    print(
        f"\nQPSO S1 M=50: {rate:.1f} iterations/s -> >= {int(rate * 5)} iterations in a 5 s budget"
    )
    assert rate >= 5.0  # >= 25 iterations inside the 5 s local budget [SPEC 8]


# -- 7.6 regression: an injection must not be cancelled by its own plateau ---------------
def test_stagnation_is_suppressed_for_a_full_cooldown_after_injection():
    """The traced Phase 7 bug: the patience window still held the pre-injection plateau,
    so `stagnated()` re-fired on the next iteration and killed the run one iteration after
    the injection meant to rescue it."""
    opt, _, _ = optimizer(M=10)
    opt.initialize()
    p = opt.cfg.patience
    flat = [1.0] * (p + 1)
    assert opt.stagnated(flat)  # a plateau flags stagnation before any injection

    opt.inject_diversity()
    assert opt.state.injection_cooldown == p
    for i in range(p):  # every iteration of the cooldown: suppressed, plateau or not
        assert not opt.stagnated(flat + [1.0] * i), f"stagnation fired {i} it after injection"
        opt.state.injection_cooldown = max(0, opt.state.injection_cooldown - 1)
    assert opt.state.injection_cooldown == 0
    assert opt.stagnated(flat)  # cooldown spent -> detection live again


@pytest.mark.parametrize("seed", SEEDS)
def test_injection_is_followed_by_meaningful_work_not_an_immediate_exit(seed):
    """Budget-utilisation guard: after the first injection the run must keep going for at
    least the cooldown, and must not exit having spent a token fraction of its budget."""
    cfg = QTrafficConfig()
    rng = np.random.default_rng(seed)
    ctx, prob = load_scenario("S1", cfg, rng)
    opt = mod.QPSO(cfg, mod.Evaluator(ctx, prob, cfg), enc.dim(ctx.n_customers), rng)

    marks, it = [], {"t": 0}
    base_update, base_inject = opt.update_positions, opt.inject_diversity
    opt.update_positions = lambda a: (it.__setitem__("t", it["t"] + 1), base_update(a))[1]
    opt.inject_diversity = lambda: (marks.append(it["t"]), base_inject())[1]

    res = opt.run(time_budget_s=S1_BUDGET)
    if not marks:
        pytest.skip("no injection fired inside this budget; nothing to assert about the gap")
    gap = res.iterations - marks[0]
    # either the run kept iterating through the whole cooldown, or it stopped because the
    # budget really was spent -- what must never happen again is exiting ~1 iteration after
    # the injection with most of the budget still unused.
    assert gap >= cfg.patience or res.elapsed_s >= 0.8 * S1_BUDGET, (
        f"exited {gap} iterations after the injection having used "
        f"{res.elapsed_s:.2f}s of {S1_BUDGET}s"
    )
    assert res.stopped_by != "stagnation" or gap >= cfg.patience
