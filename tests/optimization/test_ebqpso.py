"""Phase 0 stub test: only asserts the module imports. Flipped to real tests when its body lands."""

import math

import numpy as np
import pytest

import backend.optimization.ebqpso as mod
from backend.config import QTrafficConfig
from backend.constraints.checker import check_all
from backend.optimization import encoding as enc
from backend.optimization import qpso as qpso_mod
from backend.optimization.benchmark import load_scenario
from backend.optimization.fitness import Bounds, ProblemContext, TrafficState
from tests.constraints.helpers import make_problem

S1_BUDGET = 2.0  # s; the 5 s x 5 seeds run is the exit-condition script, not the suite
SEEDS = [0, 1, 2, 3, 4]


def toy(n=12, n_veh=3, seed=0):
    """Same small synthetic instance the Phase 7 tests use."""
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


def optimizer(cls=mod.EBQPSO, seed=0, **over):
    ctx, prob = toy()
    cfg = QTrafficConfig(**over)
    rng = np.random.default_rng(seed)
    ev = qpso_mod.Evaluator(ctx, prob, cfg)
    return cls(cfg, ev, enc.dim(ctx.n_customers), rng), ctx, prob


# -- v4 29 elite set ---------------------------------------------------------------------
def test_elite_set_size_and_order():
    """E_t = top_k(pbest), N_E = ceil(r_E * M), best first, no repeats [v4 29]."""
    opt, _, _ = optimizer(M=20)
    opt.initialize()
    opt.state.pbest_F = np.random.default_rng(0).permutation(20).astype(float)
    e = opt.select_elites()
    assert len(e) == math.ceil(opt.cfg.r_E * 20) == 3
    assert len(set(e.tolist())) == len(e)  # distinct particles
    F = opt.state.pbest_F
    assert list(F[e]) == sorted(F[e])  # sorted ascending (best first)
    assert F[e].max() <= np.sort(F)[len(e) - 1]  # really the top_k, not just any 3


def test_elite_set_is_at_least_two_for_breeding_to_be_possible():
    opt, _, _ = optimizer(M=50)
    opt.initialize()
    assert len(opt.select_elites()) >= 2  # r_E = 0.15 of 50 -> 8


# -- v4 30-32 breeding -------------------------------------------------------------------
def test_breed_count_bounds_and_parent_distinctness():
    """N_child = ceil(r_B * M) children, all in [0,1]^2N; parents always distinct [v4 30]."""
    opt, ctx, _ = optimizer(M=50)
    opt.initialize()
    kids = opt.breed(opt.select_elites())
    assert kids.shape == (math.ceil(opt.cfg.r_B * 50), enc.dim(ctx.n_customers))
    assert kids.min() >= 0.0 and kids.max() <= 1.0  # [v4 31] clip

    # A != B: with two elites pinned to opposite corners, a child bred from A and B lands
    # strictly between them; a child bred from A twice would sit on top of a corner.
    opt.state.pbest_F = np.full(50, 10.0)
    opt.state.pbest_F[:2] = [0.0, 1.0]
    opt.state.pbest[0], opt.state.pbest[1] = 0.0, 1.0
    cfg_no_noise = optimizer(M=50, sigma_A=0.0, sigma_O=0.0)[0]
    cfg_no_noise.initialize()
    cfg_no_noise.state.pbest_F = np.full(50, 10.0)
    cfg_no_noise.state.pbest_F[:2] = [0.0, 1.0]
    cfg_no_noise.state.pbest[0], cfg_no_noise.state.pbest[1] = 0.0, 1.0
    for _ in range(20):
        kids = cfg_no_noise.breed(np.array([0, 1]))
        # a blend of the 0-vector and the 1-vector is constant per block, never a corner
        assert (kids > 0).any() and (kids < 1).any()


def test_breed_blocks_use_separate_lambdas_and_sigmas():
    """Y and Z are blended independently [v4 31]: with no noise, each block of a child is
    a single constant (its own lambda), and the two blocks differ."""
    opt, ctx, _ = optimizer(M=50, sigma_A=0.0, sigma_O=0.0)
    opt.initialize()
    n = ctx.n_customers
    opt.state.pbest[0], opt.state.pbest[1] = 0.0, 1.0  # the only two parents: 0s and 1s
    kids = opt.breed(np.array([0, 1]))  # every child is a blend of those two
    for kid in kids:
        assert np.allclose(kid[:n], kid[0])  # constant across the Y block
        assert np.allclose(kid[n:], kid[n])  # constant across the Z block
    assert not np.allclose(kids[:, 0], kids[:, n])  # lambda_A and lambda_O are separate draws


def test_breed_noise_respects_sigma():
    """sigma_A / sigma_O scale the Gaussian term [v4 31]; sigma = 0 removes it exactly."""
    opt, _, _ = optimizer(M=50, sigma_A=0.0, sigma_O=0.0)
    opt.initialize()
    opt.state.pbest[:] = 0.5  # every parent identical -> any blend is exactly 0.5
    assert np.allclose(opt.breed(np.arange(8)), 0.5)

    noisy, _, _ = optimizer(M=50, sigma_A=0.05, sigma_O=0.05)
    noisy.initialize()
    noisy.state.pbest[:] = 0.5
    assert not np.allclose(noisy.breed(np.arange(8)), 0.5)


def test_breed_is_a_noop_when_breeding_is_disabled():
    """r_B = 0 -> no children AND no random draws (the ablation's whole basis) [v4 32]."""
    opt, _, _ = optimizer(M=50, r_B=0.0)
    opt.initialize()
    before = opt.state.rng.bit_generator.state["state"]["state"]
    kids = opt.breed(opt.select_elites())
    assert kids.shape[0] == 0
    assert opt.state.rng.bit_generator.state["state"]["state"] == before


# -- v4 34-35 replacement ----------------------------------------------------------------
def test_replacement_keeps_population_at_M_and_only_accepts_better_children():
    opt, ctx, _ = optimizer(M=20)
    opt.initialize()
    s = opt.state
    D = enc.dim(ctx.n_customers)
    F = np.arange(20, dtype=float)  # particle 19 is the worst
    s.pbest_F = F.copy()

    good = np.full((2, D), 0.25)
    batch = qpso_mod.EvalBatch(
        F=np.array([-1.0, -2.0]),
        fleets=[opt._best_fleet, opt._best_fleet],
        penalty=np.zeros(2),
        repair_distance=np.zeros(2),
        repair_iterations=np.zeros(2),
        capped_out=np.array([False, False]),
    )
    accepted = opt.replace_weak(good, batch, F)
    assert accepted == 2
    assert len(s.X) == opt.cfg.M == 20  # [v4 34] population never grows
    assert np.allclose(s.X[19], 0.25) and np.allclose(s.X[18], 0.25)
    assert s.gbest_F == -2.0 and np.allclose(s.gbest, 0.25)  # best child took gbest

    # a child worse than its target is discarded, elite parentage notwithstanding [v4 35]
    before = s.X.copy()
    bad = np.full((2, D), 0.9)
    batch.F = np.array([1e9, 1e9])
    assert opt.replace_weak(bad, batch, F) == 0
    assert np.array_equal(s.X, before) and len(s.X) == 20


def test_population_size_is_constant_every_iteration():
    """assert len(X) == M after every replacement step, as the correctness bar demands."""
    opt, _, _ = optimizer(M=20)
    sizes = []
    real = opt.replace_weak

    def spy(children, child_batch, F):
        out = real(children, child_batch, F)
        sizes.append(len(opt.state.X))
        return out

    opt.replace_weak = spy
    res = opt.run(T=8)
    assert sizes and set(sizes) == {20}
    assert len(opt.state.X) == 20 and len(opt.state.pbest) == 20 == len(opt.state.pbest_F)
    assert res.diagnostics["children_bred"] > 0


# -- v4 28 block-aware alpha --------------------------------------------------------------
def test_block_alpha_is_per_block_when_enabled_and_scalar_when_off():
    opt, ctx, _ = optimizer(M=10, alpha_max_order=0.9, alpha_min_order=0.2)
    n = ctx.n_customers
    a = opt.alpha(0, 100)
    assert isinstance(a, np.ndarray) and a.shape == (enc.dim(n),)
    assert np.allclose(a[:n], opt.cfg.alpha_max) and np.allclose(a[n:], 0.9)
    late = opt.alpha(100, 100)
    assert np.allclose(late[:n], opt.cfg.alpha_min) and np.allclose(late[n:], 0.2)

    off, _, _ = optimizer(M=10, block_alpha=False)
    assert isinstance(off.alpha(0, 100), float)


def test_block_alpha_defaults_match_the_shared_schedule():
    """Enabled-with-defaults must be numerically identical to disabled [v4 28]."""
    opt, _, _ = optimizer(M=10)
    assert np.allclose(opt.alpha(37, 100), QPSO_alpha(opt, 37, 100))


def QPSO_alpha(opt, t, T):
    return qpso_mod.QPSO.alpha(opt, t, T)


# -- the ablation guard: EB-QPSO with breeding off IS plain QPSO ---------------------------
def test_ebqpso_without_breeding_is_identical_to_qpso():
    """Strongest regression guard [v4 32 ablation]: same seed, r_B = 0 -> same numbers."""
    a = optimizer(cls=qpso_mod.QPSO, seed=11)[0].run(T=12)
    b = optimizer(cls=mod.EBQPSO, seed=11, r_B=0.0)[0].run(T=12)
    assert np.array_equal(a.history, b.history)
    assert np.array_equal(a.gbest, b.gbest)
    assert a.gbest_fitness == b.gbest_fitness
    assert a.iterations == b.iterations and a.stopped_by == b.stopped_by


def test_breeding_on_changes_the_trajectory():
    """Sanity counterpart: with breeding ON the run is NOT the plain-QPSO run."""
    a = optimizer(cls=qpso_mod.QPSO, seed=11)[0].run(T=12)
    b = optimizer(cls=mod.EBQPSO, seed=11)[0].run(T=12)
    assert not np.array_equal(a.history, b.history)


# -- 8.3 non-write-back, inherited ---------------------------------------------------------
def test_children_are_raw_keys_and_gbest_stays_decodable():
    opt, _, _ = optimizer(M=20)
    res = opt.run(T=10)
    s = opt.state
    for name in ("X", "pbest", "gbest", "mbest"):
        v = getattr(s, name)
        assert v.dtype == float and v.min() >= 0.0 and v.max() <= 1.0
    assert np.isclose(opt.evaluate(s.gbest[None, :]).F[0], res.gbest_fitness)


def test_run_is_reproducible_for_a_seed():
    a = optimizer(seed=5)[0].run(T=8)
    b = optimizer(seed=5)[0].run(T=8)
    assert np.array_equal(a.history, b.history) and np.array_equal(a.gbest, b.gbest)


# -- correctness bar: S1, 5 seeds -----------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_s1_budgeted_ebqpso_run_is_feasible_monotone_and_fixed_size(seed):
    cfg = QTrafficConfig()
    rng = np.random.default_rng(seed)
    ctx, prob = load_scenario("S1", cfg, rng)
    opt = mod.EBQPSO(cfg, qpso_mod.Evaluator(ctx, prob, cfg), enc.dim(ctx.n_customers), rng)
    res = opt.run(time_budget_s=S1_BUDGET)
    assert (np.diff(res.history) <= 1e-12).all()  # monotone per seed
    assert len(opt.state.X) == cfg.M  # [v4 34] population unchanged
    assert res.elapsed_s <= S1_BUDGET * 1.1
    assert check_all(res.fleet, prob).ok and res.feasible
    assert res.capped_out is False
    assert np.isfinite(res.gbest_fitness)
    d = res.diagnostics
    assert d["children_bred"] >= 1 and 0.0 <= d["child_acceptance_rate"] <= 1.0
    assert 0.0 <= d["child_repair_failure_rate"] <= 1.0  # tracked apart from the swarm's
