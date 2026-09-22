"""Phase 6: SPEC 8.3 worked example (eject C4), every move type on a minimal case,
cascade caps out with a soft penalty that flows through fitness.evaluate, the
signature has no particle argument, input is never mutated, no `while` in the module,
all three ablation strategies, and a 25-particle fuzz that always terminates + covers."""

import inspect
import re
import threading

import numpy as np
import pytest

from backend.config import QTrafficConfig
from backend.constraints import repair as mod
from backend.constraints.checker import check_all
from backend.optimization import encoding as enc
from backend.optimization import fitness
from tests.constraints.helpers import fleet, make_problem

CFG = QTrafficConfig()


def routes(res):
    return [r.tolist() for r in res.fleet.routes]


# -- SPEC 8.3 doc 2 worked example ------------------------------------------------------
def test_overload_worked_example_c1_c2_c3_c4_removes_c4():
    """100 kg vehicle (rho_max 0.95 -> 95) loaded 110: remove C4, reinsert elsewhere."""
    p = make_problem([30, 30, 20, 30], [100, 100], rho_max=CFG.rho_max)
    res = mod.repair(fleet([1, 2, 3, 4], []), p, CFG)
    assert routes(res) == [[0, 1, 2, 3, 0], [0, 4, 0]]
    assert res.iterations == 1 and not res.capped_out and res.residual == 0.0
    assert res.moves == ["capacity: eject 4 from v0@3 -> v1@0"]
    # d_R = eta_a * 1/4 (C4 changes vehicle) + eta_o * |3 - 0| / 4 (C4 pos 3 -> 0) [v4 23]
    assert np.isclose(res.distance, CFG.eta_a / 4 + CFG.eta_o * 3 / 4)
    assert res.distances == [res.distance]
    assert res.fleet.assignment.tolist() == [0, 0, 0, 1]


def test_missing_customer_inserted_closest_to_decoded_position():
    """Decoded says 2 sits at v0@1; a fleet missing it gets it back exactly there, not at
    the (equally cheap) slots v0@0 / v0@2 / v1@*."""
    p = make_problem([1, 1, 1, 1], [100, 100])
    decoded = fleet([1, 2, 3], [4])
    f = fleet([1, 3], [4], n=4)
    vio = check_all(f, p).first
    assert vio.kind == "coverage" and vio.customers == (2,)
    g, msg = mod._move_coverage(f, vio, p, CFG, "minimal", mod.positions(decoded, 4))
    assert [r.tolist() for r in g.routes] == [[0, 1, 2, 3, 0], [0, 4, 0]]
    assert msg == "coverage: insert 2 -> v0@1" and g.assignment.tolist() == [0, 0, 0, 1]
    # end-to-end through repair(): no reference -> still covered, still feasible
    res = mod.repair(f, p, CFG)
    assert res.report.ok and res.iterations == 1 and res.moves[0].startswith("coverage: insert 2")


def test_duplicate_drops_occurrence_further_from_decoded():
    p = make_problem([1, 1, 1, 1], [100, 100])
    f = fleet([1, 2, 3], [4])
    f.routes[1] = np.array([0, 4, 2, 0])  # 2 also on v1
    res = mod.repair(f, p, CFG)
    assert routes(res) == [[0, 1, 2, 3, 0], [0, 4, 0]] and res.fleet.assignment[1] == 0


def test_time_window_smallest_reordering():
    # customer 3 must be reached by t=15 -> only as first stop (legs 10 s)
    p = make_problem([1, 1, 1], [100, 100], windows=[[0, 1e9], [0, 1e9], [0, 15]])
    res = mod.repair(fleet([1, 2, 3], []), p, CFG)
    assert routes(res) == [[0, 3, 1, 2, 0], [0, 0]]
    assert res.moves == ["time_window: reorder v0 d_O=4"] and res.residual == 0.0


def test_availability_ejects_last_stop():
    p = make_problem([1, 1, 1], [100, 100], shift_end=35)  # [1,2,3] returns at 40
    res = mod.repair(fleet([1, 2, 3], []), p, CFG)
    assert routes(res) == [[0, 1, 2, 0], [0, 3, 0]] and res.moves[0].startswith("availability")


def test_depot_rewrap():
    p = make_problem([1, 1], [100])
    f = fleet([1, 2])
    f.routes[0] = np.array([0, 1, 0, 2, 0])
    res = mod.repair(f, p, CFG)
    assert routes(res) == [[0, 1, 2, 0]] and res.moves == ["depot: rewrap v0"]


def test_connectivity_relocates_far_end_stop():
    d = np.full((4, 4), 10.0)
    d[2, 3] = np.inf
    p = make_problem([1, 1, 1], [100, 100], duration=d)
    res = mod.repair(fleet([1, 2, 3], []), p, CFG)
    assert routes(res) == [[0, 1, 3, 2, 0], [0, 0]] and res.moves[0].startswith("connectivity")


# -- cascade cap ---------------------------------------------------------------------------
def test_cascade_caps_out_with_soft_penalty():
    """Single vehicle, demand 120 > capacity 100: unrepairable. Must return capped_out,
    never raise, never loop forever (10 s hard backstop)."""
    p = make_problem([60, 60], [100])
    out = {}
    t = threading.Thread(target=lambda: out.update(res=mod.repair(fleet([1, 2]), p, CFG)))
    t.daemon = True
    t.start()
    t.join(10)
    assert not t.is_alive(), "repair() did not terminate within 10 s"
    res = out["res"]
    assert res.capped_out and res.iterations == CFG.MAX_REPAIR_ITERATIONS
    assert len(res.moves) == CFG.MAX_REPAIR_ITERATIONS + 1 and res.moves[-1].startswith("capped")
    assert len(res.distances) == CFG.MAX_REPAIR_ITERATIONS
    assert np.isclose(res.residual, 20 / 100)
    # soft-penalty path: F + w_p * P through fitness.evaluate
    tr = fitness.TrafficState(p.duration, p.duration, np.ones_like(p.duration))
    b = fitness.Bounds((0, 1), (0, 1), (0, 1))
    f0 = fitness.evaluate([res.fleet], tr, None, b, CFG)
    f1 = fitness.evaluate([res.fleet], tr, None, b, CFG, penalties=[res.residual])
    assert np.isclose(f1[0] - f0[0], CFG.w_p * res.residual)


# -- SPEC 8.3 non-write-back: structurally impossible -------------------------------------
def test_repair_signature_has_no_particle_parameter():
    params = inspect.signature(mod.repair).parameters
    assert list(params) == ["decoded", "prob", "cfg", "strategy"]
    assert params["decoded"].annotation == "FleetRoute"
    bad = re.compile(r"^(x|X|particle|swarm|keys?|pbest|gbest|mbest)$")
    for fn in (mod.repair, *mod.MOVES.values(), mod._best_slot, mod._relocate):
        assert not [n for n in inspect.signature(fn).parameters if bad.match(n)], fn.__name__


def test_repair_never_mutates_input():
    p = make_problem([30, 30, 20, 30], [100, 100], rho_max=0.95)
    f = fleet([1, 2, 3, 4], [])
    before = ([r.copy() for r in f.routes], f.assignment.copy())
    mod.repair(f, p, CFG)
    assert all(np.array_equal(a, b) for a, b in zip(before[0], f.routes))
    assert np.array_equal(before[1], f.assignment)


def test_no_unbounded_loops_in_repair_module():
    """Defect #8 guard: no `while` statement at all in repair.py."""
    src = inspect.getsource(mod)
    assert not re.search(r"^\s*while\b", src, re.M)


# -- ablation strategies share the interface ---------------------------------------------
@pytest.mark.parametrize("strategy", mod.STRATEGIES)
def test_strategies_all_repair_worked_example(strategy):
    p = make_problem([30, 30, 20, 30], [100, 100], rho_max=0.95)
    res = mod.repair(fleet([1, 2, 3, 4], []), p, CFG, strategy)
    assert res.report.ok and res.iterations == 1
    ejected = {"minimal": 4, "cheapest": 1, "smallest_demand": 3}[strategy]
    assert routes(res)[1] == [0, ejected, 0]


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        mod.repair(fleet([1]), make_problem([1], [10]), CFG, "magic")


# -- fuzz: decoded swarm -> every candidate terminates, covers, and is diagnosed ----------
def test_fuzz_random_swarm_terminates_and_keeps_coverage():
    rng = np.random.default_rng(6)
    n, n_veh = 30, 5
    lat, lon = rng.uniform(0, 1, n + 1), rng.uniform(0, 1, n + 1)
    dur = np.hypot(lat[:, None] - lat, lon[:, None] - lon) * 3600
    windows = np.column_stack([rng.uniform(0, 3000, n), np.full(n, 6000.0)])
    p = make_problem(
        rng.integers(1, 10, n),
        np.full(n_veh, 45.0),
        duration=dur,
        windows=windows,
        service=np.full(n, 60.0),
        shift_end=8000,
        rho_max=0.95,
    )
    for f in enc.decode(enc.random_particle(n, rng, 25), n_veh):
        res = mod.repair(f, p, CFG)
        assert res.iterations <= CFG.MAX_REPAIR_ITERATIONS
        assert res.report.ok != res.capped_out
        stops = np.concatenate([r[1:-1] for r in res.fleet.routes])
        assert sorted(stops) == list(range(1, n + 1))  # coverage always holds
        assert (res.fleet.assignment == mod.positions(res.fleet, n)[0][1:]).all()
        assert res.report.total == check_all(res.fleet, p).total
        assert 0 <= res.distance and len(res.distances) == res.iterations
