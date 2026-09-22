"""Phase 6: arrival recurrence waits when early and is late when a window closes;
load ratio / capacity predicates; insertion_scan lateness + detour cost per slot,
availability folded in, inf for unroutable legs, capacity independent of slot."""

import numpy as np

from backend.constraints import feasibility as mod
from tests.constraints.helpers import make_problem


def test_arrival_recurrence_waits_when_early():
    # legs 10 s, service 5 s; customer 2 opens at 100 -> wait; customer 3 closes at 120
    p = make_problem([1, 1, 1], 10, service=[5, 5, 5], windows=[[0, 1e9], [100, 1e9], [0, 120]])
    r = np.array([0, 1, 2, 3, 0])
    A = mod.arrival_times(r, p.duration, p.service, p.windows, 0.0)
    assert A.tolist() == [0, 10, 100, 115, 130]  # A_k = max(e_k, A_prev + s + c)
    assert mod.lateness(r, p.duration, p.service, p.windows, 0.0).tolist() == [0, 0, 0, 0, 0]
    assert mod.is_time_feasible(r, p.duration, p.service, p.windows, 0.0)
    w = p.windows.copy()
    w[3, 1] = 112.0
    assert not mod.is_time_feasible(r, p.duration, p.service, w, 0.0)  # A_3 = 115 > 112
    assert mod.lateness(r, p.duration, p.service, w, 0.0).tolist() == [0, 0, 0, 3, 0]
    assert mod.route_end(r, p.duration, p.service, p.windows, 0.0) == 130


def test_capacity_predicates():
    p = make_problem([30, 30, 25, 30], 100)
    assert mod.load_ratio(np.array([0, 1, 2, 3, 4, 0]), p.demand, 100) == 1.15
    assert not mod.is_capacity_feasible(np.array([0, 1, 2, 3, 4, 0]), p.demand, 100, 0.95)
    assert mod.is_capacity_feasible(np.array([0, 1, 2, 3, 0]), p.demand, 100, 0.95)


def test_insertion_scan_and_can_insert():
    # customer 3 must be visited by t=15: only the slot right after the depot works
    p = make_problem([1, 1, 1], 10, windows=[[0, 1e9], [0, 1e9], [0, 15]])
    r = np.array([0, 1, 2, 0])
    late, cost, cap_ok = mod.insertion_scan(r, 3, p, 0)
    assert cap_ok and late.shape == cost.shape == (3,)
    assert late.tolist() == [0, 5, 15]  # arrival 10 / 20 / 30 vs l=15
    assert (cost == 10).all()  # c_{a,3} + c_{3,b} - c_{a,b} with all legs 10
    assert mod.can_insert(r, 1, 3, p, 0) and not mod.can_insert(r, 2, 3, p, 0)
    # availability folds into lateness at the depot return (4 legs = 40 > 35)
    p2 = make_problem([1, 1, 1], 10, shift_end=35)
    assert mod.insertion_scan(r, 3, p2, 0)[0].tolist() == [5, 5, 5]
    # unroutable leg -> inf
    d = np.full((4, 4), 10.0)
    d[2, 3] = np.inf
    p3 = make_problem([1, 1, 1], 10, duration=d)
    assert mod.insertion_scan(r, 3, p3, 0)[0].tolist() == [0, 0, np.inf]
    # capacity is a scalar test, independent of the slot
    assert not mod.insertion_scan(r, 3, make_problem([5, 5, 5], 10), 0)[2]
