import numpy as np
import pytest

from backend.fleet import route_manager as mod

ROUTE = np.array([0, 3, 1, 4, 0])  # depot, three customers, depot


def test_travelled_and_remaining_split_at_the_current_stop():
    """[SPEC 9.3 point 1] the junction stop belongs to both halves: it is where the
    vehicle stands, so it ends the frozen prefix and starts the remaining suffix."""
    assert mod.travelled_route(ROUTE, 0).tolist() == [0]
    assert mod.remaining_route(ROUTE, 0).tolist() == [0, 3, 1, 4, 0]
    assert mod.travelled_route(ROUTE, 2).tolist() == [0, 3, 1]
    assert mod.remaining_route(ROUTE, 2).tolist() == [1, 4, 0]


def test_position_is_clamped_to_the_route():
    assert mod.remaining_route(ROUTE, 99).tolist() == [0]
    assert mod.remaining_route(ROUTE, -5).tolist() == ROUTE.tolist()


def test_remaining_legs_are_the_od_pairs_still_to_drive():
    assert mod.remaining_legs(ROUTE, 0) == [(0, 3), (3, 1), (1, 4), (4, 0)]
    assert mod.remaining_legs(ROUTE, 2) == [(1, 4), (4, 0)]
    assert mod.remaining_legs(ROUTE, 4) == []  # parked


def test_remaining_customers_exclude_the_current_stop_and_the_depot():
    assert mod.remaining_customers(ROUTE, 0).tolist() == [3, 1, 4]
    assert mod.remaining_customers(ROUTE, 1).tolist() == [1, 4]  # 3 is being served
    assert mod.remaining_customers(ROUTE, 3).tolist() == []


def test_splice_rebuilds_a_whole_route_without_duplicating_the_junction():
    """R_new = R_travelled (+) R_optimized [SPEC 9.3 point 1]."""
    travelled = mod.travelled_route(ROUTE, 2)  # [0, 3, 1]
    optimized = np.array([1, 2, 0])  # a new plan from where the vehicle stands
    out = mod.splice(travelled, optimized)
    assert out.tolist() == [0, 3, 1, 2, 0]
    assert out.tolist()[:3] == travelled.tolist()  # the driven prefix is untouched


def test_splice_handles_disjoint_pieces_and_empty_suffixes():
    assert mod.splice(np.array([0, 3]), np.array([5, 0])).tolist() == [0, 3, 5, 0]
    assert mod.splice(np.array([0, 3]), np.empty(0, dtype=int)).tolist() == [0, 3]


def test_route_cost_terms_sum_only_the_remaining_legs():
    dur = np.full((5, 5), 10.0)
    dist = np.full((5, 5), 100.0)
    rho = np.full((5, 5), 2.0)
    np.fill_diagonal(dur, 0.0)
    np.fill_diagonal(dist, 0.0)

    T, D, C = mod.route_cost_terms(ROUTE, 0, dur, dist, rho)
    assert (T, D, C) == (40.0, 400.0, 800.0)  # four legs
    T2, D2, C2 = mod.route_cost_terms(ROUTE, 2, dur, dist, rho)
    assert (T2, D2, C2) == (20.0, 200.0, 400.0)  # two legs
    assert mod.route_cost_terms(ROUTE, 4, dur, dist, rho) == (0.0, 0.0, 0.0)


def test_route_cost_terms_never_reads_the_travelled_prefix():
    """A traffic event on an already-driven leg must not move the remaining cost."""
    dur = np.full((5, 5), 10.0)
    dist = np.full((5, 5), 100.0)
    rho = np.ones((5, 5))
    worse = dur.copy()
    worse[0, 3] = 1e6  # blow up the first leg, which has already been driven
    at_stop_2 = dict(position=2, distance=dist, rho=rho)
    a = mod.route_cost_terms(ROUTE, 2, dur, dist, rho)
    b = mod.route_cost_terms(ROUTE, 2, worse, dist, rho)
    assert a == b == pytest.approx(a)
    assert at_stop_2["position"] == 2
