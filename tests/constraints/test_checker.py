import numpy as np

from backend.constraints import checker as mod
from tests.constraints.helpers import fleet, make_problem


def test_fixed_order_and_structured_violations():
    """One fleet breaking every constraint: report lists kinds in SPEC 8.4 order."""
    d = np.full((6, 6), 10.0)
    d[4, 5] = np.inf  # leg 4 -> 5 unroutable (v0)
    p = make_problem(
        [30, 30, 25, 30, 1],
        [100, 100],
        duration=d,
        windows=[[0, 1e9], [0, 15], [0, 1e9], [0, 1e9], [0, 1e9]],
        shift_end=15,
    )
    f = fleet([1, 2, 3, 4, 5], [], n=5)
    # v1 = [0, 1, 0, 0]: duplicate of customer 1, interior depot, returns at 20 > H=15
    f.routes[1] = np.array([0, 1, 0, 0], np.int64)
    rep = mod.check_all(f, p)
    kinds = [v.kind for v in rep.violations]
    assert kinds == sorted(kinds, key=mod.ORDER.index) and set(kinds) == set(mod.ORDER)
    by = {v.kind: v for v in rep.violations}
    assert rep.first is by["capacity"] and by["capacity"].vehicle == 0
    assert by["capacity"].customers == (1, 2, 3, 4, 5)
    assert np.isclose(by["capacity"].magnitude, 16 / 100)
    assert by["coverage"].vehicle == -1 and by["coverage"].customers == (1,)
    assert by["time_window"].customers == (2,)  # A_2 = 20 > 15; inf arrival at 5 ignored
    assert by["availability"].vehicle == 1 and by["depot"].vehicle == 1
    assert by["connectivity"] == mod.Violation("connectivity", 0, (5,), 1.0)
    assert not rep.ok and np.isfinite(rep.total) and rep.total > 0


def test_each_check_isolated():
    p = make_problem([10, 10], 100, windows=[[0, 5], [0, 1e9]], shift_end=25)
    f = fleet([1, 2])
    assert mod.check_capacity(f, p) == []
    assert mod.check_coverage(f, p) == []
    tw = mod.check_time_windows(f, p)
    assert [(v.vehicle, v.customers) for v in tw] == [(0, (1,))]
    assert np.isclose(tw[0].magnitude, 5 / 25)  # A_1 = 10, l_1 = 5
    av = mod.check_availability(f, p)
    assert av[0].customers == (2,) and np.isclose(av[0].magnitude, 5 / 25)  # return 30 > 25
    assert mod.check_depot(f, p) == [] and mod.check_connectivity(f, p) == []
    assert mod.check_coverage(fleet([1], n=2), p)[0].customers == (2,)
    assert mod.check_all(fleet([1, 2]), make_problem([10, 10], 100)).ok


def test_magnitudes_are_dimensionless_and_additive():
    p = make_problem([60, 60], 100, rho_max=0.95)
    rep = mod.check_all(fleet([1, 2]), p)
    assert [v.kind for v in rep.violations] == ["capacity"]
    assert np.isclose(rep.total, (120 - 95) / 95)
