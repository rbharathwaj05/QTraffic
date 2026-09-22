"""Phase 2: build_cost_matrices makes exactly one /table call and rejects nan;
effective_duration is a pure array op (zero OSRM calls); update_factors reproduces the
weighted-mean formula on the synthetic index and bumps TrafficVersion; npz round-trip."""

import time

import numpy as np
import pytest

from backend.road import cost_matrix as mod
from backend.road.path_index import PathIndex

from .fake_osrm import FakeOSRM
from .test_path_index import synthetic_index


def points(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(13.0 + a, 80.0 + b) for a, b in rng.uniform(0, 0.1, (n + 1, 2))]


def test_matrices_finite_plausible_10_customers():
    D, T = mod.build_cost_matrices(points(10), FakeOSRM())
    assert D.shape == T.shape == (11, 11)
    assert np.isfinite(D).all() and np.isfinite(T).all()
    off = ~np.eye(11, dtype=bool)
    assert (D[off] > 0).all() and (T[off] > 0).all()
    assert np.allclose(np.diag(D), 0) and np.allclose(np.diag(T), 0)


def test_unroutable_pair_raises():
    class Bad(FakeOSRM):
        def table(self, s, d=None):
            dur, dist = super().table(s, d)
            dur[0, 1] = np.nan
            return dur, dist

    with pytest.raises(ValueError, match="unroutable"):
        mod.build_cost_matrices(points(3), Bad())


@pytest.mark.parametrize("n", [50, 300])
def test_build_once_then_zero_osrm_calls(n):
    client = FakeOSRM()
    t = time.perf_counter()
    m = mod.build_cost_matrix(points(n), client)
    print(f"\nN={n}: build_cost_matrix {time.perf_counter() - t:.3f}s")
    assert client.table_calls == 1
    for _ in range(100):  # optimiser-facing lookup: pure array op
        c = mod.effective_duration(m)
    assert c.shape == (n + 1, n + 1) and client.table_calls == 1 and client.route_calls == 0


def test_update_factors_inflates_and_bumps_version():
    idx = synthetic_index()
    t0 = np.zeros(14)
    t0[10:14] = [10.0, 30.0, 20.0, 5.0]
    base = np.ones((3, 3)) * 100.0
    m = mod.CostMatrix(base, base.copy(), np.ones((3, 3)), edge_t0=t0)
    f = np.ones(14)
    assert mod.update_factors(m, f, idx).shape == (0, 2) and m.version.value == 0

    f[11] = 2.0  # edge 11 on paths (0,1) [10,11] and (1,2) [11,12]
    changed = mod.update_factors(m, f, idx)
    assert {tuple(c) for c in changed} == idx.invalidate_edge(11) == {(0, 1), (1, 2)}
    assert m.version.value == 1
    assert np.isclose(m.factor[0, 1], (10 + 60) / 40)  # [SPEC dynamic cost update]
    assert np.isclose(m.factor[1, 2], (60 + 20) / 50)
    assert np.isclose(m.factor[2, 0], 1.0) and np.isclose(m.factor[0, 2], 1.0)
    assert np.isclose(mod.inflate(0, 1, f, idx, t0), m.factor[0, 1])
    assert np.allclose(mod.effective_duration(m), base * m.factor)
    assert m.version.cache_key(0, 1) == ("default", 1, 0, 1)


def test_traffic_version_redis_backend():
    class R:
        def __init__(self):
            self.d = {}

        def get(self, k):
            return self.d.get(k)

        def incr(self, k):
            self.d[k] = self.d.get(k, 0) + 1
            return self.d[k]

    v = mod.TrafficVersion("s1", R())
    assert v.value == 0 and v.bump() == 1 and v.value == 1


def test_save_load_roundtrip(tmp_path):
    m = mod.build_cost_matrix(points(4), FakeOSRM(), edge_t0=np.arange(3.0))
    m.save(tmp_path / "matrices.npz")
    back = mod.CostMatrix.load(tmp_path / "matrices.npz")
    assert np.array_equal(back.duration_s, m.duration_s)
    assert np.array_equal(back.distance_m, m.distance_m)
    assert np.array_equal(back.edge_t0, m.edge_t0) and (back.factor == 1).all()


def test_inflate_all_empty_index():
    idx = PathIndex()
    idx.n = 2
    for i in range(2):
        for j in range(2):
            idx.pair_to_edges[(i, j)] = np.empty(0, dtype=np.int64)
    assert (mod.inflate_all(idx, np.ones(1), np.ones(1)) == 1).all()
