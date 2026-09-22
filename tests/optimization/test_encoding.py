"""Phase 4: spec worked example (y=0.62, 20 vehicles), one-hot shapes, decode covers
every customer once in z-order per vehicle, decode is pure + deterministic,
decode(encode(r)) == r, and a 50 ms budget for M=50 N=300 M_veh=50."""

import time

import numpy as np

from backend.optimization import encoding as mod


def test_spec_worked_example_y062_20_vehicles():
    # [SPEC v4 7] y_j = 0.62, M_veh = 20 -> spec a_j = 13 (one-indexed) = 12 here.
    a = mod.assign(np.array([[0.62]]), 20)
    assert a.dtype == np.int64 and a[0, 0] == 12 and a[0, 0] + 1 == 13
    assert mod.assign(np.array([[1.0, 0.0, 0.999]]), 20).tolist() == [[19, 0, 19]]


def test_binary_assignment_shapes():
    a = np.array([[0, 2, 2], [1, 1, 0]])
    b = mod.to_binary_assignment(a, 3)
    assert b.shape == (2, 3, 3) and (b.sum(axis=1) == 1).all()
    assert b[0].tolist() == [[1, 0, 0], [0, 0, 0], [0, 1, 1]]
    assert mod.to_binary_assignment(a[0], 3).shape == (3, 3)


def test_decode_ordering_and_coverage_random_swarm():
    rng = np.random.default_rng(0)
    M, N, V = 20, 40, 7
    X = mod.random_particle(N, rng, M)
    Y, Z = mod.split(X)
    fleets = mod.decode(X, V)
    assert len(fleets) == M
    for m, f in enumerate(fleets):
        assert len(f.routes) == V
        seen = np.concatenate([r[1:-1] for r in f.routes])
        assert sorted(seen.tolist()) == list(range(1, N + 1))  # every customer exactly once
        for v, r in enumerate(f.routes):
            assert r[0] == 0 and r[-1] == 0
            j = r[1:-1] - 1
            assert (f.assignment[j] == v).all()
            assert (np.diff(Z[m, j]) >= 0).all()  # pi_v = argsort z within vehicle


def test_decode_is_pure_and_deterministic():
    rng = np.random.default_rng(1)
    X = mod.random_particle(30, rng, 5)
    before = X.copy()
    f1, f2 = mod.decode(X, 4), mod.decode(X, 4)
    assert np.array_equal(X, before)
    assert all(np.array_equal(a, b) for r1, r2 in zip(f1, f2) for a, b in zip(r1.routes, r2.routes))


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(2)
    N, V = 25, 4
    ref = mod.decode(mod.random_particle(N, rng), V)[0]
    x = mod.encode(ref, N, V, rng)
    assert x.shape == (2 * N,) and (0 <= x).all() and (x <= 1).all()
    back = mod.decode(x, V)[0]
    assert all(np.array_equal(a, b) for a, b in zip(back.routes, ref.routes))
    assert mod.clip(np.array([-0.5, 2.0])).tolist() == [0.0, 1.0] and mod.dim(N) == 50


def test_decode_swarm_timing_m50_n300():
    rng = np.random.default_rng(3)
    X = mod.random_particle(300, rng, 50)
    mod.decode(X, 50)  # warm
    t = time.perf_counter()
    mod.decode(X, 50)
    dt = time.perf_counter() - t
    print(f"\ndecode M=50 N=300 M_veh=50: {dt * 1e3:.1f} ms")
    assert dt < 0.05
