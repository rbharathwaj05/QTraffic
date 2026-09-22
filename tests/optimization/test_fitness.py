import time

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization import encoding as enc
from backend.optimization import fitness as mod
from backend.road.geometry import haversine_matrix


def traffic(n, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = 13.0 + rng.uniform(0, 0.1, n + 1), 80.0 + rng.uniform(0, 0.1, n + 1)
    D = haversine_matrix(lat, lon)
    rho = 1.0 + rng.uniform(0, 1, (n + 1, n + 1))
    np.fill_diagonal(rho, 1.0)
    return mod.TrafficState(D / 10.0 * rho, D, rho)


def brute_terms(fleet, tr):
    T = D = C = 0.0
    for r in fleet.routes:
        for i, j in zip(r[:-1], r[1:]):
            T += tr.duration[i, j]
            D += tr.distance[i, j]
            C += tr.congestion[i, j] * tr.distance[i, j]
    return T, D, C


def test_raw_terms_match_per_route_loop():
    tr, rng = traffic(30), np.random.default_rng(1)
    X = enc.random_particle(30, rng, 12)
    plans = enc.decode_dense(X, 5)
    T, D, C = mod.raw_terms(plans, tr)
    for m, f in enumerate(enc.decode(X, 5)):
        assert np.allclose((T[m], D[m], C[m]), brute_terms(f, tr))
    assert T.shape == (12,) and (T > 0).all()
    # list[FleetRoute] path gives the same numbers
    assert np.allclose(mod.raw_terms(mod.to_dense(enc.decode(X, 5)), tr)[0], T)


def test_route_change_v4_19_formula():
    # current: v0 = [1,2,3], v1 = [4,5]; new: v0 = [2,1,3], v1 = [5], v2 = [4]
    cur = enc.FleetRoute(
        [np.array([0, 1, 2, 3, 0]), np.array([0, 4, 5, 0]), np.array([0, 0])],
        np.array([0, 0, 0, 1, 1]),
    )
    new = enc.FleetRoute(
        [np.array([0, 2, 1, 3, 0]), np.array([0, 5, 0]), np.array([0, 4, 0])],
        np.array([0, 0, 0, 2, 1]),
    )
    R = mod.route_change(mod.to_dense([new]), mod.to_dense([cur]), 0.5, 0.5)
    # A: customer 4 moved -> 1/5. O: c1 |1-0|/3, c2 |0-1|/3, c3 0, c5 |0-1|/max(1,2,1)=1/2
    A_chg, O_chg = 1 / 5, (1 / 3 + 1 / 3 + 0 + 1 / 2) / 5
    assert np.isclose(R[0], 0.5 * A_chg + 0.5 * O_chg)
    assert np.isclose(mod.route_change(mod.to_dense([cur]), mod.to_dense([cur]), 0.5, 0.5)[0], 0)


def test_normalization_changes_ranking():
    """Spec-style worked example: raw sums rank by the largest-magnitude term (T);
    normalised terms let D and C matter."""
    cfg = QTrafficConfig()
    b = mod.Bounds(t=(2000.0, 3000.0), d=(100.0, 200.0), c=(0.0, 1.0))
    T = np.array([2500.0, 2450.0])
    D = np.array([150.0, 195.0])
    C = np.array([0.4, 0.95])
    R = np.zeros(2)
    raw = cfg.w_t * T + cfg.w_d * D + cfg.w_c * C + cfg.w_r * R
    F = mod.combine(T, D, C, R, b, cfg)
    assert raw[1] < raw[0] and F[0] < F[1]  # ranking flips under normalisation
    # penalty is additive with w_p, never part of the core F
    assert np.allclose(
        mod.combine(T, D, C, R, b, cfg, penalty=np.array([0, 2.0])), F + [0, cfg.w_p * 2]
    )


def test_bounds_pure_input_and_deterministic():
    tr, cfg = traffic(20), QTrafficConfig()
    rng = np.random.default_rng(2)
    X = enc.random_particle(20, rng, 8)
    plans = enc.decode_dense(X, 4)
    cur = enc.decode(enc.random_particle(20, rng), 4)[0]
    b1 = mod.compute_normalization_bounds(tr, 20, 4, np.random.default_rng(9))
    b2 = mod.Bounds(b1.t, b1.d, b1.c, b1.r)  # distinct object, same values
    assert b1 is not b2 and b1 == b2
    f1 = mod.evaluate(plans, tr, cur, b1, cfg)
    f2 = mod.evaluate(plans, tr, cur, b2, cfg)
    f3 = mod.evaluate(enc.decode(X, 4), tr, mod.to_dense([cur]), b1, cfg)
    assert np.array_equal(f1, f2) and np.array_equal(f1, f3) and f1.shape == (8,)
    assert np.array_equal(f1, mod.evaluate(plans, tr, cur, b1, cfg))  # regression
    # different bounds -> different F: bounds really are read from the argument
    b3 = mod.Bounds((0.0, 1.0), b1.d, b1.c)
    assert not np.array_equal(f1, mod.evaluate(plans, tr, cur, b3, cfg))
    assert np.all(mod.evaluate(plans, tr, None, b1, cfg) <= f1 + 1e-12)  # R' = 0 cold start


def test_evaluate_swarm_timing_m50_n300():
    tr, cfg = traffic(300), QTrafficConfig()
    rng = np.random.default_rng(3)
    plans = enc.decode_dense(enc.random_particle(300, rng, 50), 50)
    cur = enc.decode(enc.random_particle(300, rng), 50)[0]
    b = mod.compute_normalization_bounds(tr, 300, 50, rng)
    mod.evaluate(plans, tr, cur, b, cfg)
    t = time.perf_counter()
    F = mod.evaluate(plans, tr, cur, b, cfg)
    dt = time.perf_counter() - t
    print(f"\nevaluate M=50 N=300 M_veh=50: {dt * 1e3:.1f} ms")
    assert F.shape == (50,) and np.isfinite(F).all() and dt < 0.05
