"""Phase 0 stub test: only asserts the module imports. Flipped to real tests when its body lands."""

import time

import numpy as np

from backend.config import QTrafficConfig
from backend.road import path_index as path_index_module
from backend.road.cost_matrix import (
    CostMatrix,
    TrafficVersion,
    inflate_all,
    update_factors_for_edges,
)
from backend.traffic import simulator as mod
from backend.traffic.events import EventKind, TrafficEvent
from tests.traffic.helpers import N_EDGES, cost_matrix, path_index


def event(id_, edges, severity=0.5, t_start=0.0, duration=100.0, kind=EventKind.CONGESTION):
    return TrafficEvent(id_, kind, np.asarray(edges, dtype=np.int64), t_start, duration, severity)


def simulator(events=(), cfg=None):
    idx, m = path_index(), cost_matrix()
    cfg = cfg or QTrafficConfig()
    sim = mod.TrafficSimulator(
        N_EDGES, np.zeros(N_EDGES), np.ones(N_EDGES), list(events), idx, m, cfg
    )
    return sim, idx, m


def test_events_activate_at_their_start_and_expire_at_their_end():
    sim, _, _ = simulator([event(1, [0], t_start=10.0, duration=20.0)])
    sim.step(5.0)
    assert sim.active == [] and sim.factor[0] == 1.0
    sim.step(10.0)  # t = 15, inside [10, 30)
    assert [e.id for e in sim.active] == [1] and sim.factor[0] == 2.0
    sim.step(20.0)  # t = 35, expired
    assert sim.active == [] and sim.factor[0] == 1.0


def test_step_updates_only_the_pairs_that_use_the_changed_edge():
    """[SPEC 10.2] scoped update through PathIndex.invalidate_edge -- no full rebuild."""
    sim, idx, m = simulator([event(1, [0], severity=0.5)])
    changed = sim.step(1.0)
    touched = {(int(i), int(j)) for i, j in changed}
    assert touched == idx.pairs_using([0]) == {(0, 1), (1, 0), (0, 3), (3, 0)}
    assert len(touched) < 4 * 4  # a strict subset of the matrix


def test_scoped_update_matches_a_full_recompute():
    """The cheap path must produce the same numbers as rebuilding everything."""
    sim, idx, m = simulator([event(1, [0, 3], severity=0.6)])
    sim.step(1.0)
    full = inflate_all(idx, m.edge_t0, sim.factor)
    assert np.allclose(m.factor, full)


def test_traffic_version_bumps_once_per_change_and_not_on_quiet_steps():
    sim, _, m = simulator([event(1, [0], t_start=5.0, duration=1e6)])
    v0 = m.version.value
    assert len(sim.step(1.0)) == 0 and m.version.value == v0  # nothing happened yet
    assert len(sim.step(10.0)) > 0 and m.version.value == v0 + 1
    assert len(sim.step(1.0)) == 0 and m.version.value == v0 + 1  # steady state


def test_inject_treats_a_past_start_time_as_now():
    sim, _, _ = simulator()
    sim.step(100.0)
    sim.inject(event(9, [1], t_start=0.0, duration=50.0))
    assert sim.pending[0].t_start == sim.t
    sim.step(1.0)
    assert [e.id for e in sim.active] == [9]


# -- 9.8 debounce [SPEC 9.5] --------------------------------------------------------------
def test_simultaneous_events_are_one_batch_not_n():
    """N simultaneous edge closures -> ONE controller pass."""
    cfg = QTrafficConfig(debounce_s=2.0)
    sim, _, _ = simulator(
        [event(i, [i], severity=0.5, t_start=0.5, duration=1e6) for i in range(3)], cfg
    )
    sim.step(1.0)  # all three activate inside the same step
    assert sim.take_batch() == []  # window still open: nothing handed over yet
    sim.step(2.0)  # window closed
    assert sim.take_batch() == [0, 1, 2]  # one batch covering every edge
    assert sim.take_batch() == []  # and it is drained


def test_events_arriving_inside_the_window_join_the_same_batch():
    cfg = QTrafficConfig(debounce_s=5.0)
    sim, _, _ = simulator([event(1, [0], t_start=0.5, duration=1e6)], cfg)
    sim.step(1.0)
    sim.inject(event(2, [3], t_start=1.0, duration=1e6))
    sim.step(1.0)  # t = 2, still inside the 5 s window
    assert not sim.batch_ready()
    sim.step(4.0)
    assert sim.take_batch() == [0, 3]


def test_background_drift_alone_does_not_open_a_debounce_window():
    """Only events batch; the diurnal curve moving is not something to re-plan on."""
    cfg = QTrafficConfig()
    idx, m = path_index(), cost_matrix()
    sim = mod.TrafficSimulator(
        N_EDGES, np.full(N_EDGES, 5.0), np.full(N_EDGES, 10.0), [], idx, m, cfg
    )
    sim.step(3600.0)
    assert sim.pending_batch == set() and sim.take_batch() == []


def test_snapshot_is_serialisable_and_reports_the_live_version():
    sim, _, m = simulator([event(1, [0], severity=1.0, kind=EventKind.CLOSURE, duration=1e6)])
    sim.step(1.0)
    snap = sim.snapshot()
    assert snap["active_event_ids"] == [1]
    assert snap["level"][0] == 3 and snap["t"] == 1.0  # blocked
    assert snap["traffic_version"] == m.version.value


# -- F9: the scoped update has a performance crossover, not a correctness one -------------
def dense_layer(n=14, n_edges=40, seed=0):
    """A path index where every pair uses a handful of shared edges, so background drift
    moves nearly every pair at once -- the regime that defeated the per-pair path."""
    rng = np.random.default_rng(seed)
    idx = path_index_module.PathIndex()
    idx.n = n
    for i in range(n):
        for j in range(n):
            e = np.empty(0, dtype=np.int64) if i == j else rng.choice(n_edges, 3, replace=False)
            idx.pair_to_edges[(i, j)] = np.asarray(e, dtype=np.int64)
            for k in np.asarray(e).ravel():
                idx.edge_to_pairs.setdefault(int(k), set()).add((i, j))
    dur = np.full((n, n), 100.0)
    np.fill_diagonal(dur, 0.0)
    m = CostMatrix(dur, dur.copy(), np.ones((n, n)), np.ones(n_edges), TrafficVersion("dense"))
    return idx, m, n_edges


def test_a_sparse_event_change_stays_on_the_per_pair_path():
    """One event edge touches a handful of pairs: far below the crossover fraction."""
    idx, m, n_edges = dense_layer()
    factor = np.ones(n_edges)
    factor[0] = 3.0
    changed = update_factors_for_edges(m, factor, idx, [0], max_fraction=0.5)
    assert 0 < len(changed) <= 0.5 * idx.n * idx.n


def test_a_drift_dominated_change_trips_the_full_recompute_and_agrees_with_it():
    """Both branches must produce identical numbers -- the fallback is a traversal change,
    not an arithmetic one."""
    idx, m, n_edges = dense_layer()
    drift = np.linspace(1.1, 2.0, n_edges)  # every edge moved, as the diurnal curve does

    scoped = CostMatrix(
        m.duration_s.copy(),
        m.distance_m.copy(),
        np.ones_like(m.factor),
        m.edge_t0.copy(),
        TrafficVersion("scoped"),
    )
    dense = CostMatrix(
        m.duration_s.copy(),
        m.distance_m.copy(),
        np.ones_like(m.factor),
        m.edge_t0.copy(),
        TrafficVersion("dense2"),
    )
    all_edges = list(range(n_edges))
    forced_scoped = update_factors_for_edges(scoped, drift, idx, all_edges, max_fraction=1.0)
    fell_back = update_factors_for_edges(dense, drift, idx, all_edges, max_fraction=0.5)

    assert len(fell_back) > 0.5 * idx.n * idx.n  # the crossover really did engage
    assert np.allclose(scoped.factor, dense.factor)  # identical arithmetic
    assert np.allclose(dense.factor, inflate_all(idx, dense.edge_t0, drift))
    assert len(forced_scoped) == len(fell_back)


def test_the_fallback_is_not_slower_than_the_per_pair_path_it_replaces():
    """Rough timing, not a tight perf assertion: it only has to show the crossover is
    pointing the right way in the dense regime."""
    idx, m, n_edges = dense_layer(n=26)
    drift = np.linspace(1.1, 2.0, n_edges)
    all_edges = list(range(n_edges))

    def run(max_fraction):
        mat = CostMatrix(
            m.duration_s.copy(),
            m.distance_m.copy(),
            np.ones_like(m.factor),
            m.edge_t0.copy(),
            TrafficVersion("t"),
        )
        t = time.perf_counter()
        update_factors_for_edges(mat, drift, idx, all_edges, max_fraction=max_fraction)
        return time.perf_counter() - t

    per_pair = min(run(1.0) for _ in range(3))  # crossover disabled
    fallback = min(run(0.5) for _ in range(3))  # crossover engaged
    print(
        f"\nF9 dense regime: per-pair {per_pair * 1e3:.2f} ms, " f"fallback {fallback * 1e3:.2f} ms"
    )
    assert fallback <= per_pair * 1.5


def test_a_drifting_simulator_step_still_produces_correct_factors():
    """End-to-end: the regime that used to walk the slow path now takes the fallback and
    still matches a full recompute."""
    idx, m, n_edges = dense_layer()
    cfg = QTrafficConfig()
    sim = mod.TrafficSimulator(
        n_edges, np.full(n_edges, 8.0), np.full(n_edges, 10.0), [], idx, m, cfg
    )
    sim.t = 6 * 3600.0
    sim.step(600.0)
    assert np.allclose(m.factor, inflate_all(idx, m.edge_t0, sim.factor))
