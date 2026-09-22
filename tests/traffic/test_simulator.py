import numpy as np

from backend.config import QTrafficConfig
from backend.road.cost_matrix import inflate_all
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
