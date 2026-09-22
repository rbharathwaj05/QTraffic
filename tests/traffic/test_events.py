import numpy as np
import pytest

from backend.traffic import events as mod
from backend.traffic.events import EventKind, TrafficEvent


def test_severity_maps_to_the_spec_travel_time_factor():
    """f = 1/(1 - severity) [SPEC 7.2]: severity is the event's rho."""
    assert mod.severity_to_factor(0.0, EventKind.CONGESTION) == 1.0
    assert mod.severity_to_factor(0.7, EventKind.CONGESTION) == pytest.approx(1 / 0.3)
    assert mod.severity_to_factor(0.5, EventKind.ACCIDENT) == pytest.approx(2.0)


def test_closure_and_full_severity_are_infinite():
    assert mod.severity_to_factor(0.2, EventKind.CLOSURE) == float("inf")
    assert mod.severity_to_factor(1.0, EventKind.CONGESTION) == float("inf")


def test_severity_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        mod.severity_to_factor(1.5, EventKind.CONGESTION)


def test_active_window_is_half_open():
    e = TrafficEvent(
        1, EventKind.CONGESTION, np.array([0]), t_start=10.0, duration_s=5.0, severity=0.5
    )
    assert e.t_end == 15.0
    assert not e.active_at(9.9) and e.active_at(10.0) and e.active_at(14.9)
    assert not e.active_at(15.0)


def test_generate_events_is_seeded_and_within_bounds():
    a = mod.generate_events(100, 3600.0, 10.0, np.random.default_rng(3))
    b = mod.generate_events(100, 3600.0, 10.0, np.random.default_rng(3))
    assert [e.id for e in a] == [e.id for e in b]
    assert [e.t_start for e in a] == [e.t_start for e in b]  # reproducible [CLAUDE.md]
    for e in a:
        assert 0.0 <= e.t_start <= 3600.0
        assert 0.0 <= e.severity <= 1.0
        assert 1 <= len(e.edge_ids) <= 3
        assert (e.edge_ids < 100).all() and (e.edge_ids >= 0).all()


def test_generate_events_scales_with_the_rate():
    rng = np.random.default_rng(0)
    few = mod.generate_events(100, 3600.0, 1.0, rng)
    many = mod.generate_events(100, 3600.0, 100.0, rng)
    assert len(many) > len(few)
    assert mod.generate_events(100, 3600.0, 0.0, rng) == []


def test_events_round_trip_through_json(tmp_path):
    src = mod.generate_events(50, 7200.0, 20.0, np.random.default_rng(1))
    p = tmp_path / "events.json"
    mod.save_events(src, str(p))
    back = mod.load_events(str(p))
    assert len(back) == len(src)
    for a, b in zip(src, back):
        assert a.id == b.id and a.kind == b.kind and a.severity == pytest.approx(b.severity)
        assert np.array_equal(a.edge_ids, b.edge_ids)
        assert a.t_start == pytest.approx(b.t_start)
