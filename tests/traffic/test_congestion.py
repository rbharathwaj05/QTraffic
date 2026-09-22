from pathlib import Path

import numpy as np
"""Phase 0 stub test: only asserts the module imports. Flipped to real tests when its body lands."""

import pytest

from backend.config import QTrafficConfig
from backend.traffic import congestion as mod
from backend.traffic.events import EventKind, TrafficEvent, severity_to_factor

CAP = QTrafficConfig().rho_congestion_max


def event(edges, severity=0.5, kind=EventKind.CONGESTION, t_start=0.0, duration=600.0):
    return TrafficEvent(
        id=1,
        kind=kind,
        edge_ids=np.asarray(edges, dtype=np.int64),
        t_start=t_start,
        duration_s=duration,
        severity=severity,
    )


# -- core model [SPEC 7.2 / v4 3] --------------------------------------------------------
def test_speed_and_travel_time_follow_the_spec_formulas():
    """V = V_normal (1 - rho); T = D / V [SPEC 7.2]."""
    v_normal = np.array([10.0, 20.0])
    rho = np.array([0.0, 0.5])
    v = mod.speed(v_normal, rho, CAP)
    assert np.allclose(v, [10.0, 10.0])
    assert np.allclose(mod.travel_time(np.array([100.0, 100.0]), v), [10.0, 10.0])


def test_rho_is_capped_below_rho_max_so_speed_never_reaches_zero():
    cfg = QTrafficConfig()
    rho = mod.clip_rho(np.array([-1.0, 0.5, 0.95, 2.0]), cfg.rho_congestion_max)
    assert rho[0] == 0.0 and rho[1] == 0.5
    assert (rho < cfg.rho_congestion_max).all()  # half-open interval [SPEC 7.2]
    assert (mod.speed(np.ones(4), rho, cfg.rho_congestion_max) > 0).all()


def test_factor_and_rho_are_inverses():
    """f = 1/(1-rho) is what the cost matrix consumes; rho = 1 - 1/f comes back."""
    rho = np.array([0.0, 0.25, 0.7, 0.9])
    f = mod.rho_to_factor(rho, CAP)
    assert np.allclose(f, 1.0 / (1.0 - rho))
    assert np.allclose(mod.factor_to_rho(f), rho)
    assert mod.factor_to_rho(np.array([np.inf]))[0] == 1.0  # closure


def test_seventy_percent_congestion_triples_travel_time():
    assert mod.rho_to_factor(np.array([0.7]), CAP)[0] == pytest.approx(1 / 0.3)


# -- composition -------------------------------------------------------------------------
def test_compose_multiplies_events_on_the_same_edge_and_leaves_others_alone():
    f = mod.compose_factors(np.ones(5), [event([0], 0.5), event([0, 1], 0.5)], 5, CAP)
    assert f[0] == pytest.approx(4.0)  # two 2x events stack
    assert f[1] == pytest.approx(2.0)
    assert np.allclose(f[2:], 1.0)


def test_a_closure_dominates_everything_on_its_edge():
    f = mod.compose_factors(
        np.ones(3), [event([1], 0.3), event([1], 1.0, EventKind.CLOSURE)], 3, CAP
    )
    assert not np.isfinite(f[1]) and np.isfinite(f[0])


def test_compose_rejects_a_background_of_the_wrong_shape():
    with pytest.raises(ValueError):
        mod.compose_factors(np.ones(4), [], 5, CAP)


def test_congestion_level_buckets():
    lvl = mod.congestion_level(np.array([1.0, 1.3, 1.9, np.inf]))
    assert lvl.tolist() == [0, 1, 2, 3]


# -- background load (doc2 9, NOT core spec) ----------------------------------------------
def test_bpr_is_one_at_zero_volume_and_grows_with_load():
    assert mod.bpr_factor(np.array([0.0]), np.array([10.0]))[0] == 1.0
    grow = mod.bpr_factor(np.array([5.0, 10.0, 20.0]), np.full(3, 10.0))
    assert (np.diff(grow) > 0).all()
    assert grow[1] == pytest.approx(1.15)  # v/c = 1 -> 1 + a


def test_diurnal_volume_peaks_at_the_commuter_hours():
    base = np.array([100.0])
    at = lambda h: mod.diurnal_volume(h * 3600.0, base)[0]  # noqa: E731
    assert at(8) > at(12) > at(3)
    assert at(17) > at(21)
    assert at(8) == pytest.approx(at(8))  # deterministic


def test_fleet_congestion_is_off_unless_the_flag_is_set():
    """doc2 9 enhancement stays isolated behind `enable_fleet_congestion`."""
    base, cap, flow = np.full(3, 5.0), np.full(3, 10.0), np.full(3, 50.0)
    off = mod.background_factors(8 * 3600.0, base, cap, QTrafficConfig(), flow)
    no_flow = mod.background_factors(8 * 3600.0, base, cap, QTrafficConfig(), None)
    assert np.allclose(off, no_flow)  # flow ignored while the flag is off

    on = mod.background_factors(
        8 * 3600.0, base, cap, QTrafficConfig(enable_fleet_congestion=True), flow
    )
    assert (on > off).all()


def test_fleet_rho_is_flow_over_capacity_capped():
    rho = mod.fleet_rho(np.array([0.0, 5.0, 100.0]), np.full(3, 10.0), CAP)
    assert rho[0] == 0.0 and rho[1] == pytest.approx(0.5)
    assert rho[2] < CAP


# -- F1/F2 regression: the configured cap is the one actually enforced --------------------
def test_a_high_severity_event_cannot_push_rho_past_the_cap():
    """F2: `severity_to_factor` used to be the way round the ceiling [SPEC 7.2]."""
    f = mod.compose_factors(np.ones(2), [event([0], 0.99)], 2, CAP)
    assert mod.factor_to_rho(f)[0] < CAP  # strictly below: the interval is half-open
    assert f[0] == pytest.approx(mod.rho_to_factor(np.array([CAP]), CAP)[0])


def test_stacked_events_cannot_walk_past_the_cap_either():
    """Composition was the second route round the ceiling: three 0.8 events multiply to
    125x on their own, which alone would imply rho = 0.992."""
    f = mod.compose_factors(np.ones(2), [event([0], 0.8) for _ in range(3)], 2, CAP)
    assert mod.factor_to_rho(f)[0] < CAP


def test_a_closure_is_the_one_documented_exemption_from_the_cap():
    f = mod.compose_factors(np.ones(2), [event([0], 1.0, EventKind.CLOSURE)], 2, CAP)
    assert not np.isfinite(f[0])  # "no path", not a congestion level


def test_changing_rho_congestion_max_actually_changes_the_output():
    """THE test F1 was missing: a config field with no readers passes review trivially if
    every test only ever exercises its default value."""
    tight = QTrafficConfig(rho_congestion_max=0.80).rho_congestion_max
    assert tight != CAP
    ev = [event([0], 0.99)]

    loose_f = mod.compose_factors(np.ones(2), ev, 2, CAP)[0]
    tight_f = mod.compose_factors(np.ones(2), ev, 2, tight)[0]
    assert tight_f < loose_f
    assert mod.factor_to_rho(np.array([tight_f]))[0] < tight
    assert severity_to_factor(0.99, EventKind.CONGESTION, tight) < severity_to_factor(
        0.99, EventKind.CONGESTION, CAP
    )
    assert mod.clip_rho(np.array([0.9]), tight)[0] < tight
    assert mod.speed(np.ones(1), np.array([0.99]), tight)[0] == pytest.approx(1.0 - tight, abs=1e-9)


def test_no_hard_coded_cap_literal_survives_in_the_congestion_path():
    """Grep guard for F1: 0.95 may appear only in config.py's own default declaration.
    A literal anywhere else is a second, silent ceiling waiting to drift."""
    root = Path(__file__).resolve().parents[2]
    for rel in ("backend/traffic/congestion.py", "backend/traffic/events.py"):
        assert "0.95" not in (root / rel).read_text(), f"{rel} re-introduced a cap literal"
    assert "rho_congestion_max: float = 0.95" in (root / "backend/config.py").read_text()
