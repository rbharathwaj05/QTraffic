import pytest

from backend.simulation.clock import SPEEDS, SimClock


def test_speeds_multiply_real_time_into_sim_time():
    """[doc2 23] advance(real_dt) -> sim_dt = real_dt * speed."""
    for speed in SPEEDS:
        c = SimClock(speed)
        assert c.advance(2.0) == 2.0 * speed
        assert c.now() == 2.0 * speed


def test_speed_can_change_mid_run_without_rewinding_time():
    c = SimClock(1)
    c.advance(10.0)
    c.set_speed(60)
    c.advance(1.0)
    assert c.now() == 10.0 + 60.0  # past time keeps the rate it was accrued at


def test_only_documented_demo_speeds_are_accepted():
    with pytest.raises(ValueError):
        SimClock(7)
    with pytest.raises(ValueError):
        SimClock(1).set_speed(0)


def test_time_never_runs_backwards():
    c = SimClock(5, t0=100.0)
    assert c.now() == 100.0
    with pytest.raises(ValueError):
        c.advance(-1.0)
    with pytest.raises(ValueError):
        c.advance_sim(-1.0)


def test_advance_sim_is_the_headless_path():
    """Tests and replay jump in SIM seconds directly, unaffected by the multiplier."""
    c = SimClock(60)
    c.advance_sim(300.0)
    assert c.now() == 300.0
