"""Simulation clock [doc2 23]: the single source of SIM time.

Everything with a deadline in this system -- time windows, shift ends, the `T_cool`
cooldown, the event debounce window, `traffic_version` bumps -- is measured in SIM
seconds read from here. Wall-clock seconds are used for ONE thing only: the optimiser's
response budget (`T_response_local` / `T_response_fleet`). Because of that split, running
the demo at 60x changes how fast the world moves, never what the controller decides.
"""

from __future__ import annotations

SPEEDS = (1, 5, 10, 30, 60)  # supported demo multipliers [doc2 23]


class SimClock:
    """`advance(real_dt) -> sim_dt` at the current speed multiplier; `now()` is sim s."""

    def __init__(self, speed: int = 1, t0: float = 0.0) -> None:
        self.t = float(t0)
        self.speed = 1
        self.set_speed(speed)

    def set_speed(self, speed: int) -> None:
        """Change the multiplier mid-run. Only the documented demo speeds are allowed --
        an arbitrary float would make recorded runs unreplayable at a named speed."""
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}, got {speed}")
        self.speed = int(speed)

    def advance(self, real_dt: float) -> float:
        """Consume `real_dt` wall-clock seconds, advance sim time by `real_dt * speed`,
        and return the sim delta. Time never runs backwards."""
        if real_dt < 0:
            raise ValueError("real_dt must be >= 0")
        sim_dt = real_dt * self.speed
        self.t += sim_dt
        return sim_dt

    def advance_sim(self, sim_dt: float) -> float:
        """Jump forward by SIM seconds directly (headless tests, replay, scripted runs)."""
        if sim_dt < 0:
            raise ValueError("sim_dt must be >= 0")
        self.t += sim_dt
        return self.t

    def now(self) -> float:
        """Sim seconds since scenario start."""
        return self.t
"""clock: phase 0 placeholder. Contract defined in a later phase."""

# Phase 0-6: intentionally empty. Body + tests land with the phase that owns it.
