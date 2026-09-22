"""Single source of truth for every tunable parameter in the qtraffic spec.

Every field carries the spec's recommended default and a comment naming the spec
section it came from. Later phases read values from here; they never hard-code them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_TOL = 1e-9


@dataclass
class QTrafficConfig:
    # --- swarm (spec: QPSO core) --------------------------------------------------
    M: int = 50  # swarm size
    T_iter_max: int = 200  # hard iteration budget per optimisation call
    T_iter_min: int = 20  # floor before stagnation / response-budget may stop early
    alpha_max: float = 1.0  # contraction-expansion coefficient at t=0
    alpha_min: float = 0.4  # contraction-expansion coefficient at t=T_iter

    # --- elite breeding (spec: EB-QPSO) --------------------------------------------
    r_E: float = 0.15  # elite fraction, spec range [0.10, 0.20]
    r_B: float = 0.08  # breeding fraction, spec range [0.05, 0.10]

    # --- objective weights (spec: fitness function), must sum to 1 -----------------
    w_t: float = 0.40  # total travel time
    w_d: float = 0.20  # total distance
    w_c: float = 0.30  # constraint-violation penalty
    w_r: float = 0.10  # route-change penalty vs. incumbent plan

    # --- route-change split (spec: route-change penalty), must sum to 1 ------------
    eta_a: float = 0.5  # weight on assignment changes
    eta_o: float = 0.5  # weight on ordering changes

    # --- repair (spec: constraint repair) ------------------------------------------
    MAX_REPAIR_ITERATIONS: int = 8
    w_p: float = 0.5  # penalty weight applied to residual violations after repair

    # --- hysteresis (spec: re-plan controller) -------------------------------------
    theta_soft: float = 0.10  # relative cost delta that arms a re-plan
    theta_hard: float = 0.15  # relative cost delta that forces a re-plan
    theta_override: float = 0.30  # delta that bypasses cooldown entirely

    # --- cooldown (spec: re-plan controller) ---------------------------------------
    T_cool: float = 120.0  # seconds of sim time between non-override re-plans

    # --- warm start (spec: warm start), must sum to 1 ------------------------------
    warm_fraction: float = 0.20  # particles seeded from incumbent solution
    diverse_fraction: float = 0.80  # particles seeded randomly

    # --- diversity injection (spec: diversity maintenance) -------------------------
    delta_div: float = 0.05  # swarm-diversity threshold; spec gives no default, placeholder
    r_inject: float = 0.25  # fraction re-initialised on injection, spec range [0.20, 0.30]

    # --- stagnation (spec: stagnation detection) -----------------------------------
    epsilon_stag: float = 1e-4  # min relative gbest improvement to count as progress
    patience: int = 15  # window p: iterations without progress before stagnation

    # --- response budget, wall-clock seconds (spec: real-time requirements) --------
    T_response_local: float = 10.0  # spec: < 5-10 s; upper bound used, tighten later
    T_response_fleet: float = 30.0  # spec: < 30 s

    # --- capacity (spec: feasibility) ----------------------------------------------
    rho_max: float = 0.95  # max vehicle load ratio

    # --- infra --------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    osrm_url: str = "http://localhost:5000"
    seed: int | None = None

    _ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "r_E": (0.10, 0.20),
            "r_B": (0.05, 0.10),
            "r_inject": (0.20, 0.30),
        },
        repr=False,
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on any inconsistent parameter group."""
        groups = {
            "objective weights w_t+w_d+w_c+w_r": self.w_t + self.w_d + self.w_c + self.w_r,
            "route-change split eta_a+eta_o": self.eta_a + self.eta_o,
            "warm start warm_fraction+diverse_fraction": self.warm_fraction + self.diverse_fraction,
        }
        for name, total in groups.items():
            if not math.isclose(total, 1.0, abs_tol=_TOL):
                raise ValueError(f"{name} must sum to 1, got {total}")
        for name, (lo, hi) in self._ranges.items():
            v = getattr(self, name)
            if not lo <= v <= hi:
                raise ValueError(f"{name}={v} outside spec range [{lo}, {hi}]")
        if not 0 < self.alpha_min <= self.alpha_max:
            raise ValueError("require 0 < alpha_min <= alpha_max")
        if not self.theta_soft <= self.theta_hard <= self.theta_override:
            raise ValueError("require theta_soft <= theta_hard <= theta_override")
        if not 0 < self.rho_max <= 1:
            raise ValueError("require 0 < rho_max <= 1")
        if self.M <= 0 or self.T_iter_min <= 0 or self.T_iter_min > self.T_iter_max:
            raise ValueError("require M > 0 and 0 < T_iter_min <= T_iter_max")
