"""Single source of truth for every tunable parameter in the qtraffic spec.

Every field carries the spec's recommended default and a comment naming the spec
section it came from. Later phases read values from here; they never hard-code them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

_TOL = 1e-9  # float tolerance for the sum-to-one checks (0.4 + 0.2 + 0.3 + 0.1 != 1.0 exactly)


@dataclass
class QTrafficConfig:
    """All tunables as dataclass fields; `__post_init__` validates every instance, so an
    inconsistent config fails at construction rather than deep inside an optimiser run.
    Construct with keyword overrides, e.g. `QTrafficConfig(M=100, city_bbox=(...))`."""

    # --- swarm (spec: QPSO core) --------------------------------------------------
    M: int = 50  # swarm size
    T_iter_max: int = 200  # hard iteration budget per optimisation call
    T_iter_min: int = 20  # floor before stagnation / response-budget may stop early
    alpha_max: float = 1.0  # contraction-expansion coefficient at t=0
    alpha_min: float = 0.4  # contraction-expansion coefficient at t=T_iter
    alpha_update_every: int = 5  # re-project T every K iterations [SPEC 7.4 alpha schedule]
    two_opt_max_passes: int = 4  # bound on the gbest-route polish [SPEC 7.5]

    # --- elite breeding (spec: EB-QPSO) --------------------------------------------
    r_E: float = 0.15  # elite fraction, spec range [0.10, 0.20]           [v4 29]
    r_B: float = 0.08  # breeding fraction, spec range [0.05, 0.10]        [v4 32]
    # Breeding noise sigma [v4 31]. The spec gives the FORM of the term (sigma * eps,
    # eps ~ N(0, I)) but NO numeric default, so these are experimental knobs to be tuned
    # and ablated [v4 71]; 0.05 is a deliberately conservative starting point (a child
    # lands within ~0.1 of the parent blend in key space, i.e. usually the same permutation
    # with a few swapped ranks).
    sigma_A: float = 0.05  # breeding noise on the assignment block Y      [v4 31]
    sigma_O: float = 0.05  # breeding noise on the ordering block Z        [v4 31]
    # Block-aware alpha [v4 28]: the two key blocks may contract on separate schedules.
    # ON by default per v4; the Phase 13 ablation turns it off to separate its effect from
    # breeding itself. The spec gives no numbers for the per-block endpoints, so both
    # default to the shared alpha_max/alpha_min -- enabled-with-defaults is numerically
    # identical to disabled until these are retuned.
    block_alpha: bool = True  # per-block contraction-expansion schedules  [v4 28]
    alpha_max_order: float = 1.0  # alpha_O at t=0, Z-block                [v4 28]
    alpha_min_order: float = 0.4  # alpha_O at t=T_proj, Z-block           [v4 28]

    # --- objective weights (spec: fitness function), must sum to 1 -----------------
    w_t: float = 0.40  # total travel time
    w_d: float = 0.20  # total distance
    w_c: float = 0.30  # congestion-weighted distance C = sum rho_ij D_ij [v4 17]
    w_r: float = 0.10  # route-change penalty vs. incumbent plan

    # --- route-change split (spec: route-change penalty), must sum to 1 ------------
    eta_a: float = 0.5  # weight on assignment changes
    eta_o: float = 0.5  # weight on ordering changes

    # --- repair (spec: constraint repair) ------------------------------------------
    MAX_REPAIR_ITERATIONS: int = 8
    w_p: float = 0.5  # F_eval = F + w_p P, P = residual violation after repair [v4 22]

    # --- hysteresis (spec: re-plan controller) -------------------------------------
    theta_soft: float = 0.10  # relative cost delta that arms a re-plan
    theta_hard: float = 0.15  # relative cost delta that forces a re-plan
    theta_override: float = 0.30  # delta that bypasses cooldown entirely

    # --- cooldown (spec: re-plan controller) ---------------------------------------
    T_cool: float = 120.0  # seconds of SIM time between non-override re-plans [v4 47]
    persistence_cycles: int = 2  # consecutive cycles above theta_soft to fire [v4 46]
    epsilon_F: float = 1e-9  # floor in Delta = (F_after - F_before) / max(F_before, eps)
    debounce_s: float = 2.0  # SIM seconds: events inside this window = one controller pass
    # Performance crossover, not a correctness knob: above this share of the OD matrix the
    # per-pair scoped update is slower than one vectorised full recompute [SPEC 10.2].
    scoped_update_max_fraction: float = 0.5
    local_scope_fraction: float = 0.5  # affected/total above this -> FLEET not LOCAL

    # --- congestion (spec 7.2 / v4 3) ----------------------------------------------
    # NOTE: distinct from `rho_max` above, which is the vehicle LOAD ratio. This one caps
    # the road congestion level rho_ij(t) so V_ij = V_normal (1 - rho) never reaches 0.
    rho_congestion_max: float = 0.95  # rho_ij(t) in [0, 0.95) [SPEC 7.2]
    # Fleet-induced congestion is NOT in the core v3/v4 spec -- it is an enhancement from
    # the architecture doc (doc2 9). Off by default; the core degradation math never reads
    # it, so the ablation is a flag flip.
    enable_fleet_congestion: bool = False
    bpr_alpha: float = 0.15  # BPR a: t/t0 = 1 + a (q/c)^b (doc2 9, not core spec)
    bpr_beta: float = 4.0  # BPR b

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
    # T_response = T_detection + T_optimization + T_deployment [SPEC 9.1 / v4 56-58].
    # These are TARGETS to measure against per event, never a claim [SPEC 15].
    T_response_local: float = 10.0  # spec: < 5-10 s; upper bound used, tighten later
    T_response_fleet: float = 30.0  # spec: < 30 s
    T_detection_target: float = 1.0  # < 1 s: cost-matrix lookup, never a graph search
    T_deployment_target: float = 1.0  # < 1 s: writing the new plan to the fleet
    # Fast-fallback ladder (doc2 31): the share of B_available already spent when the
    # decision is taken picks the level. Level 1 cached route, 2 greedy insertion, 3 swarm.
    fallback_l1_fraction: float = 0.2  # <= this much budget used -> a cached route will do
    fallback_l2_fraction: float = 0.9  # <= this -> greedy insertion; above -> keep current

    # --- switching cost (spec 9.3 point 4 / v4 20-21) ------------------------------
    # S = lambda_1 D_change + lambda_2 N_changes + lambda_3 D_backtrack, and a new plan is
    # deployed iff I_net = F(R_current) - (F(R_new) + lambda_s S) > epsilon_switch.
    # The spec gives the FORM, not the numbers: these are tuned so the worked example of
    # 9.3 (a 30 min route replaced by a 29 min one) is correctly NOT deployed [v4 71].
    lambda_1: float = 0.10  # per unit of normalised distance changed
    lambda_2: float = 0.02  # per customer whose (vehicle, predecessor, successor) moved
    lambda_3: float = 0.10  # per unit of normalised backtracking distance
    lambda_s: float = 1.0  # overall weight of S inside F_switch
    epsilon_switch: float = 0.02  # net improvement a reroute must beat to be worth it

    # --- capacity (spec: feasibility) ----------------------------------------------
    rho_max: float = 0.95  # max vehicle load ratio

    # --- road graph (spec: road model, data acquisition) ---------------------------
    city_query: str = "Chennai, India"  # osmnx place query
    # (west, south, east, north); when set, overrides city_query. Small bboxes for tests.
    city_bbox: tuple[float, float, float, float] | None = None
    city_cache_dir: Path = Path("data/city")  # GraphML cache location
    depot_latlon: tuple[float, float] | None = None  # None -> graph centroid

    # --- scenario generation (spec v2 doc 6: customer/vehicle placement) ------------
    shift_end_s: float = 28800.0  # H_v: 8 h shift, sim seconds from scenario start
    vehicle_capacity: int = 50  # Q_v, demand units
    demand_range: tuple[int, int] = (1, 10)  # inclusive uniform per customer
    service_time_range: tuple[int, int] = (120, 600)  # s, inclusive uniform
    tw_width_range: tuple[int, int] = (1800, 7200)  # s, uniform width of a tight window
    tw_anytime_fraction: float = 0.20  # share of customers with window [0, shift_end_s)

    # --- offline fallback (no OSRM): haversine distance / this speed -> durations -----
    fallback_speed_mps: float = 8.33  # 30 km/h; only used when no OSRM matrices exist

    # --- infra --------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    osrm_url: str = "http://localhost:5000"
    seed: int | None = None

    # Closed [lo, hi] spec ranges checked by validate(); repr=False keeps it out of print().
    _ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "r_E": (0.10, 0.20),
            "r_B": (0.05, 0.10),
            "r_inject": (0.20, 0.30),
        },
        repr=False,
    )

    def __post_init__(self) -> None:
        self.validate()  # dataclass hook: runs right after field assignment

    def validate(self) -> None:
        """Raise ValueError on any inconsistent parameter group."""
        # 1. weight groups that must partition 1.0
        groups = {
            "objective weights w_t+w_d+w_c+w_r": self.w_t + self.w_d + self.w_c + self.w_r,
            "route-change split eta_a+eta_o": self.eta_a + self.eta_o,
            "warm start warm_fraction+diverse_fraction": self.warm_fraction + self.diverse_fraction,
        }
        for name, total in groups.items():
            if not math.isclose(total, 1.0, abs_tol=_TOL):
                raise ValueError(f"{name} must sum to 1, got {total}")
        # 2. scalar fields pinned to a spec interval
        for name, (lo, hi) in self._ranges.items():
            v = getattr(self, name)
            # r_B == 0 is the documented off-switch for the Phase 13 breeding ablation
            # ("EB-QPSO with breeding disabled must equal plain QPSO"); any other value
            # must sit inside the spec range [v4 32].
            if name == "r_B" and v == 0.0:
                continue
            if not lo <= v <= hi:
                raise ValueError(f"{name}={v} outside spec range [{lo}, {hi}]")
        # 3. ordering / sign constraints between related fields
        if not 0 < self.alpha_min <= self.alpha_max:
            raise ValueError("require 0 < alpha_min <= alpha_max")
        if not 0 < self.alpha_min_order <= self.alpha_max_order:
            raise ValueError("require 0 < alpha_min_order <= alpha_max_order")
        if self.sigma_A < 0 or self.sigma_O < 0:
            raise ValueError("require sigma_A >= 0 and sigma_O >= 0")
        if not self.theta_soft <= self.theta_hard <= self.theta_override:
            raise ValueError("require theta_soft <= theta_hard <= theta_override")
        if not 0 < self.rho_max <= 1:
            raise ValueError("require 0 < rho_max <= 1")
        if not 0 < self.rho_congestion_max < 1:
            raise ValueError("require 0 < rho_congestion_max < 1")
        if self.T_cool < 0 or self.debounce_s < 0 or self.persistence_cycles < 1:
            raise ValueError("require T_cool >= 0, debounce_s >= 0, persistence_cycles >= 1")
        if not 0 <= self.scoped_update_max_fraction <= 1:
            raise ValueError("require 0 <= scoped_update_max_fraction <= 1")
        if not 0 <= self.tw_anytime_fraction <= 1:
            raise ValueError("require 0 <= tw_anytime_fraction <= 1")
        if self.tw_width_range[1] > self.shift_end_s or self.tw_width_range[0] <= 0:
            raise ValueError("require 0 < tw_width_range <= shift_end_s")
        if self.M <= 0 or self.T_iter_min <= 0 or self.T_iter_min > self.T_iter_max:
            raise ValueError("require M > 0 and 0 < T_iter_min <= T_iter_max")
        if self.alpha_update_every <= 0 or self.two_opt_max_passes < 0:
            raise ValueError("require alpha_update_every > 0 and two_opt_max_passes >= 0")
