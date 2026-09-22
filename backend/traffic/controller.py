"""Re-plan controller: decides if/when/how wide to re-optimise after traffic changes.

THE ONLY PATH to the optimiser in response to an event [CLAUDE.md rule 7]. Nothing else
calls the solver because of traffic; everything goes through `ReplanController.decide`.

Degradation [SPEC 9.2 single-vehicle / v4 44 fleet form -- the fleet form is canonical
here because Phase 9.7 needs affected-SET granularity]:

    F_A_before = F(R_A_remaining, t-)        cost of the remaining plan, OLD matrices
    F_A_after  = F(R_A_remaining, t+)        SAME plan, NEW matrices
    Delta_A    = (F_A_after - F_A_before) / max(F_A_before, epsilon_F)

Same route both sides: this measures what the WORLD did, not what a re-plan could win.
It is computed on the remaining (untravelled) suffix only [SPEC 9.3 point 1].

Hysteresis ladder [SPEC 9.2 / v4 45-47], thresholds straight from config:

    Delta <= theta_soft      (0.10)  -> NONE: noise, disarm
    theta_soft < D <= hard   (0.15)  -> arm; fire only if the NEXT cycle is also above
                                        theta_soft (persistence, v4 46)
    theta_hard < D <= over   (0.30)  -> fire now IF the cooldown has elapsed
    Delta > theta_override   (0.30)  -> severe: fire now, cooldown overridden

Cooldown is per vehicle and in SIM seconds [v4 47]: a vehicle may be re-planned only if
`t_sim - last_reopt[v] >= T_cool`, unless the override fires.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from backend.config import QTrafficConfig
from backend.fleet import route_manager as rm
from backend.fleet.state import FleetState
from backend.optimization.fitness import Bounds, TrafficState
from backend.road.path_index import PathIndex


class Scope(str, Enum):
    NONE = "none"
    LOCAL = "local"  # only affected vehicles, budget T_response_local
    FLEET = "fleet"  # whole fleet, budget T_response_fleet


@dataclass(frozen=True)
class Decision:
    scope: Scope
    delta: float  # relative cost increase that triggered it
    reason: str
    vehicles: tuple[int, ...] = ()  # the vehicles this decision re-plans
    override: bool = False  # cooldown was bypassed [v4 45 severe branch]


# -- 9.4 degradation --------------------------------------------------------------------
def remaining_cost(
    state: FleetState,
    vehicles: Iterable[int],
    traffic: TrafficState,
    bounds: Bounds,
    cfg: QTrafficConfig,
) -> float:
    """F over the REMAINING routes of `vehicles` under one traffic snapshot [v4 44].

    Same weights and same frozen per-scenario scales as the optimiser, but the RATIO form
    of the objective [CLAUDE.md rule 5: F = w_t T/T_ref + w_d D/D_ref + w_c C/C_ref], not
    `fitness.combine`'s min-max form. The reason is specific to this function: Delta is a
    RATIO, and the -lo offset in (x - lo)/(hi - lo) does not cancel in a ratio the way it
    cancels in a difference. Worse, a single vehicle's remaining leg costs sit far below a
    fleet-scale `lo`, so the offset can drive F_before negative and make Delta meaningless
    exactly where Phase 9.7 needs it (per-vehicle). Scale without offset keeps Delta
    scale-invariant and comparable between one vehicle and the whole fleet.

    R' = 0 on both sides: the plan is compared with itself, only the world moved.
    """
    T = D = C = 0.0
    for v in vehicles:
        t, d, c = rm.route_cost_terms(
            state.routes[int(v)],
            int(state.position[int(v)]),
            traffic.duration,
            traffic.distance,
            traffic.congestion,
        )
        T, D, C = T + t, D + d, C + c
    return (
        cfg.w_t * T / _scale(bounds.t)
        + cfg.w_d * D / _scale(bounds.d)
        + cfg.w_c * C / _scale(bounds.c)
    )


def _scale(lo_hi: tuple[float, float]) -> float:
    """X_ref: the frozen per-scenario spread, never recomputed mid-run [SPEC 7.2]."""
    lo, hi = lo_hi
    return (hi - lo) or 1.0


def degradation(
    state: FleetState,
    vehicles: Iterable[int],
    before: TrafficState,
    after: TrafficState,
    bounds: Bounds,
    cfg: QTrafficConfig,
) -> float:
    """Delta_A = (F_A_after - F_A_before) / max(F_A_before, epsilon_F) [SPEC 9.2, v4 44].

    `epsilon_F` guards the division when the remaining plan is (nearly) free -- an empty
    remaining route has F_before = 0 and no meaningful relative degradation.
    """
    vehicles = list(vehicles)
    f_before = remaining_cost(state, vehicles, before, bounds, cfg)
    f_after = remaining_cost(state, vehicles, after, bounds, cfg)
    return (f_after - f_before) / max(abs(f_before), cfg.epsilon_F)


# -- 9.7 affected vehicles / customers [SPEC 10.3, v4 48-49] -----------------------------
def affected_vehicles(state: FleetState, index: PathIndex, edge_ids: Iterable[int]) -> set[int]:
    """A(E) = {v : some REMAINING leg (i, j) of R_v has a cached path through some e in E}.

    Pure `PathIndex` lookup [SPEC 10.3]. Geographic proximity is the wrong test and the
    spec says why: it "over-counts" vehicles that are near the incident but routed around
    it and "under-counts" vehicles that are far away but whose path runs through it.
    `test_controller.py` constructs exactly that pair of cases.

    Batching several edges is the union over them [v4 48], which is what
    `pairs_using` already computes in one pass.
    """
    pairs = index.pairs_using(int(e) for e in edge_ids)
    if not pairs:
        return set()
    return {v for v in state.vehicles() if any(leg in pairs for leg in state.remaining_legs(v))}


def affected_customers(state: FleetState, vehicles: Iterable[int]) -> set[int]:
    """C_A = union of C_v_remaining over the affected vehicles [v4 49]: matrix indices of
    customers not yet served, so a re-plan never moves a delivery that already happened."""
    out: set[int] = set()
    for v in vehicles:
        out |= {int(c) for c in state.remaining_customers(int(v))}
    return out


# -- 9.5 / 9.6 the gate ------------------------------------------------------------------
@dataclass
class ReplanController:
    """Hysteresis + persistence + per-vehicle cooldown in front of the optimiser."""

    cfg: QTrafficConfig
    armed_cycles: int = 0  # consecutive cycles above theta_soft [v4 46]
    last_replan_t: float = -np.inf  # fleet-wide, for reporting; the gate is per vehicle
    history: list[float] = field(default_factory=list)  # deltas seen, for diagnostics

    # -- metric ---------------------------------------------------------------------
    def cost_delta(self, cost_now: float, cost_planned: float) -> float:
        """delta = (cost_now - cost_planned) / max(|cost_planned|, epsilon_F).

        Phase 0 wrote a bare `/ cost_planned`; that divides by zero for a finished plan,
        so the same `epsilon_F` floor as `degradation` is applied -- one definition of the
        relative-increase metric, not two.
        """
        return (cost_now - cost_planned) / max(abs(cost_planned), self.cfg.epsilon_F)

    def cooldown_ok(self, state: FleetState, vehicles: Iterable[int], t_sim: float) -> bool:
        """True when EVERY vehicle in the set has served its `T_cool` [v4 47]. Sim time."""
        return all(t_sim - float(state.last_reopt[int(v)]) >= self.cfg.T_cool for v in vehicles)

    def decide(
        self,
        t_sim: float,
        delta: float,
        affected: Iterable[int],
        total_vehicles: int,
        state: FleetState | None = None,
    ) -> Decision:
        """Walk the ladder [SPEC 9.2 / v4 45-47] and return what to do.

        Phase 0's signature took `affected_vehicles: int`; it is now the affected SET,
        because the cooldown is per vehicle [v4 47] and the caller needs to know which
        vehicles a LOCAL re-plan covers. `state` may be None only when there is nothing to
        gate (no affected vehicles).

        Scope: LOCAL while the affected share is at or below `cfg.local_scope_fraction`,
        FLEET above it -- and always FLEET for the severe override, which is a fleet-wide
        disruption by definition.
        """
        c = self.cfg
        affected = [int(v) for v in affected]
        self.history.append(float(delta))

        if delta <= c.theta_soft:
            self.armed_cycles = 0  # noise: disarm [v4 45]
            return Decision(Scope.NONE, delta, "below theta_soft (noise)")

        scope = self._scope(len(affected), total_vehicles)

        if delta > c.theta_override:  # severe: cooldown does not apply [v4 45]
            self.armed_cycles = 0
            return self._fire(t_sim, delta, affected, Scope.FLEET, "severe override", True)

        if delta > c.theta_hard:  # hard trigger, cooldown applies [v4 45]
            if state is not None and not self.cooldown_ok(state, affected, t_sim):
                return Decision(Scope.NONE, delta, "hard trigger held by cooldown", tuple(affected))
            self.armed_cycles = 0
            return self._fire(t_sim, delta, affected, scope, "above theta_hard", False)

        # theta_soft < delta <= theta_hard: persistence [v4 46]
        self.armed_cycles += 1
        if self.armed_cycles < c.persistence_cycles:
            return Decision(
                Scope.NONE,
                delta,
                f"armed {self.armed_cycles}/{c.persistence_cycles} cycles above theta_soft",
                tuple(affected),
            )
        if state is not None and not self.cooldown_ok(state, affected, t_sim):
            return Decision(Scope.NONE, delta, "persistent but held by cooldown", tuple(affected))
        self.armed_cycles = 0
        return self._fire(t_sim, delta, affected, scope, "persistent above theta_soft", False)

    def _scope(self, n_affected: int, total: int) -> Scope:
        share = n_affected / total if total else 0.0
        return Scope.LOCAL if share <= self.cfg.local_scope_fraction else Scope.FLEET

    def _fire(self, t_sim, delta, affected, scope, reason, override) -> Decision:
        self.last_replan_t = float(t_sim)
        return Decision(scope, delta, reason, tuple(affected), override)

    def record(self, t_sim: float, state: FleetState | None = None, vehicles=()) -> None:
        """Mark a re-plan as EXECUTED at `t_sim`: resets the persistence counter and starts
        the cooldown clock of every vehicle that was re-planned [v4 47]."""
        self.last_replan_t = float(t_sim)
        self.armed_cycles = 0
        if state is not None:
            state.record_replan(vehicles, t_sim)
