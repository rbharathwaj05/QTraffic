"""Phase 10: the reactive re-optimisation loop [SPEC 9.1-9.5 / 11 steps 31-42].

The loop lives beside `ReplanController` rather than inside it: the controller answers
"should we re-plan, how wide", this module answers "and then what happens" -- fallback,
scoped search, switching gate, deployment. `ReactiveLoop.on_event` is still the only way
the optimiser is reached in response to traffic [CLAUDE.md rule 7]; it calls
`ReplanController.decide` and does nothing when the answer is NONE.

RESPONSE BUDGET [SPEC 9.1 / v4 56-58]

    T_response = T_detection + T_optimization + T_deployment
    B_available = T_response_budget - T_detection - T_deployment

`B_available` is what the swarm is handed -- a wall-clock box, never an iteration count
[v4 57]. Every component is measured and logged per event. Per SPEC 15 this system does
not claim to "respond in real time": it targets T_response under the configured budget and
measures it per event, which is what `ResponseLog` is for.

DEGRADE IN QUALITY, NEVER IN AVAILABILITY [SPEC 9.5]

    "The optimizer's result replaces the fallback only when it arrives in budget."

Implemented literally: a fallback route is deployed FIRST, and the swarm result is swapped
in afterwards only if it finished inside `B_available` AND cleared the switching gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.checker import check_all
from backend.constraints.feasibility import Problem, insertion_scan
from backend.fleet import route_manager as rm
from backend.fleet.state import FleetState
from backend.optimization import ebqpso
from backend.optimization.fitness import Bounds, ProblemContext, TrafficState
from backend.road.path_index import PathIndex
from backend.traffic.controller import (
    Decision,
    ReplanController,
    Scope,
    affected_customers,
    affected_vehicles,
    degradation,
    remaining_cost,
)


class Fallback(IntEnum):
    """The fast-fallback ladder [doc2 31], picked by how much budget is already gone."""

    NONE = 0
    CACHED = 1  # 0-0.2 of the box: a cached alternative from a prior similar event
    GREEDY = 2  # 0.2-1.0: greedy nearest-feasible insertion, no swarm
    SWARM = 3  # the full scoped EB-QPSO result, if it lands in budget


# -- 10.4 switching cost [SPEC 9.3 point 4 / v4 20-21] ------------------------------------
@dataclass(frozen=True)
class SwitchAssessment:
    """Everything the gate looked at, kept for the log and for the tests."""

    f_current: float
    f_new: float
    s: float  # switching cost S
    f_switch: float  # F(R_new) + lambda_s S
    i_net: float  # F(R_current) - F_switch
    deploy: bool
    d_change: float
    n_changes: int
    d_backtrack: float


def neighbours(route: np.ndarray) -> dict[int, tuple[int, int]]:
    """customer -> (predecessor, successor) along one route, for N_changes [v4 21]."""
    r = np.asarray(route)
    return {int(r[k]): (int(r[k - 1]), int(r[k + 1])) for k in range(1, len(r) - 1)}


def switching_cost(
    current: dict[int, np.ndarray],
    new: dict[int, np.ndarray],
    traffic: TrafficState,
    cfg: QTrafficConfig,
    scale: float,
) -> tuple[float, float, int, float]:
    """S = lambda_1 D_change + lambda_2 N_changes + lambda_3 D_backtrack [v4 20].

        N_changes = sum_j 1[(a_j, pred_j, succ_j)_new != (a_j, pred_j, succ_j)_cur]   [v4 21]
        D_change  = |distance(new) - distance(current)|, normalised by `scale`
        D_backtrack = distance the new plan re-drives: legs whose reversal (j, i) the
                      current plan already contains, normalised by `scale`

    `scale` is the frozen per-scenario distance spread, so S is on the same footing as F.
    """
    d_cur = sum(_route_distance(r, traffic) for r in current.values())
    d_new = sum(_route_distance(r, traffic) for r in new.values())
    d_change = abs(d_new - d_cur) / (scale or 1.0)

    cur_pos, new_pos = {}, {}
    for v, r in current.items():
        cur_pos.update({c: (v, *pn) for c, pn in neighbours(r).items()})
    for v, r in new.items():
        new_pos.update({c: (v, *pn) for c, pn in neighbours(r).items()})
    n_changes = sum(1 for c, key in new_pos.items() if cur_pos.get(c) != key)

    cur_legs = {(int(i), int(j)) for r in current.values() for i, j in zip(r[:-1], r[1:])}
    back = 0.0
    for r in new.values():
        for i, j in zip(r[:-1], r[1:]):
            if (int(j), int(i)) in cur_legs:  # the new plan drives a leg back on itself
                back += float(traffic.distance[int(i), int(j)])
    d_backtrack = back / (scale or 1.0)

    s = cfg.lambda_1 * d_change + cfg.lambda_2 * n_changes + cfg.lambda_3 * d_backtrack
    return s, d_change, n_changes, d_backtrack


def _route_distance(route: np.ndarray, traffic: TrafficState) -> float:
    r = np.asarray(route)
    return float(traffic.distance[r[:-1], r[1:]].sum()) if len(r) > 1 else 0.0


def assess_switch(
    state: FleetState,
    vehicles: list[int],
    new_routes: dict[int, np.ndarray],
    traffic: TrafficState,
    bounds: Bounds,
    cfg: QTrafficConfig,
) -> SwitchAssessment:
    """Deploy iff I_net = F(R_current) - (F(R_new) + lambda_s S) > epsilon_switch [v4 20].

    Both F values are `remaining_cost` -- the same `fitness.combine` on the same frozen
    scales the optimiser uses -- so the comparison is like for like, and both are taken on
    the same traffic snapshot, so this measures the PLAN, not the world.
    """
    f_current = remaining_cost(state, vehicles, traffic, bounds, cfg)
    f_new = _cost_of(state, vehicles, new_routes, traffic, bounds, cfg)
    current_routes = {v: state.remaining(v) for v in vehicles}
    remaining_new = {v: _remaining_of(state, v, new_routes[v]) for v in new_routes}
    scale = (bounds.d[1] - bounds.d[0]) or 1.0
    s, d_change, n_changes, d_backtrack = switching_cost(
        current_routes, remaining_new, traffic, cfg, scale
    )
    f_switch = f_new + cfg.lambda_s * s
    i_net = f_current - f_switch
    return SwitchAssessment(
        f_current=f_current,
        f_new=f_new,
        s=s,
        f_switch=f_switch,
        i_net=i_net,
        deploy=bool(i_net > cfg.epsilon_switch),
        d_change=d_change,
        n_changes=n_changes,
        d_backtrack=d_backtrack,
    )


def _remaining_of(state: FleetState, v: int, route: np.ndarray) -> np.ndarray:
    """The untravelled tail of a CANDIDATE route, aligned with `state.remaining(v)` so the
    two are compared over the same span [SPEC 9.3 point 1]."""
    return rm.remaining_route(route, int(state.position[v]))


def _cost_of(
    state: FleetState,
    vehicles: list[int],
    routes: dict[int, np.ndarray],
    traffic: TrafficState,
    bounds: Bounds,
    cfg: QTrafficConfig,
) -> float:
    """F of a candidate plan, measured exactly like `remaining_cost` measures the incumbent
    -- by swapping the routes into a shallow copy of the state rather than by a second
    formula [F3 of the Phase 9 review: there is one definition of F]."""
    probe = FleetState(
        routes=[routes.get(v, state.routes[v]).copy() for v in range(state.n_vehicles)],
        assignment=state.assignment.copy(),
        position=state.position.copy(),
        last_reopt=state.last_reopt.copy(),
        served=set(state.served),
    )
    return remaining_cost(probe, vehicles, traffic, bounds, cfg)


# -- 10.2 the fast-fallback ladder [doc2 31] ----------------------------------------------
def greedy_insertion_route(
    state: FleetState, v: int, prob: Problem, cfg: QTrafficConfig
) -> np.ndarray:
    """Level 2: rebuild vehicle `v`'s remaining stops by cheapest feasible insertion.

    Reuses `feasibility.insertion_scan`, the same scan `repair.py`'s cheapest-insertion
    move uses -- no swarm, no second implementation of insertion. Stops are re-inserted
    one at a time into the cheapest slot that keeps the route time-feasible; the travelled
    prefix is never touched [SPEC 9.3 point 1].
    """
    pending = [int(c) for c in state.remaining_customers(v)]
    if not pending:
        return state.routes[v].copy()
    route = np.array([0, 0], dtype=np.int64)
    for cust in pending:
        late, cost, _ = insertion_scan(route, cust, prob, min(v, len(prob.t_start) - 1))
        rank = np.where(late == 0.0, cost, np.inf + 0 * cost + late + cost)
        slot = int(np.argmin(rank)) + 1
        route = np.insert(route, slot, cust)
    travelled = rm.travelled_route(state.routes[v], int(state.position[v]))
    return rm.splice(travelled, route[1:])


@dataclass
class RouteCache:
    """Level 1: alternatives that worked for a (vehicle, edge) pair in a PRIOR event.

    Deliberately tiny and in-process. A cached route is only offered back when the same
    vehicle meets the same edge again, which is the doc2 31 wording ("from a prior similar
    event"); anything cleverer is a similarity metric nobody has asked for yet.
    """

    routes: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def get(self, v: int, edges) -> np.ndarray | None:
        for e in edges:
            hit = self.routes.get((int(v), int(e)))
            if hit is not None:
                return hit.copy()
        return None

    def put(self, v: int, edges, route: np.ndarray) -> None:
        for e in edges:
            self.routes[(int(v), int(e))] = np.asarray(route).copy()


# -- 10.1 the response log [SPEC 9.1 / 15] -------------------------------------------------
@dataclass
class ResponseLog:
    """One row per event. SPEC 15: never claim real time -- measure it and print it."""

    t_sim: float
    delta: float
    decision: str
    scope: str
    affected: tuple[int, ...]
    t_detection: float = 0.0
    t_optimization: float = 0.0
    t_deployment: float = 0.0
    fallback: Fallback = Fallback.NONE
    deployed: Fallback = Fallback.NONE
    swarm_in_budget: bool = False
    i_net: float = 0.0
    switching_cost: float = 0.0
    n_changes: int = 0
    repair_distance: float = 0.0
    repair_iterations: float = 0.0
    feasible: bool = True  # deployed plan feasible AFTER this event
    feasible_before: bool = True  # and before it, so a pre-existing violation is visible
    b_available: float = 0.0

    @property
    def t_response(self) -> float:
        return self.t_detection + self.t_optimization + self.t_deployment

    def as_row(self) -> dict:
        return {
            "t_sim": round(self.t_sim, 1),
            "delta": round(self.delta, 4),
            "decision": self.decision,
            "scope": self.scope,
            "n_affected": len(self.affected),
            "T_detection": round(self.t_detection, 4),
            "T_optimization": round(self.t_optimization, 4),
            "T_deployment": round(self.t_deployment, 4),
            "T_response": round(self.t_response, 4),
            "B_available": round(self.b_available, 3),
            "fallback": self.fallback.name,
            "deployed": self.deployed.name,
            "in_budget": self.swarm_in_budget,
            "I_net": round(self.i_net, 4),
            "S": round(self.switching_cost, 4),
            "N_changes": self.n_changes,
            "RepairDistance": round(self.repair_distance, 4),
            "RepairIterations": round(self.repair_iterations, 2),
            "feasible": self.feasible,
            "was_feasible": self.feasible_before,
        }


# -- 10.5 the loop [SPEC 9.5 / 11 steps 31-42] ---------------------------------------------
@dataclass
class ReactiveLoop:
    """Traffic event -> decision -> fallback -> scoped search -> gate -> deployment."""

    cfg: QTrafficConfig
    state: FleetState
    prob: Problem
    ctx: ProblemContext
    index: PathIndex
    controller: ReplanController
    rng: np.random.Generator
    cache: RouteCache = field(default_factory=RouteCache)
    log: list[ResponseLog] = field(default_factory=list)

    def on_event(
        self,
        t_sim: float,
        edges,
        before: TrafficState,
        after: TrafficState,
    ) -> ResponseLog:
        """One pass of [SPEC 11 steps 31-42]. `before`/`after` are the traffic snapshots
        either side of the (already applied, already debounced) cost-matrix refresh.

        The caller owns debouncing and the incremental matrix refresh, because both belong
        to the simulator [SPEC 9.5 first steps]; by the time this is called the matrices
        are current and the edge batch is one batch.
        """
        edges = [int(e) for e in edges]
        # diagnostic, deliberately OUTSIDE every timer: a plan can arrive at this event
        # already violating something (a tight scenario the initial solve never fixed), and
        # the log must show that rather than blame it on the loop.
        was_feasible = check_all(self.state.to_fleet_route(), self.prob).ok

        # -- detection: path-index lookup + a Delta on cached matrices, no graph search ---
        t0 = time.perf_counter()
        A = sorted(affected_vehicles(self.state, self.index, edges))
        scope_v = A or list(self.state.vehicles())
        delta = degradation(self.state, scope_v, before, after, self.ctx.bounds, self.cfg)
        decision = self.controller.decide(t_sim, delta, A, self.state.n_vehicles, self.state)
        t_detection = time.perf_counter() - t0

        entry = ResponseLog(
            t_sim=t_sim,
            delta=delta,
            decision=decision.reason,
            scope=decision.scope.value,
            affected=decision.vehicles or tuple(A),
            t_detection=t_detection,
            feasible_before=was_feasible,
        )
        if decision.scope is Scope.NONE or not A:
            self.log.append(entry)
            return entry

        fleet_wide = decision.scope is Scope.FLEET
        budget = self.cfg.T_response_fleet if fleet_wide else self.cfg.T_response_local
        # [v4 57] the swarm gets the wall-clock remainder, not an iteration count
        entry.b_available = max(0.0, budget - t_detection - self.cfg.T_deployment_target)

        # -- availability first: a valid route is deployed before any search starts -------
        t1 = time.perf_counter()
        fallback_routes, level = self.fast_fallback(A, edges, entry.b_available, t_detection)
        if fallback_routes and not self._is_improvement(fallback_routes):
            fallback_routes, level = {}, Fallback.NONE  # never deploy a worse plan
        if fallback_routes:
            self._deploy(fallback_routes, t_sim)
        entry.fallback = level
        entry.deployed = level
        t_fallback = time.perf_counter() - t1

        # -- 10.3 scoped, warm-started EB-QPSO, time-boxed to what is left ----------------
        t2 = time.perf_counter()
        remaining_box = max(0.0, entry.b_available - (time.perf_counter() - t0))
        routes, result, _scope = ebqpso.reoptimize_local(
            self.state,
            self.prob,
            self.ctx,
            A,
            self.cfg,
            self.rng,
            time_budget_s=remaining_box,
            t_sim=t_sim,
        )
        entry.t_optimization = time.perf_counter() - t2
        spent = time.perf_counter() - t0 - t_detection
        entry.swarm_in_budget = bool(routes) and spent <= entry.b_available
        if result is not None:
            entry.repair_distance = result.diagnostics.get("repair_distance_mean", 0.0)
            entry.repair_iterations = result.diagnostics.get("repair_iterations_mean", 0.0)

        # -- 10.4 the gate: upgrade past the fallback only if it pays ---------------------
        t3 = time.perf_counter()
        if entry.swarm_in_budget:
            check = assess_switch(self.state, A, routes, after, self.ctx.bounds, self.cfg)
            entry.i_net, entry.switching_cost, entry.n_changes = (
                check.i_net,
                check.s,
                check.n_changes,
            )
            if check.deploy and self._is_improvement(routes):
                self._deploy(routes, t_sim)
                entry.deployed = Fallback.SWARM
                for v in A:
                    self.cache.put(v, edges, routes[v])
        # "keep current" when the gate says no: the FALLBACK stays deployed if one was.
        # Reverting it would trade a feasible, already-live plan for the pre-event plan the
        # event just invalidated -- availability is the one thing [SPEC 9.5] refuses to
        # degrade. The gate decides whether to UPGRADE beyond the fallback, nothing else.
        entry.t_deployment = time.perf_counter() - t3 + t_fallback

        if entry.deployed is not Fallback.NONE:
            self.controller.record(t_sim, self.state, A)  # cooldown starts on deployment
        entry.feasible = check_all(self.state.to_fleet_route(), self.prob).ok
        self.log.append(entry)
        return entry

    # -- 10.2 --------------------------------------------------------------------------
    def fast_fallback(
        self, A: list[int], edges, b_available: float, spent: float
    ) -> tuple[dict[int, np.ndarray], Fallback]:
        """Pick a ladder level by how much of the box is already gone [doc2 31].

        Level 1 (cheap, <= `fallback_l1_fraction` of the box spent): a cached alternative
        from a prior event on the same (vehicle, edge). Level 2: greedy cheapest-feasible
        insertion. Above `fallback_l2_fraction` there is no time even for that, so the
        current route stays -- which is itself a valid route, so availability holds.
        """
        used = spent / b_available if b_available > 0 else 1.0
        if used <= self.cfg.fallback_l1_fraction:
            cached = {v: r for v in A if (r := self.cache.get(v, edges)) is not None}
            if cached:
                return cached, Fallback.CACHED
        if used <= self.cfg.fallback_l2_fraction:
            return (
                {v: greedy_insertion_route(self.state, v, self.prob, self.cfg) for v in A},
                Fallback.GREEDY,
            )
        return {}, Fallback.NONE

    def _is_improvement(self, routes: dict[int, np.ndarray]) -> bool:
        """A fallback may only replace the current plan if it does not add violations.

        [SPEC 9.5] promises availability, not miracles: a greedy re-insertion that breaks a
        time window is worse than the route already on the truck, so it is dropped and the
        current plan stays (itself a valid, already-deployed route). One `check_all` per
        event, on the candidate fleet -- cheap next to the swarm it precedes.
        """
        probe = FleetState(
            routes=[
                routes.get(v, self.state.routes[v]).copy() for v in range(self.state.n_vehicles)
            ],
            assignment=self.state.assignment.copy(),
            position=self.state.position.copy(),
            last_reopt=self.state.last_reopt.copy(),
            served=set(self.state.served),
        )
        now = check_all(self.state.to_fleet_route(), self.prob).total
        cand = check_all(probe.to_fleet_route(), self.prob).total
        return cand <= now + 1e-12

    def _deploy(self, routes: dict[int, np.ndarray], t_sim: float) -> None:
        """Write new suffixes into the live plan [SPEC 11 step 41]. `apply_replan` splices
        onto the frozen prefix, so a deployment can never rewrite driven history."""
        for v, route in routes.items():
            tail = rm.remaining_route(route, int(self.state.position[v]))
            self.state.apply_replan(int(v), tail, t_sim)

    # -- reporting ---------------------------------------------------------------------
    def summary(self) -> dict:
        """`reroutes issued` counts events that deployed a swarm plan; `unnecessary`
        counts the ones where the swarm finished in budget and the gate still said no --
        i.e. reroutes the switching cost correctly prevented [SPEC 9.3 point 4]."""
        issued = sum(1 for e in self.log if e.deployed is Fallback.SWARM)
        unnecessary = sum(
            1 for e in self.log if e.swarm_in_budget and e.deployed is not Fallback.SWARM
        )
        return {
            "events": len(self.log),
            "reroutes_issued": issued,
            "unnecessary": unnecessary,
            "triggered": sum(1 for e in self.log if e.scope != Scope.NONE.value),
            "max_T_response": round(max((e.t_response for e in self.log), default=0.0), 4),
        }


__all__ = [
    "Decision",
    "Fallback",
    "ReactiveLoop",
    "ResponseLog",
    "RouteCache",
    "SwitchAssessment",
    "affected_customers",
    "assess_switch",
    "greedy_insertion_route",
    "switching_cost",
]
