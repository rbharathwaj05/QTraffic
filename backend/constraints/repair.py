"""Bounded minimal-perturbation repair cascade [SPEC 8.2-8.4, v4 23].

=====================================================================================
NON-WRITE-BACK RULE [SPEC 8.3 / v4 10] -- THE most important invariant in qtraffic.

`repair()` takes a DECODED `FleetRoute` and returns a repaired `FleetRoute` plus
diagnostics. It never receives, returns, or touches the raw key vector X, pbest,
gbest or mbest; there is no such parameter (`test_repair.py` asserts this on the
signature). The optimiser loop keeps X untouched and only uses the repaired route to
compute F_eval. Do not add an X / particle / swarm argument to anything in this file.
=====================================================================================

Cascade [SPEC 8.4], implemented literally in `repair`:

    repair_attempts = 0
    LOOP (bounded by MAX_REPAIR_ITERATIONS + 1 checks -- no unbounded `while`):
        check constraints in fixed order (checker.ORDER)
        all satisfied -> accept, exit
        repair_attempts += 1
        repair_attempts > MAX_REPAIR_ITERATIONS -> accept as-is with residual P,
                                                    capped_out = True, exit
        apply the minimal-perturbation move for the FIRST violation; log distance

Move table [SPEC 8.2]  (repaired = argmin_feasible distance(r, decoded), insertion cost
breaks ties only):
    overloaded vehicle : eject the customer whose removal restores feasibility with the
                         smallest sequence change (fewest shifted stops); tie-break by
                         cheapest insertion elsewhere; reinsert at the closest feasible slot
    missing customer   : insert at the feasible slot closest to its decoded (vehicle, pos)
    duplicate customer : drop the occurrence deviating further from its decoded position
    time-window        : feasible in-route relocation with the smallest sequence change;
                         if none exists, eject the first late stop and reinsert elsewhere
    availability       : eject the last stop (shifts nobody) and reinsert elsewhere
    depot              : re-wrap the route as [0, stops, 0]
    connectivity       : the cost matrix IS the shortest path, so at sequence level the
                         spec's "substitute shortest path" is the identity; a non-finite
                         leg can only be avoided by moving its far-end stop
                         (# ponytail: relocation instead of a graph re-route; add a
                         PathIndex-level detour if closures ever produce inf legs)

Distance [v4 23]: d_R = eta_a d_A + eta_o d_O,
    d_A = N_assignment_changes / N,  d_O = sum_j |pos_j_decoded - pos_j_repaired| / N.

Ablation strategies (Phase 13) share the interface via `strategy`:
    "minimal"         spec behaviour above
    "cheapest"        eject the largest violator, reinsert at the cheapest insertion cost
    "smallest_demand" eject the smallest-demand stop on the violating route, reinsert cheapest
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.checker import Violation, ViolationReport, check_all
from backend.constraints.feasibility import Problem, insertion_scan, lateness
from backend.optimization.encoding import FleetRoute

STRATEGIES = ("minimal", "cheapest", "smallest_demand")


@dataclass
class RepairResult:
    """Repaired route + the Phase 13 diagnostics [SPEC 8.4 RepairDistance / RepairIterations /
    RepairFailureRate = mean(capped_out)]."""

    fleet: FleetRoute
    report: ViolationReport  # residual violations after the cascade
    distance: float  # d_R(decoded, fleet) [v4 23]
    iterations: int  # moves applied
    capped_out: bool  # True -> evaluate as-is with F + w_p * residual
    moves: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)  # d_R after each move

    @property
    def residual(self) -> float:
        """P for F_eval = F + w_p P [v4 22]; 0 when feasible."""
        return self.report.total


# -- positions / distance ------------------------------------------------------------
def positions(fleet: FleetRoute, n_customers: int) -> tuple[np.ndarray, np.ndarray]:
    """(veh, pos) per matrix index 1..N (index 0 unused): vehicle and stop index of each
    customer, first occurrence wins, -1 / 0 when absent. Loops over vehicles only."""
    segs = [r[1:-1] for r in fleet.routes]  # customer stops only, per vehicle
    flat = np.concatenate(segs)
    length = np.array([len(s) for s in segs])
    vv = np.repeat(np.arange(len(segs)), length)  # vehicle id of each flat stop
    pp = np.arange(len(flat)) - np.repeat(np.cumsum(length) - length, length)  # index in route
    veh = np.full(n_customers + 1, -1, np.int64)  # -1 = customer absent (coverage gap)
    pos = np.zeros(n_customers + 1, np.int64)
    # NumPy scatter with duplicate indices keeps the LAST write; reversing the arrays
    # makes the first occurrence in route order the one that survives.
    veh[flat[::-1]] = vv[::-1]  # reversed scatter: the FIRST occurrence wins
    pos[flat[::-1]] = pp[::-1]
    return veh, pos


def route_distance(decoded: FleetRoute, repaired: FleetRoute, eta_a: float, eta_o: float) -> float:
    """d_R = eta_a d_A + eta_o d_O [v4 23]; absent customers count as assignment changes."""
    n = len(decoded.assignment)
    v0, p0 = positions(decoded, n)
    v1, p1 = positions(repaired, n)
    d_a = np.count_nonzero(v0[1:] != v1[1:]) / n  # [1:] skips the unused depot slot
    d_o = np.abs(p0[1:] - p1[1:]).sum() / n
    return float(eta_a * d_a + eta_o * d_o)


# -- primitives (pure: always return a new FleetRoute) --------------------------------
def _copy(f: FleetRoute) -> FleetRoute:
    return FleetRoute([r.copy() for r in f.routes], f.assignment.copy())


def _remove(f: FleetRoute, v: int, idx: int) -> FleetRoute:
    """Drop route[v][idx] (idx into the full [0, ..., 0] array)."""
    g = _copy(f)
    g.routes[v] = np.delete(g.routes[v], idx)
    return g


def _insert(f: FleetRoute, v: int, slot: int, cust: int) -> FleetRoute:
    """Insert `cust` before route[v][slot]; keeps `assignment` in sync."""
    g = _copy(f)
    g.routes[v] = np.insert(g.routes[v], slot, cust)
    g.assignment[cust - 1] = v
    return g


def _best_slot(
    f: FleetRoute,
    cust: int,
    prob: Problem,
    dec_veh: int,
    dec_pos: int,
    cfg: QTrafficConfig,
    strategy: str,
) -> tuple[int, int]:
    """(vehicle, slot) for `cust` [SPEC 8.2]. "minimal": feasible slot minimising the
    distance delta eta_a [v != dec_veh] + eta_o (|pos - dec_pos| + #shifted), tie insertion
    cost; other strategies: cheapest feasible insertion. If no slot is feasible anywhere,
    the least-infeasible one (capacity ok > least lateness > cheapest) so coverage always
    holds and the residual is measured honestly."""
    # ponytail: one jitted scan per vehicle (~8 ms/particle at S4); fold into a single
    # CSR kernel over all vehicles if repair shows up in the Phase 13 profile
    scans = [insertion_scan(r, cust, prob, v) for v, r in enumerate(f.routes)]
    # Flatten every (vehicle, slot) candidate into parallel arrays so one lexsort picks
    # the winner across the whole fleet.
    n_slot = np.array([len(r) - 1 for r in f.routes])  # route of L stops has L-1 slots
    late = np.concatenate([s[0] for s in scans])
    cost = np.concatenate([s[1] for s in scans])
    cap_ok = np.repeat([s[2] for s in scans], n_slot)  # per-vehicle scalar -> per slot
    veh = np.repeat(np.arange(len(f.routes)), n_slot)
    new_pos = np.arange(len(late)) - np.repeat(np.cumsum(n_slot) - n_slot, n_slot)  # slot p -> p-1
    if strategy == "minimal":
        # Stops after the insertion point all shift by one: count them as perturbation.
        shifted = np.repeat(n_slot - 1, n_slot) - new_pos
        dist = cfg.eta_a * (veh != dec_veh) + cfg.eta_o * (np.abs(new_pos - dec_pos) + shifted)
    else:
        dist = cost
    # rank: infeasibility first (feasible < capacity-ok-but-late < over capacity)
    rank = np.where(cap_ok & (late == 0.0), 0.0, np.where(cap_ok, 1.0 + late, np.inf))
    # lexsort: last key is primary -> sort by rank, then dist, then cost; take the best.
    k = np.lexsort((cost, dist, rank))[0]
    return int(veh[k]), int(new_pos[k]) + 1  # back to slot index p


def _eject_choice(seg: np.ndarray, score: np.ndarray, prob: Problem, strategy: str) -> int:
    """Index into `seg` to eject for the ablation strategies: "cheapest" -> largest
    `score` (violation contribution), "smallest_demand" -> smallest demand."""
    if strategy == "smallest_demand":
        return int(np.argmin(prob.demand[seg]))
    return int(np.argmax(score))


# -- moves ------------------------------------------------------------------------------
def _move_capacity(
    f: FleetRoute, vio: Violation, prob: Problem, cfg: QTrafficConfig, strategy: str, dec
) -> tuple[FleetRoute, str]:
    v = vio.vehicle
    seg = f.routes[v][1:-1]
    excess = prob.demand[seg].sum() - prob.rho_max * prob.capacity[v]
    if strategy == "minimal":
        # Candidates whose single removal fixes the overload; pick the last one in route
        # order (shifts the fewest downstream stops). Fallback: largest demand.
        restores = np.flatnonzero(prob.demand[seg] >= excess)
        # smallest sequence change = fewest stops shifted = latest position; the
        # insertion-cost tie-break lives in _best_slot
        idx = int(restores[-1]) if restores.size else int(np.argmax(prob.demand[seg]))
    else:
        idx = _eject_choice(seg, prob.demand[seg], prob, strategy)
    return _relocate(f, v, idx + 1, prob, cfg, strategy, dec, "capacity")  # +1: skip depot


def _move_time_window(
    f: FleetRoute, vio: Violation, prob: Problem, cfg: QTrafficConfig, strategy: str, dec
) -> tuple[FleetRoute, str]:
    v = vio.vehicle
    r = f.routes[v]
    late = lateness(r, prob.duration, prob.service, prob.windows, prob.t_start[v])[1:-1]
    if strategy == "minimal":
        # feasible in-route relocation with the smallest sequence change [SPEC 8.2]
        best, best_key = None, None
        pos_r = np.zeros(prob.n_customers + 1, np.int64)  # current position of each stop
        pos_r[r[1:-1]] = np.arange(len(r) - 2)
        # Try moving each stop i to every other slot of the same route (O(L^2) scans,
        # each jitted); keep the feasible candidate with the smallest total shift.
        for i in range(1, len(r) - 1):
            rest = np.delete(r, i)
            l2, cost, _ = insertion_scan(rest, int(r[i]), prob, v)
            for p in np.flatnonzero(l2 == 0.0):  # only zero-lateness slots qualify
                cand = np.insert(rest, p + 1, r[i])
                d = int(np.abs(pos_r[cand[1:-1]] - np.arange(len(r) - 2)).sum())  # d_O * N
                key = (d, cost[p])  # tuple compare: distance first, cost breaks ties
                if best_key is None or key < best_key:
                    best, best_key = cand, key
        if best is not None:
            g = _copy(f)
            g.routes[v] = best
            return g, f"time_window: reorder v{v} d_O={best_key[0]}"
        idx = int(np.flatnonzero(late)[0])  # no in-route fix exists: eject first late stop
    else:
        idx = _eject_choice(r[1:-1], late, prob, strategy)
    return _relocate(f, v, idx + 1, prob, cfg, strategy, dec, "time_window")


def _move_availability(
    f: FleetRoute, vio: Violation, prob: Problem, cfg: QTrafficConfig, strategy: str, dec
) -> tuple[FleetRoute, str]:
    v = vio.vehicle
    seg = f.routes[v][1:-1]
    # Removing the last stop shortens the route without shifting anyone else.
    idx = len(seg) - 1 if strategy != "smallest_demand" else int(np.argmin(prob.demand[seg]))
    return _relocate(f, v, idx + 1, prob, cfg, strategy, dec, "availability")


def _move_coverage(
    f: FleetRoute, vio: Violation, prob: Problem, cfg: QTrafficConfig, strategy: str, dec
) -> tuple[FleetRoute, str]:
    dec_veh, dec_pos = dec
    veh, _ = positions(f, prob.n_customers)
    c = vio.customers[0]
    if veh[c] == -1:  # missing -> closest feasible slot to the decoded position
        v, p = _best_slot(f, c, prob, dec_veh[c], dec_pos[c], cfg, strategy)
        return _insert(f, v, p, c), f"coverage: insert {c} -> v{v}@{p - 1}"
    # duplicate -> drop the occurrence deviating further from decoded (veh, pos)
    occ = [(v, i) for v, r in enumerate(f.routes) for i in np.flatnonzero(r[1:-1] == c)]
    dev = [cfg.eta_a * (v != dec_veh[c]) + cfg.eta_o * abs(i - dec_pos[c]) for v, i in occ]
    v, i = occ[int(np.argmax(dev))]  # the occurrence farthest from where decode put it
    g = _remove(f, v, i + 1)
    g.assignment[c - 1] = positions(g, prob.n_customers)[0][c]  # re-sync to surviving copy
    return g, f"coverage: drop duplicate {c} from v{v}@{i}"


def _move_depot(f: FleetRoute, vio: Violation, prob, cfg, strategy, dec) -> tuple[FleetRoute, str]:
    g = _copy(f)
    r = g.routes[vio.vehicle]
    # Strip every depot occurrence, then wrap once: fixes missing ends and interior 0s.
    g.routes[vio.vehicle] = np.concatenate(([0], r[r != 0], [0])).astype(np.int64)
    return g, f"depot: rewrap v{vio.vehicle}"


def _move_connectivity(
    f: FleetRoute, vio: Violation, prob: Problem, cfg: QTrafficConfig, strategy: str, dec
) -> tuple[FleetRoute, str]:
    v, r = vio.vehicle, f.routes[vio.vehicle]
    c = vio.customers[0]
    idx = int(np.flatnonzero(r == c)[0]) if c != 0 else len(r) - 2  # depot far end -> last stop
    return _relocate(f, v, idx, prob, cfg, strategy, dec, "connectivity")


def _relocate(
    f: FleetRoute, v: int, idx: int, prob: Problem, cfg: QTrafficConfig, strategy: str, dec, kind
) -> tuple[FleetRoute, str]:
    """Eject route[v][idx] and reinsert it at `_best_slot` [SPEC 8.2 eject / reinsert]."""
    c = int(f.routes[v][idx])
    g = _remove(f, v, idx)
    dec_veh, dec_pos = dec
    nv, p = _best_slot(g, c, prob, dec_veh[c], dec_pos[c], cfg, strategy)
    return _insert(g, nv, p, c), f"{kind}: eject {c} from v{v}@{idx - 1} -> v{nv}@{p - 1}"


MOVES = {
    "capacity": _move_capacity,
    "coverage": _move_coverage,
    "time_window": _move_time_window,
    "availability": _move_availability,
    "depot": _move_depot,
    "connectivity": _move_connectivity,
}


# -- cascade ------------------------------------------------------------------------
def repair(
    decoded: FleetRoute, prob: Problem, cfg: QTrafficConfig, strategy: str = "minimal"
) -> RepairResult:
    """Bounded cascade [SPEC 8.4]: at most `cfg.MAX_REPAIR_ITERATIONS` moves, then accept
    as-is with `capped_out=True` and residual P for F + w_p P. Input is never mutated.
    NO X / particle argument exists here by design [SPEC 8.3] -- see module docstring."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    dec = positions(decoded, prob.n_customers)  # reference (veh, pos) for "minimal" moves
    fleet, moves, dists = _copy(decoded), [], []  # work on a copy: input stays untouched
    attempts, d = 0, 0.0
    for _ in range(cfg.MAX_REPAIR_ITERATIONS + 1):  # bounded: never `while not feasible`
        report = check_all(fleet, prob)
        if report.ok:  # feasible -> done, residual P = 0
            return RepairResult(fleet, report, d, attempts, False, moves, dists)
        attempts += 1
        if attempts > cfg.MAX_REPAIR_ITERATIONS:  # budget spent: hand back with residual P
            moves.append(f"capped out: residual P={report.total:.4g}")
            return RepairResult(fleet, report, d, attempts - 1, True, moves, dists)
        # Dispatch on the FIRST violation kind only; the next check_all re-evaluates all.
        fleet, msg = MOVES[report.first.kind](fleet, report.first, prob, cfg, strategy, dec)
        moves.append(msg)
        d = route_distance(decoded, fleet, cfg.eta_a, cfg.eta_o)  # perturbation so far
        dists.append(d)
    raise AssertionError("unreachable: the loop always returns")  # pragma: no cover
