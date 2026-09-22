"""The affected-subset subproblem [SPEC v4 50-55]: solve for A and C_A, not the fleet.

    R_A* = argmin_{R_A in H_A} F_A(R_A, t),   R_A = {R_v : v in A}

H_A restricts the search to the AFFECTED vehicles A and the AFFECTED (still unserved)
customers C_A. The point of v4 55 is the decoder: a local problem over four vehicles must
decode into four vehicles, never into the fleet's hundred --

    "This prevents a local rerouting problem involving four vehicles from accidentally
     assigning customers to all 100 vehicles."

`encoding.decode` already takes `n_vehicles`; this module is the remapping layer around
it. Inside the subproblem everything is LOCAL: vehicles 0..|A|-1, customers 1..|C_A| with
local index 0 as the depot. `LocalScope.to_global` translates a decoded local fleet back
to global vehicle ids and global matrix indices, and splices each result onto the frozen
travelled prefix [SPEC 9.3 point 1].

ponytail: the local matrix roots every vehicle at the DEPOT, while a real re-plan starts
from wherever the vehicle currently stands. Only the first leg's cost differs, and the
deployed route is re-evaluated and feasibility-checked on the true GLOBAL matrix before
the switching gate sees it -- so the approximation costs search quality, never
correctness. Upgrade path: one virtual origin node per affected vehicle in the local
matrix (|A| extra rows), which `Problem` would need to carry as a per-vehicle start index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.feasibility import Problem
from backend.fleet import route_manager as rm
from backend.fleet.state import FleetState
from backend.optimization.encoding import FleetRoute, clip, encode
from backend.optimization.fitness import Bounds, ProblemContext, TrafficState


@dataclass(frozen=True)
class LocalScope:
    """The A / C_A subproblem plus the maps back to global ids [v4 50-55]."""

    vehicles: tuple[int, ...]  # global vehicle ids; local index = position in this tuple
    customers: tuple[int, ...]  # global matrix indices; local index = position + 1
    ctx: ProblemContext  # local fitness context, n_vehicles = |A|
    prob: Problem  # local constraint problem
    current: FleetRoute  # the incumbent, in local indices

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    def to_global(self, local: FleetRoute, state: FleetState) -> dict[int, np.ndarray]:
        """Local decoded fleet -> {global vehicle id: full route}, travelled prefixes kept.

        Each local route [0, ...local customers..., 0] is translated back to global matrix
        indices and spliced onto that vehicle's travelled prefix [SPEC 9.3 point 1], so the
        vehicle keeps everything it has already driven and only the suffix is new. The
        local leading depot is dropped: the vehicle resumes from where it stands, which is
        the last stop of the prefix.
        """
        out: dict[int, np.ndarray] = {}
        for v_local, route in enumerate(local.routes):
            v = self.vehicles[v_local]
            stops = [self.customers[int(c) - 1] for c in route if int(c) != 0]
            travelled = rm.travelled_route(state.routes[v], int(state.position[v]))
            out[v] = rm.splice(travelled, np.array([*stops, 0], dtype=np.int64))
        return out


def build_scope(
    state: FleetState,
    prob: Problem,
    ctx: ProblemContext,
    affected: list[int] | tuple[int, ...],
    t_sim: float = 0.0,
    customers: set[int] | None = None,
) -> LocalScope:
    """Carve the A / C_A subproblem out of the live fleet state [v4 50-52].

    `customers` defaults to the union of the affected vehicles' REMAINING customers, i.e.
    C_A [v4 49] -- a served stop is never re-planned. Each vehicle's remaining capacity is
    charged for what it has already delivered, and `t_start` is the current sim time, so
    the local time windows mean exactly what they mean globally.
    """
    A = tuple(int(v) for v in affected)
    if customers is None:
        cust: set[int] = set()
        for v in A:
            cust |= {int(c) for c in state.remaining_customers(v)}
    else:
        cust = {int(c) for c in customers}
    C = tuple(sorted(cust))

    nodes = np.array([0, *C], dtype=np.int64)  # local 0 = depot, 1.. = C_A
    g2l = {int(g): i + 1 for i, g in enumerate(C)}

    local_prob = Problem(
        duration=prob.duration[np.ix_(nodes, nodes)].copy(),
        demand=prob.demand[nodes].copy(),
        service=prob.service[nodes].copy(),
        windows=prob.windows[nodes].copy(),
        # free capacity: what the truck can still take, after what it already dropped off
        capacity=np.array([prob.capacity[v] - _served_demand(state, prob, v) for v in A], float),
        shift_end=np.array([prob.shift_end[v] for v in A], float),
        t_start=np.full(len(A), float(t_sim)),  # every vehicle resumes now [doc2 23]
        rho_max=prob.rho_max,
    )
    local_traffic = TrafficState(
        duration=ctx.traffic.duration[np.ix_(nodes, nodes)].copy(),
        distance=ctx.traffic.distance[np.ix_(nodes, nodes)].copy(),
        congestion=ctx.traffic.congestion[np.ix_(nodes, nodes)].copy(),
    )
    current = _incumbent(state, A, g2l)
    local_ctx = ProblemContext(
        traffic=local_traffic,
        bounds=_scaled_bounds(ctx.bounds, len(C), ctx.n_customers),
        n_customers=len(C),
        n_vehicles=len(A),
        current=current,
    )
    return LocalScope(A, C, local_ctx, local_prob, current)


def _served_demand(state: FleetState, prob: Problem, v: int) -> float:
    """Demand vehicle `v` has already delivered: gone from the truck, and absent from the
    local problem, so it is charged against the local capacity instead."""
    driven = rm.travelled_route(state.routes[v], int(state.position[v]))
    return float(prob.demand[driven[driven != 0]].sum())


def _incumbent(state: FleetState, A: tuple[int, ...], g2l: dict[int, int]) -> FleetRoute:
    """The current plan restricted to A and C_A, in local indices -- both the warm-start
    seed [v4 53] and the R' reference the local F measures change against [v4 19]."""
    routes, assignment = [], {}
    for v_local, v in enumerate(A):
        stops = [g2l[int(c)] for c in state.remaining_customers(v) if int(c) in g2l]
        routes.append(np.array([0, *stops, 0], dtype=np.int64))
        for c in stops:
            assignment[c - 1] = v_local
    a = np.array([assignment.get(j, 0) for j in range(len(g2l))], dtype=np.int64)
    return FleetRoute(routes, a)


def _scaled_bounds(bounds: Bounds, n_local: int, n_global: int) -> Bounds:
    """The frozen per-scenario scales, pro-rated to the size of the subproblem [SPEC 7.2].

    Nothing is re-sampled: the bounds were computed once for the scenario and stay frozen.
    A subproblem holding a tenth of the customers simply carries a tenth of the scale, so a
    local F sits on the same footing as the fleet F it is compared against.
    """
    k = (n_local / n_global) if n_global else 1.0
    return Bounds(
        t=(bounds.t[0] * k, bounds.t[1] * k),
        d=(bounds.d[0] * k, bounds.d[1] * k),
        c=(bounds.c[0] * k, bounds.c[1] * k),
        r=bounds.r,
    )


def warm_start_particles(
    scope: LocalScope, cfg: QTrafficConfig, rng: np.random.Generator
) -> np.ndarray:
    """N_warm = warm_fraction * M perturbed copies of X_current [v4 53-54].

    The perturbation reuses the mechanism the system already has rather than inventing one
    (9.4 is explicit that warm start is the diversity-injection idea pointed the other
    way): per-block Gaussian jitter at `cfg.sigma_A` / `cfg.sigma_O`, the same scale elite
    breeding uses, so a warm particle stays in the incumbent's neighbourhood -- normally
    the same assignment with a couple of ranks swapped. Row 0 is the exact incumbent, so
    the swarm can never do worse than the plan it started from.

    The other N_diverse = diverse_fraction * M particles are deliberately NOT produced
    here: `QPSO.initialize` fills every row it is not handed with a uniform random
    particle, which is precisely the cold-start half [v4 54].
    """
    n_warm = int(round(cfg.warm_fraction * cfg.M))
    if n_warm <= 0 or scope.n_customers == 0:
        return np.empty((0, 2 * scope.n_customers))
    x = encode(scope.current, scope.n_customers, scope.n_vehicles, rng)
    seeds = np.repeat(np.atleast_2d(x), n_warm, axis=0)
    n = scope.n_customers
    noise = np.empty_like(seeds)
    noise[:, :n] = cfg.sigma_A * rng.standard_normal((n_warm, n))
    noise[:, n:] = cfg.sigma_O * rng.standard_normal((n_warm, n))
    seeds[1:] = clip(seeds[1:] + noise[1:])  # row 0 stays the exact incumbent
    return seeds
