"""The travelled/remaining split of a deployed route [SPEC 9.3 point 1].

    "Freeze what has been driven":  R_new = R_travelled (+) R_optimized

Every re-plan, every degradation Delta and every affected-customer set is computed on the
REMAINING (untravelled) suffix only. The travelled prefix is history: re-optimising it
would produce a plan that tells a vehicle to undo stops it already made.

Routes here are matrix-index arrays exactly as `encoding.FleetRoute` produces them:
`[0, j1, ..., jk, 0]`, depot 0 at both ends, customers 1..N. `position` is the index in
that array of the stop the vehicle is AT (0 = still at the depot, len(route) - 1 = back
at the depot, done).
"""

from __future__ import annotations

import numpy as np


def remaining_route(route: np.ndarray, position: int) -> np.ndarray:
    """R_remaining: the suffix from the current stop onwards, current stop included.

    The current stop is kept because it is the leg's origin -- `remaining_legs` needs a
    start node, and the travel time from where the vehicle stands is exactly the part of
    the plan that a traffic event can still make worse.

    Returns a COPY, not a slice view: `FleetState.remaining` hands this straight to
    callers, and a view would let an unrelated caller's in-place edit rewrite the live
    deployed route. Routes are O(N/M_veh) short, so the copy is free; the alias was not.
    """
    p = int(np.clip(position, 0, len(route) - 1))
    return np.array(route[p:], copy=True)


def travelled_route(route: np.ndarray, position: int) -> np.ndarray:
    """R_travelled: the frozen prefix, current stop included, so that
    `splice(travelled_route(r, p), optimized)` reconstructs a whole route.

    A COPY for the same reason as `remaining_route`: "frozen" must mean the caller cannot
    write through it into the live route either.
    """
    p = int(np.clip(position, 0, len(route) - 1))
    return np.array(route[: p + 1], copy=True)


def remaining_legs(route: np.ndarray, position: int) -> list[tuple[int, int]]:
    """Consecutive (i, j) OD pairs still to be driven -- the lookup keys into
    `PathIndex.pairs_using` for affected-vehicle identification [SPEC 10.3]."""
    r = remaining_route(route, position)
    return [(int(i), int(j)) for i, j in zip(r[:-1], r[1:])]


def remaining_customers(route: np.ndarray, position: int) -> np.ndarray:
    """Unserved customers on this route, as matrix indices, depot dropped [v4 49].

    The stop the vehicle is AT is already served (or being served), so it is excluded --
    C_v_remaining is what a re-plan is still free to move.
    """
    r = remaining_route(route, position)[1:]
    return r[r != 0]


def splice(travelled: np.ndarray, optimized: np.ndarray) -> np.ndarray:
    """R_new = R_travelled (+) R_optimized [SPEC 9.3 point 1].

    The junction stop is shared: `travelled` ends at the vehicle's current stop and
    `optimized` starts there, so the duplicate is dropped rather than visited twice.
    """
    travelled, optimized = np.asarray(travelled), np.asarray(optimized)
    if len(travelled) and len(optimized) and travelled[-1] == optimized[0]:
        optimized = optimized[1:]
    return np.concatenate([travelled, optimized]).astype(np.int64)


def route_cost_terms(
    route: np.ndarray, position: int, duration: np.ndarray, distance: np.ndarray, rho: np.ndarray
) -> tuple[float, float, float]:
    """(T, D, C) of the REMAINING portion of one route [SPEC 7.2 raw terms, 9.2 scope]:

        T = sum T_ij(t),  D = sum D_ij,  C = sum rho_ij(t) D_ij   over remaining legs

    Pure gathers into the Phase 2 matrices; no path is recomputed here, and the travelled
    prefix contributes nothing by construction.
    """
    r = remaining_route(route, position)
    if len(r) < 2:
        return 0.0, 0.0, 0.0
    i, j = r[:-1], r[1:]
    d = distance[i, j]
    return float(duration[i, j].sum()), float(d.sum()), float((rho[i, j] * d).sum())
