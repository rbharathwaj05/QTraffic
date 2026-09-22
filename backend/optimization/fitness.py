"""Canonical four-term objective F, minimised by every optimiser [SPEC 7.2, v4 17].

    F = w_t T' + w_d D' + w_c C' + w_r R'      with X' = (X - X_min) / (X_max - X_min)
    F_eval = F + w_p P                          (P: residual violation from repair, v4 22)

Raw terms are pure array gathers into the Phase 2 matrices (no path is ever recomputed
here) and are vectorised across the WHOLE swarm: one fancy-index per matrix.
Runtime / wall-clock is NEVER part of F; it is logged on a separate axis.

`evaluate` is the single public entry point; no optimiser module defines its own F.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization.encoding import FleetRoute, decode_dense, random_particle
from backend.road.cost_matrix import CostMatrix, effective_duration

Dense = tuple[np.ndarray, np.ndarray, np.ndarray]  # (assignment, order, ptr) of decode_dense


@dataclass(frozen=True)
class TrafficState:
    """Matrices at one traffic_version: T_path(t), D_path, rho(t) (path-level congestion
    factor). Index 0 = depot, 1..N = customers."""

    duration: np.ndarray  # (N+1, N+1) effective travel time, s
    distance: np.ndarray  # (N+1, N+1) m
    congestion: np.ndarray  # (N+1, N+1) rho_ij(t), 1.0 = free flow

    @classmethod
    def from_cost_matrix(cls, m: CostMatrix) -> TrafficState:
        return cls(effective_duration(m), m.distance_m, m.factor)


@dataclass(frozen=True)
class Bounds:
    """Per-scenario normalisation bounds, computed ONCE by `compute_normalization_bounds`
    and frozen [SPEC 7.2]. Never recompute inside an optimisation loop."""

    t: tuple[float, float]
    d: tuple[float, float]
    c: tuple[float, float]
    r: tuple[float, float] = (0.0, 1.0)  # R' in [0, 1] by construction [v4 19]


@dataclass
class ProblemContext:
    """Everything an optimiser needs to call `evaluate` for one scenario."""

    traffic: TrafficState
    bounds: Bounds
    n_customers: int
    n_vehicles: int
    current: FleetRoute | None = None  # deployed plan for R' (None -> R' = 0)


# -- route set -> flat visit sequences ------------------------------------------------
def to_dense(fleets: list[FleetRoute]) -> Dense:
    """Inverse of `encoding.decode` on route lists (used when repair hands back
    FleetRoutes). Loops over (particle, vehicle) only."""
    M, V = len(fleets), len(fleets[0].routes)
    N = fleets[0].assignment.shape[0]
    a = np.stack([f.assignment for f in fleets])
    order = np.empty((M, N), dtype=np.int64)
    ptr = np.zeros((M, V + 1), dtype=np.int64)
    for m, f in enumerate(fleets):
        segs = [r[1:-1] - 1 for r in f.routes]  # strip depot ends, back to 0..N-1
        ptr[m, 1:] = np.cumsum([len(s) for s in segs])  # segment bounds per vehicle
        order[m] = np.concatenate(segs)  # vehicle 0's stops, then vehicle 1's, ...
    return a, order, ptr


def visit_sequence(plans: Dense) -> np.ndarray:
    """(M, N + 2V) matrix-index sequence per particle: [0, pi_0, 0, 0, pi_1, 0, ...].
    Consecutive pairs are exactly the traversed (i, j); the (0, 0) joins between
    vehicles and inside empty vehicles cost 0 (zero diagonal). Pure array scatter."""
    a, order, ptr = plans
    M, N = order.shape
    V = ptr.shape[1] - 1
    # Start from all-depot (zeros); every vehicle block is [0, stops..., 0], so the
    # sequence has N customer slots plus 2 depot slots per vehicle.
    seq = np.zeros((M, N + 2 * V), dtype=np.int64)
    a_sorted = np.take_along_axis(a, order, axis=1)  # vehicle of k-th visit
    # k-th visit lands at k + (2 per preceding vehicle block) + 1 (leading depot).
    pos = np.arange(N) + 2 * a_sorted + 1  # each vehicle block adds 2 depot slots
    np.put_along_axis(seq, pos, order + 1, axis=1)  # +1: matrix indices 1..N
    return seq


def raw_terms(plans: Dense, traffic: TrafficState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, D, C) per particle [SPEC 7.2, v4 5]:
        T = sum T_path_ij(t),  D = sum D_path_ij,  C = sum rho_ij(t) * D_ij
    over consecutive (i, j) of every route, one gather per matrix for the whole swarm."""
    seq = visit_sequence(plans)
    i, j = seq[:, :-1], seq[:, 1:]  # all consecutive legs, shape (M, N + 2V - 1)
    d = traffic.distance[i, j]  # fancy-index gather, reused for D and C
    return traffic.duration[i, j].sum(1), d.sum(1), (traffic.congestion[i, j] * d).sum(1)


# -- route change ------------------------------------------------------------------
def plan_positions(plans: Dense) -> tuple[np.ndarray, np.ndarray]:
    """(pos, length): pos[m, j] = index of customer j inside its vehicle's route,
    length[m, j] = number of customers on that route."""
    a, order, ptr = plans
    M, N = order.shape
    a_sorted = np.take_along_axis(a, order, axis=1)  # vehicle of the k-th visit
    start = np.take_along_axis(ptr, a_sorted, axis=1)  # where that vehicle's block begins
    # Position within route = global rank k minus the block start; scatter back so
    # pos is indexed by customer id rather than by visit rank.
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(N) - start, axis=1)
    length = np.take_along_axis(np.diff(ptr, axis=1), a, axis=1)  # route length per customer
    return pos, length


def route_change(plans: Dense, current: Dense, eta_a: float, eta_o: float) -> np.ndarray:
    """R' per particle vs the deployed plan [SPEC v4 19]:
        A = (1/N) sum_j 1[a_j != a_j^cur]
        O = (1/N) sum_j 1[a_j == a_j^cur] |pos_j - pos_j^cur| / L_j,
            L_j = max(len_new(a_j), len_cur(a_j), 1)
        R' = eta_a A + eta_o O
    `current` is a single-particle Dense (M=1) and broadcasts over the swarm."""
    a, _, _ = plans
    a_cur, _, ptr_cur = current
    pos, length = plan_positions(plans)
    pos_cur, _ = plan_positions(current)
    # L_j normalises the position shift by the longer of the two routes involved.
    len_cur_new_vehicle = np.take_along_axis(np.diff(ptr_cur, axis=1), a, axis=1)
    L = np.maximum(np.maximum(length, len_cur_new_vehicle), 1)
    same = a == a_cur  # (M, N) mask: customer kept its vehicle
    A_chg = (~same).mean(axis=1)  # fraction reassigned
    O_chg = (same * np.abs(pos - pos_cur) / L).mean(axis=1)  # mean relative shift, stayers
    return eta_a * A_chg + eta_o * O_chg


# -- normalisation + combination -------------------------------------------------------
def compute_normalization_bounds(
    traffic: TrafficState,
    n_customers: int,
    n_vehicles: int,
    rng: np.random.Generator,
    n_samples: int = 200,
) -> Bounds:
    """Min/max of T, D, C over `n_samples` random particles [SPEC 7.2]. Called ONCE per
    scenario at setup; the result is frozen and passed into `evaluate`."""
    plans = decode_dense(random_particle(n_customers, rng, n_samples), n_vehicles)
    T, D, C = raw_terms(plans, traffic)
    return Bounds(
        t=(float(T.min()), float(T.max())),
        d=(float(D.min()), float(D.max())),
        c=(float(C.min()), float(C.max())),
    )


def _norm(x: np.ndarray, lo_hi: tuple[float, float]) -> np.ndarray:
    # Min-max scale; `or 1.0` guards a degenerate range (hi == lo) against divide-by-zero.
    lo, hi = lo_hi
    return (x - lo) / ((hi - lo) or 1.0)


def combine(
    T: np.ndarray,
    D: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    bounds: Bounds,
    cfg: QTrafficConfig,
    penalty: np.ndarray | float = 0.0,
) -> np.ndarray:
    """F = w_t T' + w_d D' + w_c C' + w_r R' (+ w_p P) [SPEC 7.2, v4 17, v4 22].
    Normalised terms may exceed [0, 1] for solutions outside the sampled bounds; that
    is intended (the bounds are a scale, not a clamp)."""
    F = (
        cfg.w_t * _norm(T, bounds.t)
        + cfg.w_d * _norm(D, bounds.d)
        + cfg.w_c * _norm(C, bounds.c)
        + cfg.w_r * _norm(R, bounds.r)
    )
    return F + cfg.w_p * penalty


def evaluate(
    plans: Dense | list[FleetRoute],
    traffic: TrafficState,
    current: FleetRoute | Dense | None,
    bounds: Bounds,
    cfg: QTrafficConfig,
    penalties: np.ndarray | None = None,
) -> np.ndarray:
    """One F per particle, vectorised over the batch [SPEC 7.2, v4 17].

    plans:     `decode_dense` output (preferred, zero copies) or a list of FleetRoute.
    current:   deployed plan for R'; None -> R' = 0 (cold start).
    bounds:    frozen per-scenario `Bounds` -- a pure input, never read from a global.
    penalties: optional (M,) residual violation P per particle -> F + w_p P [v4 22].
    Deterministic: same inputs -> bit-identical output. Never mutates any input.
    """
    if isinstance(plans, list):  # convenience path for callers holding FleetRoutes
        plans = to_dense(plans)
    T, D, C = raw_terms(plans, traffic)
    if current is None:
        R = np.zeros_like(T)  # cold start: no deployed plan to deviate from
    else:
        if isinstance(current, FleetRoute):
            current = to_dense([current])  # single-particle Dense, broadcasts in route_change
        R = route_change(plans, current, cfg.eta_a, cfg.eta_o)
    P = 0.0 if penalties is None else np.asarray(penalties, dtype=float)
    return combine(T, D, C, R, bounds, cfg, P)
