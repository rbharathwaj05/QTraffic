"""Edge-level congestion model: events + background load -> per-edge time factor.

Core model [SPEC 7.2 / v4 3], all per edge e and sim time t:

    rho_e(t) in [0, rho_congestion_max)          <- cfg.rho_congestion_max, read live
    V_e(t)   = V_normal_e * (1 - rho_e(t))
    T_e(t)   = D_e / V_e(t) = T_e^0 / (1 - rho_e(t))

so the travel-time FACTOR carried through `cost_matrix.update_factors` is
f_e = T_e(t) / T_e^0 = 1 / (1 - rho_e(t)), and the inverse is rho = 1 - 1/f. The cap on
rho is what keeps f finite; a full closure is represented as severity 1 -> f = inf, which
is handled in `events.severity_to_factor`, not by pushing rho to 1.

FLEET-INDUCED congestion (`bpr_factor`, `fleet_rho`) is NOT part of the core v3/v4 spec.
It comes from the architecture doc (doc2 9) and stays behind `cfg.enable_fleet_congestion`
so it can be switched off without touching any of the degradation math above.
"""

from __future__ import annotations

import numpy as np

from backend.road.path_index import PathIndex
from backend.traffic.events import TrafficEvent, severity_to_factor

DAY_S = 86400.0


# -- core model [SPEC 7.2 / v4 3] --------------------------------------------------
def speed(v_normal: np.ndarray, rho: np.ndarray, rho_congestion_max: float) -> np.ndarray:
    """V_ij(t) = V_normal_ij * (1 - rho_ij(t)), rho clipped to [0, rho_congestion_max)
    [SPEC 7.2]. The cap is `cfg.rho_congestion_max` and has NO default here: a silent
    hard-coded ceiling is what let the configured one go unread in the first place."""
    return np.asarray(v_normal, float) * (1.0 - clip_rho(rho, rho_congestion_max))


def travel_time(distance: np.ndarray, v: np.ndarray) -> np.ndarray:
    """T_ij(t) = D_ij / V_ij(t) [SPEC 7.2]; inf where the speed has collapsed to 0."""
    v = np.asarray(v, float)
    return np.where(v > 0, np.asarray(distance, float) / np.where(v > 0, v, 1.0), np.inf)


def clip_rho(rho: np.ndarray, rho_congestion_max: float) -> np.ndarray:
    """rho in [0, rho_congestion_max) [SPEC 7.2] -- THE one clamp in the system.

    The upper end is open, so the cap is applied just below the ceiling: at exactly
    rho_congestion_max the speed model is still finite, but the spec writes the interval
    half-open and every downstream 1/(1-rho) assumes it. `rho_congestion_max` is
    `cfg.rho_congestion_max`, never the vehicle-LOAD ratio `cfg.rho_max` -- the two are
    unrelated quantities that happen to share a letter, so they never share a name.

    Every other clamp in `traffic/` calls this function; none of them re-derives it.
    """
    return np.clip(np.asarray(rho, float), 0.0, np.nextafter(rho_congestion_max, 0.0))


def rho_to_factor(rho: np.ndarray, rho_congestion_max: float) -> np.ndarray:
    """f_e = T_e(t)/T_e^0 = 1 / (1 - rho_e(t)), rho capped first [SPEC 7.2]."""
    return 1.0 / (1.0 - clip_rho(rho, rho_congestion_max))


def factor_to_rho(factor: np.ndarray) -> np.ndarray:
    """Inverse of `rho_to_factor`: rho = 1 - 1/f. f = inf (closure) -> rho = 1."""
    f = np.asarray(factor, float)
    return np.where(np.isfinite(f), 1.0 - 1.0 / np.where(f > 0, f, 1.0), 1.0)


# -- background load ----------------------------------------------------------------
def bpr_factor(
    volume: np.ndarray, capacity: np.ndarray, a: float = 0.15, b: float = 4.0
) -> np.ndarray:
    """Bureau of Public Roads: t/t0 = 1 + a (v/c)^b (doc2 9, NOT core spec).

    Used for the background diurnal load and, when `cfg.enable_fleet_congestion` is on,
    for the fleet's own contribution to the volume.
    """
    c = np.asarray(capacity, float)
    return 1.0 + a * (np.asarray(volume, float) / np.where(c > 0, c, np.inf)) ** b


def diurnal_volume(t_sim: float, base_volume: np.ndarray) -> np.ndarray:
    """Background volume v(t): two commuter peaks at 08:00 and 17:00, trough overnight
    (spec: congestion model, time of day). Returns `base_volume` scaled by a factor in
    [0.5, 1.5] -- the shape is a modelling choice, the scale lives in `base_volume`."""
    h = (t_sim % DAY_S) / 3600.0
    peaks = np.exp(-(((h - 8.0) / 1.5) ** 2)) + np.exp(-(((h - 17.0) / 1.5) ** 2))
    return np.asarray(base_volume, float) * (0.5 + peaks)


def fleet_rho(flow: np.ndarray, capacity: np.ndarray, rho_congestion_max: float) -> np.ndarray:
    """rho_e = flow_e / capacity_e from the fleet's own vehicles (doc2 9, NOT core spec).

    Isolated on purpose: nothing in the degradation/hysteresis path calls this, so
    `cfg.enable_fleet_congestion = False` removes the feature completely.
    """
    c = np.asarray(capacity, float)
    return clip_rho(np.asarray(flow, float) / np.where(c > 0, c, np.inf), rho_congestion_max)


# -- composition ---------------------------------------------------------------------
def compose_factors(
    background: np.ndarray,
    active_events: list[TrafficEvent],
    n_edges: int,
    rho_congestion_max: float,
) -> np.ndarray:
    """f_e = background_e * prod_{events on e} severity_to_factor(...), then capped
    (spec: factor composition; cap per [SPEC 7.2]).

    Events on the same edge multiply, so two 50 % slowdowns are worse than one. The
    PRODUCT is then pushed back through the cap: stacking events must not walk the
    implied rho past `rho_congestion_max` any more than a single event may. Without this
    the composition was the second way round the ceiling, after `severity_to_factor`.

    A closure (inf) is the ONE deliberate exception: it means "no path", not a congestion
    level, so it is not a rho the cap applies to and it survives composition untouched.
    """
    f = np.ones(n_edges) if background is None else np.array(background, float)
    if f.shape != (n_edges,):
        raise ValueError(f"background must be ({n_edges},), got {f.shape}")
    for ev in active_events:
        e = np.asarray(ev.edge_ids, dtype=np.int64)
        f[e] = f[e] * severity_to_factor(ev.severity, ev.kind, rho_congestion_max)
    finite = np.isfinite(f)  # inf = closure, exempt by definition
    f[finite] = np.maximum(1.0, rho_to_factor(factor_to_rho(f[finite]), rho_congestion_max))
    return f


def background_factors(
    t_sim: float,
    base_volume: np.ndarray,
    capacity: np.ndarray,
    cfg=None,
    fleet_flow: np.ndarray | None = None,
) -> np.ndarray:
    """Per-edge background factor at sim time `t_sim`: BPR over the diurnal volume, plus
    the fleet's own flow only when `cfg.enable_fleet_congestion` is set (doc2 9)."""
    a = 0.15 if cfg is None else cfg.bpr_alpha
    b = 4.0 if cfg is None else cfg.bpr_beta
    volume = diurnal_volume(t_sim, base_volume)
    if cfg is not None and cfg.enable_fleet_congestion and fleet_flow is not None:
        volume = volume + np.asarray(fleet_flow, float)  # doc2 9 enhancement, off by default
    return bpr_factor(volume, capacity, a, b)


def affected_pairs(events: list[TrafficEvent], index: PathIndex) -> set[tuple[int, int]]:
    """OD pairs whose cached path crosses any edge in `events` [SPEC 10.3].

    Straight `PathIndex` lookup -- the one structure that answers this. No geographic
    proximity, no second index, no matrix rebuild.
    """
    edges = [int(e) for ev in events for e in np.asarray(ev.edge_ids).ravel()]
    return index.pairs_using(edges)


def congestion_level(factor: np.ndarray) -> np.ndarray:
    """Discretise f_e into {0: free, 1: moderate (>1.2), 2: heavy (>1.5), 3: blocked (inf)}
    for the frontend heat layer (spec: visualisation)."""
    f = np.asarray(factor, float)
    lvl = np.zeros(f.shape, dtype=np.int64)
    lvl[f > 1.2] = 1
    lvl[f > 1.5] = 2
    lvl[~np.isfinite(f)] = 3
    return lvl
