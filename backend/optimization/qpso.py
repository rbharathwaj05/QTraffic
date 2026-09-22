"""Canonical quantum-behaved PSO, time-boxed [SPEC 7.4-7.5, 11 steps 5-28; v4 24-27].

Vanilla QPSO only -- elite breeding / warm start are Phase 8 (`ebqpso.py`). One
iteration is exactly:

    mbest = (1/M) sum_i pbest_i                                    [SPEC 7.4, v4 25]
    p_i   = phi_i * pbest_i + (1 - phi_i) * gbest                  [SPEC 7.4, v4 24]
    x_i   = p_i + s_i * alpha^t * |mbest - x_i| * ln(1/u_i)        [SPEC 7.4, v4 26]
    x_i   = clip(x_i, 0, 1)                                        [v4 26]
    alpha^t = alpha_max - (t / T_proj) (alpha_max - alpha_min)     [SPEC 7.4, v4 27]

phi, u and the sign s are drawn fresh per particle AND per dimension every iteration;
every draw comes from the injected `rng`. The ln(1/u) long tail is the method -- it is
never truncated or bounded "for stability".

=====================================================================================
NON-WRITE-BACK [SPEC 8.3 / v4 10]: X, pbest, gbest and mbest are raw key vectors in
[0, 1]^(2N), forever. Decoding, repair and the 2-opt polish act on their OUTPUT only.
`two_opt` returns a FleetRoute for reporting / deployment and is NEVER re-encoded into
gbest. Anything typed `FleetRoute` in this module is a report artifact, not swarm state.
=====================================================================================

T in the alpha schedule is the PROJECTED iteration count under the wall-clock budget
[SPEC 7.4]: T_proj = budget / mean_iteration_time, re-estimated every
`cfg.alpha_update_every` iterations so alpha does not jitter with timing noise.
Wall-clock never enters F -- it only sets the budget and is logged on its own axis.
"""

# STATUS (through Phase 6): contract only. Signatures + docstrings define the formulas;
# every body raises NotImplementedError until its phase lands. Tests for this module skip.

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from backend.config import QTrafficConfig
from backend.constraints.checker import check_all
from backend.constraints.feasibility import Problem, arrival_times
from backend.constraints.repair import repair
from backend.optimization import fitness
from backend.optimization.encoding import FleetRoute, clip, decode, random_particle
from backend.optimization.fitness import ProblemContext

# Phase 0 alias still used by the `pso` / `ga` stubs; QPSO itself takes an `Evaluator`
# (see below) because a bare float per particle cannot carry the repaired route.
FitnessFn = Callable[[np.ndarray], np.ndarray]  # (M, D) -> (M,)


# -- decode -> repair -> F, one batch --------------------------------------------------
@dataclass
class EvalBatch:
    """One evaluation of a whole swarm. `fleets` are REPAIRED routes (report artifacts);
    the swarm keys that produced them stay untouched [SPEC 8.3]."""

    F: np.ndarray  # (M,) F_eval = F + w_p P [v4 22]
    fleets: list[FleetRoute]
    penalty: np.ndarray  # (M,) residual violation P after repair
    repair_distance: np.ndarray  # (M,) d_R [v4 23]
    repair_iterations: np.ndarray  # (M,) int
    capped_out: np.ndarray  # (M,) bool -> RepairFailureRate = mean(capped_out)


@dataclass
class Evaluator:
    """decode (Phase 4) -> repair (Phase 6) -> fitness (Phase 5) for a whole swarm.

    Replaces the Phase 0 `FitnessFn = (M, D) -> (M,)` contract: a bare float per particle
    cannot carry the repaired route that gets deployed, nor the repair diagnostics that
    SPEC 8.4 / Phase 13 require, and both must be visible to `run()` (correctness bar:
    "if repair capped out for the reported gbest, that must be surfaced, not hidden")."""

    ctx: ProblemContext
    prob: Problem
    cfg: QTrafficConfig
    strategy: str = "minimal"

    def __call__(self, X: np.ndarray) -> EvalBatch:
        fleets = decode(X, self.ctx.n_vehicles)
        # ponytail: Python loop over particles -- repair() is per-candidate by SPEC 8.3/8.4
        # (a cascade with its own control flow), so it cannot be one array op. ~8 ms per
        # particle at S4; the decode and F passes around it are fully vectorised.
        res = [repair(f, self.prob, self.cfg, self.strategy) for f in fleets]
        P = np.array([r.residual for r in res])
        F = fitness.evaluate(
            [r.fleet for r in res],
            self.ctx.traffic,
            self.ctx.current,
            self.ctx.bounds,
            self.cfg,
            penalties=P,
        )
        return EvalBatch(
            F=F,
            fleets=[r.fleet for r in res],
            penalty=P,
            repair_distance=np.array([r.distance for r in res]),
            repair_iterations=np.array([r.iterations for r in res]),
            capped_out=np.array([r.capped_out for r in res]),
        )


# -- 2-opt polish of the reported route -------------------------------------------------
def route_duration(route: np.ndarray, prob: Problem) -> float:
    """sum c_{i,j} over consecutive stops; inf if any leg is unroutable."""
    return float(prob.duration[route[:-1], route[1:]].sum())


def two_opt(fleet: FleetRoute, prob: Problem, cfg: QTrafficConfig) -> FleetRoute:
    """Best-improvement 2-opt on each route of a DECODED fleet [SPEC 7.5].

    Applied to exactly one solution -- the one that will be reported -- and its result is
    never encoded back into gbest. Segment reversal is scored by recomputing the route's
    total duration (the OSRM matrix is asymmetric, so the symmetric delta shortcut would
    be wrong), and a move is accepted only if it keeps the route time-feasible and
    routable. Bounded by `cfg.two_opt_max_passes`; no unbounded loop.
    """
    out = []
    for v, r in enumerate(fleet.routes):
        r = r.copy()
        for _ in range(cfg.two_opt_max_passes):
            base = route_duration(r, prob)
            best, best_cost = None, base
            for i in range(1, len(r) - 2):
                for j in range(i + 1, len(r) - 1):
                    cand = np.concatenate([r[:i], r[i : j + 1][::-1], r[j + 1 :]])
                    cost = route_duration(cand, prob)
                    if cost < best_cost - 1e-9:
                        A = arrival_times(
                            cand, prob.duration, prob.service, prob.windows, prob.t_start[v]
                        )
                        if (A <= prob.windows[cand, 1]).all() and A[-1] <= prob.shift_end[v]:
                            best, best_cost = cand, cost
            if best is None:
                break
            r = best
        out.append(r)
    return FleetRoute(out, fleet.assignment.copy())


# -- swarm state ------------------------------------------------------------------------
@dataclass
class QPSOState:
    """Raw-key swarm state [SPEC 7.4]. Every field below is an np.ndarray in [0, 1]^(2N)
    (or a stack of them) -- never a FleetRoute [SPEC 8.3]."""

    X: np.ndarray  # (M, 2N) positions
    pbest: np.ndarray  # (M, 2N) personal bests, raw keys
    pbest_F: np.ndarray  # (M,)
    gbest: np.ndarray  # (2N,) raw keys
    gbest_F: float
    mbest: np.ndarray  # (2N,) mean of pbest [v4 25]
    rng: np.random.Generator
    alpha: float = 1.0
    T_proj: float = 0.0  # projected iteration count driving the alpha schedule
    injection_cooldown: int = 0  # iterations left in which stagnation cannot fire [SPEC 7.5]


@dataclass
class SwarmResult:
    """Uniform return type for every algorithm (QPSO, EB-QPSO, PSO, GA, ACO)."""

    gbest: np.ndarray  # raw key vector [SPEC 8.3]
    gbest_fitness: float
    history: np.ndarray  # F per iteration        [SPEC 13.1 curve 1]
    wallclock: np.ndarray  # elapsed s per iteration [SPEC 13.1 curve 2]
    iterations: int
    elapsed_s: float
    stopped_by: str  # "budget" | "iterations" | "stagnation"
    fleet: FleetRoute | None = None  # deployed route: repaired, then 2-opt polished
    feasible: bool = False  # checker all-clear on `fleet`
    capped_out: bool = False  # repair hit the cap for the reported gbest
    residual: float = 0.0  # P of the reported gbest
    diagnostics: dict[str, float] = field(default_factory=dict)


# -- the optimiser ------------------------------------------------------------------------
class QPSO:
    """Canonical QPSO [SPEC 7.4-7.5]. EB-QPSO (Phase 8) subclasses this."""

    def __init__(
        self, cfg: QTrafficConfig, evaluator: Evaluator, dim_or_n: int, rng: np.random.Generator
    ) -> None:
        """`dim_or_n` is the particle dimension D = 2N (`encoding.dim(N)`)."""
        self.cfg = cfg
        self.evaluate = evaluator
        self.dim = dim_or_n
        self.n_customers = dim_or_n // 2
        self.rng = rng
        self.state: QPSOState | None = None

    # -- 7.1 initialisation ---------------------------------------------------------
    def initialize(self, seed_particles: np.ndarray | None = None) -> EvalBatch:
        """M uniform particles (first rows replaced by `seed_particles` when given, the
        Phase 8 warm-start hook); evaluate once and set pbest / gbest / mbest."""
        X = random_particle(self.n_customers, self.rng, self.cfg.M)
        if seed_particles is not None:
            s = np.atleast_2d(seed_particles)
            X[: len(s)] = clip(s)
        batch = self.evaluate(X)
        g = int(np.argmin(batch.F))
        self.state = QPSOState(
            X=X,
            pbest=X.copy(),
            pbest_F=batch.F.copy(),
            gbest=X[g].copy(),
            gbest_F=float(batch.F[g]),
            mbest=X.mean(axis=0),
            rng=self.rng,
            alpha=self.cfg.alpha_max,
        )
        self._best_fleet = batch.fleets[g]
        self._best_capped = bool(batch.capped_out[g])
        return batch

    # -- 7.5 alpha ------------------------------------------------------------------
    def alpha(self, t: int, T: float) -> float:
        """alpha^t = alpha_max - (t / T)(alpha_max - alpha_min) [SPEC 7.4, v4 27].
        Clamped to [alpha_min, alpha_max] so an over-run past T cannot flip the sign."""
        c = self.cfg
        a = c.alpha_max - (t / max(T, 1.0)) * (c.alpha_max - c.alpha_min)
        return float(np.clip(a, c.alpha_min, c.alpha_max))

    # -- 7.3 mbest ------------------------------------------------------------------
    def mean_best(self) -> np.ndarray:
        """mbest = (1/M) sum_i pbest_i [SPEC 7.4, v4 25]. Raw keys only -- meaningful
        exactly because no repaired route is ever written into pbest [SPEC 8.3]."""
        return self.state.pbest.mean(axis=0)

    # -- 7.2 local attractor --------------------------------------------------------
    def local_attractor(self) -> np.ndarray:
        """p_i = phi_i pbest_i + (1 - phi_i) gbest, phi ~ U(0,1) per particle and
        per dimension, fresh every call [SPEC 7.4, v4 24]."""
        s = self.state
        phi = s.rng.random(s.X.shape)
        return phi * s.pbest + (1.0 - phi) * s.gbest

    # -- 7.4 position update ---------------------------------------------------------
    def update_positions(self, alpha: float) -> None:
        """x_i = p_i +- alpha |mbest - x_i| ln(1/u_i), u ~ U(0,1), then clip to [0,1]
        [SPEC 7.4, v4 26]. Sign and u are per-dimension; the ln(1/u) tail is unbounded
        by design."""
        s = self.state
        p = self.local_attractor()
        u = s.rng.random(s.X.shape)  # in [0, 1); shifted off 0 to keep ln finite
        u = np.where(u > 0.0, u, np.finfo(float).tiny)
        sign = np.where(s.rng.random(s.X.shape) < 0.5, -1.0, 1.0)
        s.X = clip(p + sign * alpha * np.abs(s.mbest - s.X) * np.log(1.0 / u))

    # -- 7.6 stagnation / diversity ---------------------------------------------------
    def diversity(self) -> float:
        """Diversity_t = (1/M) sum_i ||x_i - mean(x)|| [SPEC 7.5]."""
        X = self.state.X
        return float(np.linalg.norm(X - X.mean(axis=0), axis=1).mean())

    def stagnated(self, history: list[float]) -> bool:
        """True when gbest improved by < epsilon_stag (relative) over the last
        `cfg.patience` iterations [SPEC 7.5].

        Suppressed while `state.injection_cooldown > 0`: the patience window would
        otherwise still span the pre-injection plateau and re-flag stagnation on the
        very next iteration, ending the run ~1 iteration after the injection meant to
        rescue it. The cooldown, not the window, is the invariant -- see `run()`.
        """
        if self.state is not None and self.state.injection_cooldown > 0:
            return False
        p = self.cfg.patience
        if len(history) <= p:
            return False
        old, new = history[-p - 1], history[-1]
        return (old - new) / max(abs(old), 1e-12) < self.cfg.epsilon_stag

    def inject_diversity(self) -> int:
        """Re-initialise the worst `r_inject` fraction of particles [SPEC 7.5]; gbest and
        the top performers are untouched. Their pbest is reset with them (F = inf) so the
        fresh positions define the new personal best on the next evaluation."""
        s = self.state
        k = max(1, int(round(self.cfg.r_inject * self.cfg.M)))
        worst = np.argsort(s.pbest_F)[-k:]
        s.X[worst] = random_particle(self.n_customers, s.rng, k)
        s.pbest[worst] = s.X[worst]
        s.pbest_F[worst] = np.inf
        # [SPEC 7.5] give the fresh particles a full patience window to show progress
        # before stagnation may fire again; `cfg.patience` is reused rather than adding a
        # second knob -- a shorter cooldown cannot clear the plateau from the window.
        s.injection_cooldown = self.cfg.patience
        return k

    # -- pbest / gbest ----------------------------------------------------------------
    def _absorb(self, batch: EvalBatch) -> None:
        """pbest <- elementwise min, gbest <- swarm min. Min-preserving by construction,
        so the gbest trajectory is monotone non-increasing."""
        s = self.state
        better = batch.F < s.pbest_F
        s.pbest[better] = s.X[better]
        s.pbest_F[better] = batch.F[better]
        g = int(np.argmin(batch.F))
        if batch.F[g] < s.gbest_F:
            s.gbest = s.X[g].copy()
            s.gbest_F = float(batch.F[g])
            self._best_fleet = batch.fleets[g]
            self._best_capped = bool(batch.capped_out[g])

    # -- extension hooks (Phase 8) ------------------------------------------------------
    def post_absorb(self, t: int, batch: EvalBatch) -> None:
        """Called once per iteration after pbest/gbest are updated and before the history
        entry is recorded. No-op in plain QPSO; EB-QPSO hangs steps 7-13 of its iteration
        (mbest, elites, breeding, replacement) here so it composes this loop instead of
        re-implementing it [v4 36]."""

    def extra_diagnostics(self) -> dict[str, float]:
        """Per-algorithm additions to `SwarmResult.diagnostics`. Empty for plain QPSO."""
        return {}

    # -- 7.7 main loop -----------------------------------------------------------------
    def run(
        self,
        T: int | None = None,
        time_budget_s: float | None = None,
        seed_particles: np.ndarray | None = None,
    ) -> SwarmResult:
        """Iterate until the iteration cap T (default `cfg.T_iter_max`), the wall-clock
        budget, or stagnation that survives a diversity injection (never before
        `cfg.T_iter_min`) [SPEC 11 steps 5-28, plain-QPSO subset].

        `seed_particles` is the Phase 10 warm start [v4 53-54]: rows handed to
        `initialize`, the rest of the swarm cold. The uniform `run(T, time_budget_s)`
        signature every algorithm shares is unchanged -- `benchmark.py` never passes it.

        The loop stops early when the next iteration would not fit in the remaining
        budget. The post-loop work (final 2-opt polish + `check_all` on the reported
        route) is deliberately OUTSIDE that accounting -- the deployed route must be
        polished and checked whatever the budget says -- so `elapsed_s` can exceed
        `time_budget_s` by that tail. The tail is a function of route length, not of the
        budget: measured < 1 ms on S1 (5 seeds, 2 s and 10 s budgets; worst observed
        `elapsed_s` = 100 % of budget, never above). The 10 % tolerance in the tests is
        headroom for bigger scenarios, not the measured overshoot. Against the
        `T_response_local = 10 s` bound of SPEC 9.1 this tail is noise.
        """
        cfg = self.cfg
        T_cap = cfg.T_iter_max if T is None else T
        budget = np.inf if time_budget_s is None else float(time_budget_s)
        t0 = time.perf_counter()

        batch = self.initialize(seed_particles)  # Phase 10 warm start [v4 53]
        self._absorb(batch)
        s = self.state
        history, wallclock = [s.gbest_F], [time.perf_counter() - t0]
        injections, stopped = 0, "iterations"
        s.T_proj = float(T_cap)

        for t in range(1, T_cap + 1):
            elapsed = time.perf_counter() - t0
            per_iter = elapsed / max(len(history) - 1, 1)
            if elapsed + per_iter > budget:  # next iteration would overrun [SPEC 8]
                stopped = "budget"
                break
            if t % cfg.alpha_update_every == 1 or cfg.alpha_update_every == 1:
                # project the iteration count the budget actually affords [SPEC 7.4]
                s.T_proj = min(float(T_cap), budget / per_iter) if np.isfinite(budget) else T_cap
            s.injection_cooldown = max(0, s.injection_cooldown - 1)
            s.alpha = self.alpha(t, s.T_proj)
            s.mbest = self.mean_best()
            self.update_positions(s.alpha)
            batch = self.evaluate(s.X)
            self._absorb(batch)
            self.post_absorb(t, batch)  # EB-QPSO breeds + replaces here [v4 36 steps 7-13]
            history.append(s.gbest_F)
            wallclock.append(time.perf_counter() - t0)

            if self.stagnated(history) or self.diversity() < cfg.delta_div:
                if injections and t >= cfg.T_iter_min:
                    stopped = "stagnation"  # stagnated again after an injection
                    break
                # Option 1 of the fix: `inject_diversity` arms a patience-long cooldown in
                # which `stagnated()` cannot fire. Without it the pre-injection plateau is
                # still inside the patience window next iteration, so the run exits one
                # iteration after the injection and burns ~15 % of its budget. A counter is
                # a single invariant to test; truncating the history window at the
                # injection index (Option 2) re-derives the same rule in slicing logic.
                self.inject_diversity()
                injections += 1
                # no in-loop 2-opt: `_best_fleet` is polished once after the loop, and any
                # mid-loop polish is either overwritten by the next gbest or redone there.
        else:
            stopped = "iterations"

        fleet = two_opt(self._best_fleet, self.evaluate.prob, cfg)
        report = check_all(fleet, self.evaluate.prob)
        return SwarmResult(
            gbest=s.gbest.copy(),
            gbest_fitness=s.gbest_F,
            history=np.array(history),
            wallclock=np.array(wallclock),
            iterations=len(history) - 1,
            elapsed_s=time.perf_counter() - t0,
            stopped_by=stopped,
            fleet=fleet,
            feasible=report.ok,
            capped_out=self._best_capped,
            residual=report.total,
            diagnostics={
                "injections": float(injections),
                # mean, because EB-QPSO's block-aware alpha [v4 28] makes this a per-block
                # vector; for plain QPSO the mean of a scalar is that scalar.
                "alpha_final": float(np.mean(s.alpha)),
                "T_projected": s.T_proj,
                "diversity_final": self.diversity(),
                "repair_failure_rate": float(batch.capped_out.mean()),
                "repair_distance_mean": float(batch.repair_distance.mean()),
                "repair_iterations_mean": float(batch.repair_iterations.mean()),
                **self.extra_diagnostics(),
            },
        )
