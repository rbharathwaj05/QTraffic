"""Elite-Breeding QPSO = Phase 7 QPSO + an elite crossover layer [SPEC v4 28-36].

ADDITIVE BY CONSTRUCTION. Every QPSO step (position update, decode, repair, evaluate,
pbest/gbest, mbest, stagnation, injection, budget accounting) is INHERITED from
`qpso.QPSO` and called, never re-implemented here. This module only adds steps 7-13 of
the v4 §36 iteration, hung off the `post_absorb` hook:

     1 QPSO update -> 2 decode -> 3 repair -> 4 evaluate -> 5 pbest -> 6 gbest   (QPSO)
     7 mbest -> 8 elites -> 9 breed -> 10 decode -> 11 repair -> 12 evaluate
       -> 13 replace weak                                                        (here)
    14 stagnation / diversity -> 15 continue until the budget                    (QPSO)

Steps 10-12 are a SEPARATE `Evaluator` call from steps 2-4 so the ablation in Phase 13
can tell "a child capped out in repair" from "a regular particle capped out" -- the two
counts are reported separately in `SwarmResult.diagnostics`.

With `cfg.r_B = 0` no child is bred, no extra random number is drawn, and this class is
numerically identical to plain QPSO for the same seed. That equivalence is the Phase 13
breeding ablation and is pinned by `test_ebqpso.py`.

NON-WRITE-BACK [SPEC 8.3] holds unchanged: parents are raw `pbest` key vectors, children
are raw key vectors, and a surviving child enters `X`/`pbest`/`gbest` as keys only. No
repaired or 2-opt-polished route is ever encoded back.
"""

from __future__ import annotations

import math

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization.encoding import clip
from backend.optimization.qpso import QPSO, EvalBatch, Evaluator


class EBQPSO(QPSO):
    """QPSO with elitist breeding [v4 29-36] and block-aware alpha [v4 28]."""

    def __init__(
        self, cfg: QTrafficConfig, evaluator: Evaluator, dim_or_n: int, rng: np.random.Generator
    ) -> None:
        super().__init__(cfg, evaluator, dim_or_n, rng)
        self.children_bred = 0
        self.children_accepted = 0
        self._child_capped = 0

    # -- v4 28 block-aware alpha ---------------------------------------------------
    def alpha(self, t: int, T: float) -> float | np.ndarray:
        """alpha_d = alpha_A on the assignment block, alpha_O on the ordering block, each
        on its own linear schedule [v4 28]. Returns a scalar when `cfg.block_alpha` is off
        (then this is exactly `QPSO.alpha`), else a (2N,) vector that broadcasts over the
        position update unchanged.

        Both per-block schedules default to the shared alpha_max/alpha_min, so
        block_alpha=True with default config is numerically identical to block_alpha=False
        -- the toggle exists so Phase 13 can separate this effect from breeding after the
        endpoints are tuned.
        """
        a_assign = super().alpha(t, T)  # [SPEC 7.4, v4 27]
        if not self.cfg.block_alpha:
            return a_assign
        c = self.cfg
        a_order = c.alpha_max_order - (t / max(T, 1.0)) * (c.alpha_max_order - c.alpha_min_order)
        a_order = float(np.clip(a_order, c.alpha_min_order, c.alpha_max_order))
        out = np.empty(self.dim)
        out[: self.n_customers] = a_assign  # Y block: assignment keys
        out[self.n_customers :] = a_order  # Z block: ordering keys
        return out

    # -- v4 29 elite set -------------------------------------------------------------
    def select_elites(self) -> np.ndarray:
        """E_t = top_k({pbest_1..pbest_M}), N_E = ceil(r_E * M) [v4 29].

        Returns particle INDICES ordered best-first by `pbest_F`; the elite vectors
        themselves are `state.pbest[idx]`, raw keys as always [SPEC 8.3].
        """
        n_e = min(self.cfg.M, max(1, math.ceil(self.cfg.r_E * self.cfg.M)))
        return np.argsort(self.state.pbest_F, kind="stable")[:n_e]

    # -- v4 30-32 breeding -----------------------------------------------------------
    def breed(self, elite_idx: np.ndarray) -> np.ndarray:
        """N_child = ceil(r_B * M) children from distinct elite parents [v4 30-32].

            A, B ~ E_t, A != B                                              [v4 30]
            Y_c = clip(lambda_A Y_A + (1 - lambda_A) Y_B + sigma_A eps_A, 0, 1)  [v4 31]
            Z_c = clip(lambda_O Z_A + (1 - lambda_O) Z_B + sigma_O eps_O, 0, 1)  [v4 31]
            X_c = [Y_c, Z_c]

        lambda_A and lambda_O are drawn fresh per child and per BLOCK (one scalar each,
        not per dimension -- the blend of two parents is a whole-block convex combination);
        eps ~ N(0, I) is per dimension. Parents are sampled WITHOUT replacement via the
        first two entries of a per-child random permutation of the elite set, which makes
        A == B impossible by construction; v4 §30 calls this out because sampling with
        replacement degenerates into breeding from the global best alone.

        Returns a (N_child, 2N) array; (0, 2N) when breeding is off (r_B = 0) or the elite
        set is too small to give two distinct parents -- and in that case NO random number
        is drawn, which is what keeps the r_B = 0 ablation bit-identical to plain QPSO.
        """
        n_child = math.ceil(self.cfg.r_B * self.cfg.M)
        if n_child <= 0 or len(elite_idx) < 2:
            return np.empty((0, self.dim))

        s, c, n = self.state, self.cfg, self.n_customers
        # [v4 30] two DISTINCT elites per child: argsort of a random row is a permutation
        order = np.argsort(s.rng.random((n_child, len(elite_idx))), axis=1)[:, :2]
        A, B = s.pbest[elite_idx[order[:, 0]]], s.pbest[elite_idx[order[:, 1]]]

        lam = s.rng.random((n_child, 2))  # [:, 0] = lambda_A, [:, 1] = lambda_O
        lam_d = np.empty((n_child, self.dim))
        lam_d[:, :n], lam_d[:, n:] = lam[:, :1], lam[:, 1:]
        sigma = np.concatenate([np.full(n, c.sigma_A), np.full(self.dim - n, c.sigma_O)])
        eps = s.rng.standard_normal((n_child, self.dim))
        return clip(lam_d * A + (1.0 - lam_d) * B + sigma * eps)  # [v4 31]

    # -- v4 34-35 replacement --------------------------------------------------------
    def replace_weak(self, children: np.ndarray, child_batch: EvalBatch, F: np.ndarray) -> int:
        """Best child challenges worst particle; population stays exactly M [v4 34-35].

        The swarm is ranked by the CURRENT iteration's F, the `len(children)` worst are the
        challenge set W, and children are matched best-to-worst. A child replaces its
        target only if `F_c < F_w` (v4 §35: "No child is accepted merely because its
        parents were elite"); otherwise it is discarded. Nothing is appended -- v4 §34
        fixes the population at M to stop breeding growing the swarm without bound.

        A surviving child updates pbest and gbest by the ordinary QPSO rules: it has just
        been evaluated through the same decode -> repair -> F pipeline, so its F is
        comparable with every other particle's.
        """
        if len(children) == 0:
            return 0
        s = self.state
        k = len(children)
        worst = np.argsort(F, kind="stable")[-k:][::-1]  # worst first
        best_child = np.argsort(child_batch.F, kind="stable")  # best first
        accepted = 0
        for w, c in zip(worst, best_child):  # k = ceil(r_B * M) = 4 at M = 50
            if not child_batch.F[c] < F[w]:
                continue  # [v4 35] elite parentage buys nothing; the child is discarded
            accepted += 1
            s.X[w] = children[c]
            if child_batch.F[c] < s.pbest_F[w]:
                s.pbest[w] = children[c]
                s.pbest_F[w] = child_batch.F[c]
            if child_batch.F[c] < s.gbest_F:
                s.gbest = children[c].copy()
                s.gbest_F = float(child_batch.F[c])
                self._best_fleet = child_batch.fleets[c]
                self._best_capped = bool(child_batch.capped_out[c])
        return accepted

    # -- v4 36 steps 7-13 --------------------------------------------------------------
    def post_absorb(self, t: int, batch: EvalBatch) -> None:
        """mbest -> elites -> breed -> decode/repair/evaluate the children -> replace."""
        s, cfg = self.state, self.cfg
        s.mbest = self.mean_best()  # step 7  [v4 25]
        elites = self.select_elites()  # step 8  [v4 29]
        children = self.breed(elites)  # step 9  [v4 30-32]
        if len(children) == 0:
            return
        # steps 10-12: children go through the SAME pipeline, in their own batch call so
        # their repair diagnostics stay separable from the main swarm's [v4 33]
        child_batch = self.evaluate(children)
        self.children_bred += len(children)
        self._child_capped += int(child_batch.capped_out.sum())
        self.children_accepted += self.replace_weak(children, child_batch, batch.F)  # step 13
        assert len(s.X) == cfg.M, "breeding must not change the population size [v4 34]"

    def extra_diagnostics(self) -> dict[str, float]:
        """Breeding counters for the Phase 13 ablation; `child_repair_failure_rate` is kept
        apart from the swarm's `repair_failure_rate` on purpose [v4 33]."""
        bred = self.children_bred
        return {
            "children_bred": float(bred),
            "children_accepted": float(self.children_accepted),
            "child_acceptance_rate": float(self.children_accepted / bred) if bred else 0.0,
            "child_repair_failure_rate": float(self._child_capped / bred) if bred else 0.0,
        }
