"""Run QPSO or EB-QPSO on one scenario under a wall-clock budget [SPEC 7.4-7.5, 11; v4 36].

Usage: python scripts/run_qpso.py [--algo qpso|ebqpso] --scenario S1 --budget 5
                                 [--seed 0] [--swarm 50] [--no-breeding]

Prints gbest_F, iterations, wall-clock and feasibility, and writes the two mandatory
convergence curves [SPEC 13.1] to benchmarks/qpso_<scenario>_seed<seed>.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from backend.config import QTrafficConfig
from backend.optimization.benchmark import load_scenario
from backend.optimization.ebqpso import EBQPSO
from backend.optimization.encoding import dim
from backend.optimization.qpso import QPSO, Evaluator

BENCHMARK_DIR = Path("benchmarks")
ALGORITHMS = {"qpso": QPSO, "ebqpso": EBQPSO}


def write_curves(result, out: Path) -> Path:
    """cost-vs-iteration and cost-vs-wallclock in one tidy CSV [SPEC 13.1]."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "elapsed_s", "gbest_F"])
        w.writerows(zip(range(len(result.history)), result.wallclock, result.history))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="qpso", choices=sorted(ALGORITHMS))
    p.add_argument("--scenario", default="S1")
    p.add_argument("--budget", type=float, default=5.0, help="wall-clock seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swarm", type=int, default=None, help="M, default cfg.M")
    p.add_argument(
        "--no-breeding",
        action="store_true",
        help="r_B = 0: the Phase 13 ablation; EB-QPSO then reduces to plain QPSO [v4 32]",
    )
    a = p.parse_args()

    over = {}
    if a.swarm:
        over["M"] = a.swarm
    if a.no_breeding:
        over["r_B"] = 0.0
    cfg = QTrafficConfig(**over)
    rng = np.random.default_rng(a.seed)
    ctx, prob = load_scenario(a.scenario, cfg, rng)
    opt = ALGORITHMS[a.algo](cfg, Evaluator(ctx, prob, cfg), dim(ctx.n_customers), rng)
    res = opt.run(time_budget_s=a.budget)

    out = write_curves(res, BENCHMARK_DIR / f"{a.algo}_{a.scenario}_seed{a.seed}.csv")
    print(
        f"algo={a.algo} scenario={a.scenario} N={ctx.n_customers} M_veh={ctx.n_vehicles} M={cfg.M} "
        f"seed={a.seed}\n"
        f"gbest_F={res.gbest_fitness:.6f} iterations={res.iterations} "
        f"wall_clock={res.elapsed_s:.2f}s (budget {a.budget}s) stopped_by={res.stopped_by}\n"
        f"feasible={res.feasible} capped_out={res.capped_out} residual={res.residual:.4g}\n"
        f"diagnostics={ {k: round(v, 4) for k, v in res.diagnostics.items()} }\n"
        f"curves -> {out}"
    )


if __name__ == "__main__":
    main()
