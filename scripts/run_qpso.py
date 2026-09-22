"""Run plain QPSO on one scenario under a wall-clock budget [SPEC 7.4-7.5, 11].

Usage: python scripts/run_qpso.py --scenario S1 --budget 5 [--seed 0] [--swarm 50]

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
from backend.optimization.encoding import dim
from backend.optimization.qpso import QPSO, Evaluator

BENCHMARK_DIR = Path("benchmarks")


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
    p.add_argument("--scenario", default="S1")
    p.add_argument("--budget", type=float, default=5.0, help="wall-clock seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swarm", type=int, default=None, help="M, default cfg.M")
    a = p.parse_args()

    cfg = QTrafficConfig(M=a.swarm) if a.swarm else QTrafficConfig()
    rng = np.random.default_rng(a.seed)
    ctx, prob = load_scenario(a.scenario, cfg, rng)
    opt = QPSO(cfg, Evaluator(ctx, prob, cfg), dim(ctx.n_customers), rng)
    res = opt.run(time_budget_s=a.budget)

    out = write_curves(res, BENCHMARK_DIR / f"qpso_{a.scenario}_seed{a.seed}.csv")
    print(
        f"scenario={a.scenario} N={ctx.n_customers} M_veh={ctx.n_vehicles} M={cfg.M} "
        f"seed={a.seed}\n"
        f"gbest_F={res.gbest_fitness:.6f} iterations={res.iterations} "
        f"wall_clock={res.elapsed_s:.2f}s (budget {a.budget}s) stopped_by={res.stopped_by}\n"
        f"feasible={res.feasible} capped_out={res.capped_out} residual={res.residual:.4g}\n"
        f"diagnostics={ {k: round(v, 4) for k, v in res.diagnostics.items()} }\n"
        f"curves -> {out}"
    )


if __name__ == "__main__":
    main()
