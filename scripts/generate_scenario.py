"""Generate the S1..S5 scenario ladder (or one size) onto data/scenarios/<name>/.

Usage: python scripts/generate_scenario.py [--only S1] [--seed 0] [--bbox W S E N]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from backend.config import QTrafficConfig
from backend.fleet import scenario
from backend.road.osm_loader import load_road_graph


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(scenario.SIZES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = QTrafficConfig(city_bbox=tuple(a.bbox)) if a.bbox else QTrafficConfig()
    graph = load_road_graph(cfg)
    names = [a.only] if a.only else list(scenario.SIZES)
    for i, name in enumerate(names):
        n, m = scenario.SIZES[name]
        customers, vehicles, depot = scenario.generate(
            graph, n, m, cfg, np.random.default_rng(a.seed + i)
        )
        out = scenario.save(name, customers, vehicles, depot)
        demand = sum(c.demand for c in customers)
        cap = sum(v.capacity for v in vehicles)
        print(f"{name}: {n} customers / {m} vehicles, demand {demand}/{cap} -> {out}")


if __name__ == "__main__":
    main()
