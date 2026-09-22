"""One-time scenario precompute: D, T_base, PathIndex -> data/scenarios/<id>/.

    python -m backend.road.build_scenario --id demo50 --n 50 [--seed 0]

Requires `docker compose up osrm` (see `osrm_client` docstring for the data build).
Later phases call `load_scenario`, never this.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from backend.config import QTrafficConfig
from backend.road.cost_matrix import CostMatrix, TrafficVersion, build_cost_matrix
from backend.road.osm_loader import load_road_graph, pick_depot_node
from backend.road.osrm_client import OSRMClient
from backend.road.path_index import PathIndex, osm_edge_index

log = logging.getLogger(__name__)
SCENARIO_DIR = Path("data/scenarios")


def build_scenario(
    scenario_id: str,
    n_customers: int,
    config: QTrafficConfig,
    rng: np.random.Generator,
    out_dir: Path = SCENARIO_DIR,
) -> tuple[CostMatrix, PathIndex, list[tuple[float, float]]]:
    """Depot (index 0) + `n_customers` random graph nodes -> matrices + path index,
    persisted under `out_dir/scenario_id`. Returns (matrix, index, points)."""
    t = time.perf_counter()
    graph = load_road_graph(config)
    depot = pick_depot_node(graph, config.depot_latlon)
    pool = [n for n in graph.nodes if n != depot]
    nodes = [depot] + [int(x) for x in rng.choice(pool, n_customers, replace=False)]
    points = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in nodes]
    edge_t0 = np.array([d["travel_time"] for _, _, d in graph.edges(data=True)], float)

    client = OSRMClient(config.osrm_url)
    matrix = build_cost_matrix(points, client, edge_t0, TrafficVersion(scenario_id))
    t_table = time.perf_counter() - t
    index = PathIndex()
    index.build(points, client, osm_edge_index(graph))
    t_total = time.perf_counter() - t

    d = Path(out_dir) / scenario_id
    d.mkdir(parents=True, exist_ok=True)
    matrix.save(d / "matrices.npz")
    index.save(d / "path_index.pkl")
    np.save(d / "points.npy", np.array(points))
    log.info(
        "scenario %s: N=%d table %.1fs, total %.1fs, %d partial pairs -> %s",
        scenario_id,
        n_customers,
        t_table,
        t_total,
        len(index.partial),
        d,
    )
    return matrix, index, points


def load_scenario(
    scenario_id: str, out_dir: Path = SCENARIO_DIR, redis=None
) -> tuple[CostMatrix, PathIndex, np.ndarray]:
    """Load persisted artifacts; no OSRM, no graph."""
    d = Path(out_dir) / scenario_id
    matrix = CostMatrix.load(d / "matrices.npz", TrafficVersion(scenario_id, redis))
    return matrix, PathIndex.load(d / "path_index.pkl"), np.load(d / "points.npy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_scenario(a.id, a.n, QTrafficConfig(), np.random.default_rng(a.seed))
