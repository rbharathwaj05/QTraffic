"""Scenario generation: customers/vehicles snapped to the road graph, persisted as
data/scenarios/<name>/{customers.json, vehicles.json, depot.json}.

Scalability ladder [spec: scalability]:
    S1 10/5 correctness, S2 50/10 integration, S3 100/20 optimisation,
    S4 300/50 target, S5 300/100 final demo.

Placement [SPEC v2 doc 6]: uniform lat/lon inside the graph's node bbox -> one batched
`nearest_nodes` snap -> drop depot + duplicate nodes -> repeat until N unique customer
nodes. Customers therefore sit on the road network with density following the network.

Demand / service time: inclusive uniform over `config.demand_range` /
`config.service_time_range`.

Time windows: `tw_anytime_fraction` of customers get [0, shift_end_s) ("anytime");
the rest get a tight window of width ~ U(tw_width_range) placed at start ~ U(0,
shift_end_s - width), so windows spread across the whole shift and are never all-day.

Feasibility [SPEC rho_max]: sum(demand) <= rho_max * sum(capacity) or `generate` raises
ValueError instead of shipping an infeasible scenario; re-run with another seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from backend.config import QTrafficConfig
from backend.fleet.customer import Customer
from backend.fleet.vehicle import Vehicle
from backend.road.osm_loader import pick_depot_node, snap_points_to_nodes

SCENARIO_DIR = Path("data/scenarios")
SIZES: dict[str, tuple[int, int]] = {  # name -> (N customers, M_veh vehicles)
    "S1": (10, 5),
    "S2": (50, 10),
    "S3": (100, 20),
    "S4": (300, 50),
    "S5": (300, 100),
}


def check_feasible(customers: list[Customer], vehicles: list[Vehicle], rho_max: float) -> None:
    """Raise ValueError unless sum(demand) <= rho_max * sum(capacity) [SPEC rho_max]."""
    demand = sum(c.demand for c in customers)
    cap = rho_max * sum(v.capacity for v in vehicles)
    if demand > cap:
        raise ValueError(f"infeasible scenario: demand {demand} > rho_max*capacity {cap:.1f}")


def snap_customer_nodes(
    graph: nx.MultiDiGraph, n: int, rng: np.random.Generator, exclude: set[int]
) -> list[int]:
    """`n` distinct graph node ids: random bbox points -> batched snap [SPEC v2 doc 6]."""
    if graph.number_of_nodes() - len(exclude) < n:
        raise ValueError(f"graph has fewer than {n} usable nodes")
    lat = np.fromiter((d["y"] for _, d in graph.nodes(data=True)), float)
    lon = np.fromiter((d["x"] for _, d in graph.nodes(data=True)), float)
    out: dict[int, None] = {}  # ordered set
    # Rejection loop: sample 2x the shortfall each round (snapping collapses nearby
    # points onto the same node, so some duplicates are expected), keep the new ones.
    while len(out) < n:
        k = 2 * (n - len(out))
        pts = np.column_stack(
            [rng.uniform(lat.min(), lat.max(), k), rng.uniform(lon.min(), lon.max(), k)]
        )
        for node in snap_points_to_nodes(graph, [tuple(p) for p in pts]):
            if node not in exclude:  # skip depot (and duplicates via dict keys)
                out[node] = None
    return list(out)[:n]  # may overshoot on the last round; trim to exactly n


def generate(
    graph: nx.MultiDiGraph,
    n_customers: int,
    n_vehicles: int,
    config: QTrafficConfig,
    rng: np.random.Generator,
) -> tuple[list[Customer], list[Vehicle], dict]:
    """Build (customers, vehicles, depot) per the module docstring; raises if infeasible."""
    depot_node = pick_depot_node(graph, config.depot_latlon)
    dn = graph.nodes[depot_node]
    depot = {"node_id": int(depot_node), "lat": float(dn["y"]), "lon": float(dn["x"])}
    nodes = snap_customer_nodes(graph, n_customers, rng, {depot_node})

    H = config.shift_end_s
    # `integers` upper bound is exclusive, hence +1 for an inclusive range.
    demand = rng.integers(config.demand_range[0], config.demand_range[1] + 1, n_customers)
    service = rng.integers(
        config.service_time_range[0], config.service_time_range[1] + 1, n_customers
    )
    # Time windows: draw both variants for everyone, then select per customer with
    # np.where so the random stream is consumed identically regardless of the mask.
    anytime = rng.random(n_customers) < config.tw_anytime_fraction
    width = rng.uniform(config.tw_width_range[0], config.tw_width_range[1], n_customers)
    start = rng.random(n_customers) * (H - width)  # keeps start + width <= H
    tw_start = np.where(anytime, 0.0, start)
    tw_end = np.where(anytime, H, start + width)

    customers = [
        Customer(
            customer_id=f"c{i:03d}",
            lat=float(graph.nodes[nd]["y"]),
            lon=float(graph.nodes[nd]["x"]),
            node_id=int(nd),
            demand=int(demand[i]),
            service_time=int(service[i]),
            time_window_start=round(float(tw_start[i]), 1),
            time_window_end=round(float(tw_end[i]), 1),
        )
        for i, nd in enumerate(nodes)
    ]
    # Homogeneous fleet: same capacity, all start at the depot, same shift end.
    vehicles = [
        Vehicle(f"v{k:03d}", config.vehicle_capacity, int(depot_node), H) for k in range(n_vehicles)
    ]
    check_feasible(customers, vehicles, config.rho_max)  # refuse to ship an impossible instance
    return customers, vehicles, depot


def save(
    name: str,
    customers: list[Customer],
    vehicles: list[Vehicle],
    depot: dict,
    out_dir: Path = SCENARIO_DIR,
) -> Path:
    """Write the three JSON files for scenario `name`; returns the directory."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    # indent=1 keeps the committed files diff-friendly (one field per line).
    (d / "customers.json").write_text(json.dumps([c.to_dict() for c in customers], indent=1))
    (d / "vehicles.json").write_text(json.dumps([v.to_dict() for v in vehicles], indent=1))
    (d / "depot.json").write_text(json.dumps(depot))
    return d


def load(name: str, out_dir: Path = SCENARIO_DIR) -> tuple[list[Customer], list[Vehicle], dict]:
    """Inverse of `save`: (customers, vehicles, depot) from data/scenarios/<name>/."""
    d = Path(out_dir) / name
    customers = [Customer.from_dict(x) for x in json.loads((d / "customers.json").read_text())]
    vehicles = [Vehicle.from_dict(x) for x in json.loads((d / "vehicles.json").read_text())]
    return customers, vehicles, json.loads((d / "depot.json").read_text())
