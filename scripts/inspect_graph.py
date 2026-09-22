"""Load the configured city graph and print node/edge counts plus the depot node.

Usage: python scripts/inspect_graph.py [--bbox W S E N] [--place "City, Country"]
"""

from __future__ import annotations

import argparse
import logging

from backend.config import QTrafficConfig
from backend.road.osm_loader import load_road_graph, pick_depot_node


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--place")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # CLI flags override config defaults; --bbox wins over --place inside load_city_graph.
    kw = {}
    if a.place:
        kw["city_query"] = a.place
    if a.bbox:
        kw["city_bbox"] = tuple(a.bbox)
    cfg = QTrafficConfig(**kw)
    graph = load_road_graph(cfg)  # first run downloads + caches; later runs hit GraphML
    depot = pick_depot_node(graph, cfg.depot_latlon)
    d = graph.nodes[depot]  # osmnx stores lat as "y", lon as "x"
    print(f"city: {cfg.city_bbox or cfg.city_query}")
    print(f"nodes: {graph.number_of_nodes()}  edges: {graph.number_of_edges()}")
    print(f"depot node: {depot}  (lat={d['y']:.6f}, lon={d['x']:.6f})")


if __name__ == "__main__":
    main()
