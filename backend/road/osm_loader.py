"""Load, clean and cache the city road graph from OpenStreetMap via osmnx."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox

from backend.config import QTrafficConfig

log = logging.getLogger(__name__)

# Free-flow speed (km/h) per OSM `highway` tag, used when `maxspeed` is absent.
# Values follow osmnx's conventions for urban Indian roads; unlisted tags use FALLBACK_KPH.
# `osmnx.add_edge_speeds` prefers a numeric `maxspeed` tag and only falls back to this table.
HWY_SPEEDS_KPH: dict[str, float] = {
    "motorway": 80,
    "motorway_link": 60,
    "trunk": 60,
    "trunk_link": 50,
    "primary": 50,
    "primary_link": 40,
    "secondary": 40,
    "secondary_link": 35,
    "tertiary": 35,
    "tertiary_link": 30,
    "unclassified": 30,
    "residential": 25,
    "living_street": 15,
    "service": 15,
}
FALLBACK_KPH = 25.0


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_city_graph(
    place: str,
    cache_dir: Path,
    network_type: str = "drive",
    bbox: tuple[float, float, float, float] | None = None,
) -> nx.MultiDiGraph:
    """Return a cleaned osmnx MultiDiGraph for `place` (or `bbox`), cached as GraphML in
    `cache_dir` (spec: road model, data acquisition).

    Cache hit -> load from disk; miss -> download, clean, save. Cleaning:
      * keep only the largest strongly connected component (spec: continuity constraint,
        §7.3/§16 — an unreachable node is an unrepairable "no path exists" later);
      * every edge carries `length` (m), `speed_kph` (maxspeed tag else HWY_SPEEDS_KPH)
        and `travel_time` (s) via `osmnx.add_edge_speeds` / `add_edge_travel_times`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"bbox-{'_'.join(f'{v:.5f}' for v in bbox)}" if bbox else _slug(place)
    path = cache_dir / f"{key}-{network_type}.graphml"
    if path.exists():
        log.info("road graph cache hit: %s", path)
        return ox.load_graphml(path)

    if bbox:
        graph = ox.graph_from_bbox(bbox, network_type=network_type)
    else:
        graph = ox.graph_from_place(place, network_type=network_type)
    n0, e0 = graph.number_of_nodes(), graph.number_of_edges()
    graph = ox.truncate.largest_component(graph, strongly=True)
    log.info(
        "largest SCC kept: dropped %d nodes, %d edges (%d nodes, %d edges remain)",
        n0 - graph.number_of_nodes(),
        e0 - graph.number_of_edges(),
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    graph = ox.add_edge_speeds(graph, hwy_speeds=HWY_SPEEDS_KPH, fallback=FALLBACK_KPH)
    graph = ox.add_edge_travel_times(graph)
    ox.save_graphml(graph, path)
    return graph


def load_road_graph(config: QTrafficConfig) -> nx.MultiDiGraph:
    """`load_city_graph` driven by config (`city_query`, `city_bbox`, `city_cache_dir`)."""
    return load_city_graph(config.city_query, config.city_cache_dir, bbox=config.city_bbox)


def snap_points_to_nodes(graph: nx.MultiDiGraph, points: list[tuple[float, float]]) -> list[int]:
    """Nearest graph node for each (lat, lon), one vectorised KD-tree query
    (spec: customer/depot snapping)."""
    if not points:
        return []
    pts = np.asarray(points, dtype=float)
    ids = ox.distance.nearest_nodes(graph, X=pts[:, 1], Y=pts[:, 0])
    return [int(i) for i in np.atleast_1d(ids)]


def pick_depot_node(graph: nx.MultiDiGraph, near: tuple[float, float] | None = None) -> int:
    """Depot node: nearest to `near`, else nearest to the graph's node centroid."""
    if near is None:
        lat, lon = graph_to_arrays(graph)[:2]
        near = (float(lat.mean()), float(lon.mean()))
    return snap_points_to_nodes(graph, [near])[0]


def graph_to_geojson(graph: nx.MultiDiGraph) -> dict:
    """Road edges as a GeoJSON FeatureCollection of LineStrings, properties {u, v, key}
    only (spec: frontend map layer). Live per-edge speeds are a separate layer."""
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    return edges[["geometry"]].reset_index().__geo_interface__


def graph_to_arrays(graph: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten graph to (node_lat, node_lon, edge_uv[int32, (E,2)], edge_length[float64, (E,)])
    (spec: road model, array representation for numba kernels).

    Node index = position in `graph.nodes` order; the mapping is stable for the session.
    """
    index = {n: i for i, n in enumerate(graph.nodes)}
    lat = np.fromiter((d["y"] for _, d in graph.nodes(data=True)), float, len(index))
    lon = np.fromiter((d["x"] for _, d in graph.nodes(data=True)), float, len(index))
    uv = np.array([(index[u], index[v]) for u, v in graph.edges()], dtype=np.int32).reshape(-1, 2)
    length = np.array([d["length"] for _, _, d in graph.edges(data=True)], dtype=np.float64)
    return lat, lon, uv, length


def free_flow_travel_time(edge_length: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    """Per-edge free-flow time in seconds: t0 = length / (speed_kph / 3.6) (spec: road model)."""
    return np.asarray(edge_length, float) / (np.asarray(speed_kph, float) / 3.6)
