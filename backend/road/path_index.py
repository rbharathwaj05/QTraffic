"""Bidirectional index: OD pair -> edges on its path, edge -> OD pairs using it."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from backend.road.osrm_client import OSRMClient

_EMPTY = np.empty(0, dtype=np.int64)


def osm_node_to_edge_map(graph: Any) -> dict[tuple[int, int], int]:
    """(u, v) OSM node ids -> internal edge id, in the same `graph.edges()` order that
    `osm_loader.graph_to_arrays` uses. Parallel edges (same u, v) keep the first id."""
    out: dict[tuple[int, int], int] = {}
    for i, (u, v) in enumerate(graph.edges()):
        out.setdefault((int(u), int(v)), i)
    return out


def edge_free_flow_times(graph: Any) -> np.ndarray:
    """Per-edge free-flow time t0_e (s) from the loader's `travel_time` attribute, in
    `graph.edges()` order (spec: road model). This is the Phase-2 answer to "does the cost
    layer need speed_kph?": no — t0 is already stored on every edge by `load_city_graph`."""
    return np.array([d["travel_time"] for _, _, d in graph.edges(data=True)], dtype=np.float64)


class PathIndex:
    """Maps each (i, j) OD pair to the edge ids on its OSRM path and inverts that map
    so a traffic event on edge e can find every affected pair in O(1)
    (spec: path index / event-to-route impact)."""

    def __init__(self) -> None:
        self.pair_to_edges: dict[tuple[int, int], np.ndarray] = {}
        self.edge_to_pairs: dict[int, set[tuple[int, int]]] = {}
        self.edge_t0: np.ndarray | None = None  # t0_e weights for cost_matrix.update_factors

    def build(
        self,
        points: list[tuple[float, float]],
        client: OSRMClient,
        osm_node_to_edge: dict[tuple[int, int], int],
        edge_t0: np.ndarray | None = None,
    ) -> None:
        """Query `/route` for each ordered pair, translate consecutive OSM node ids into
        internal edge ids via `osm_node_to_edge`, populate both maps (spec: path index).

        OSRM routes over the raw OSM graph while the osmnx graph is simplified, so the
        route's node list is first filtered to nodes the graph knows; consecutive survivors
        are then looked up as (u, v). Pairs with no known edge are left out.
        `edge_t0` (see `edge_free_flow_times`) is kept for `cost_matrix.update_factors`.
        """
        self.edge_t0 = None if edge_t0 is None else np.asarray(edge_t0, dtype=np.float64)
        known = {u for u, _ in osm_node_to_edge} | {v for _, v in osm_node_to_edge}
        self.pair_to_edges.clear()
        self.edge_to_pairs.clear()
        # ponytail: n(n-1) /route calls; switch to /table + /match batching if n grows past ~100
        for i, a in enumerate(points):
            for j, b in enumerate(points):
                if i == j:
                    continue
                nodes = [n for n in client.route([a, b]).node_ids if n in known]
                ids = [
                    osm_node_to_edge[(u, v)]
                    for u, v in zip(nodes, nodes[1:])
                    if (u, v) in osm_node_to_edge
                ]
                edges = np.array(ids, dtype=np.int64)
                self.pair_to_edges[(i, j)] = edges
                for e in ids:
                    self.edge_to_pairs.setdefault(e, set()).add((i, j))

    def edges_on(self, i: int, j: int) -> np.ndarray:
        """Edge ids along path(i, j); empty array if i == j."""
        return self.pair_to_edges.get((i, j), _EMPTY)

    def pairs_using(self, edge_ids: Iterable[int]) -> set[tuple[int, int]]:
        """Union of OD pairs whose path traverses any edge in `edge_ids`."""
        out: set[tuple[int, int]] = set()
        for e in edge_ids:
            out |= self.edge_to_pairs.get(int(e), set())
        return out
