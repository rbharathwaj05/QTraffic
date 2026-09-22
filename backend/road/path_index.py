"""Bidirectional index: OD pair -> edges on its path, edge -> OD pairs using it."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from backend.road.osrm_client import OSRMClient


class PathIndex:
    """Maps each (i, j) OD pair to the edge ids on its OSRM path and inverts that map
    so a traffic event on edge e can find every affected pair in O(1)
    (spec: path index / event-to-route impact)."""

    def __init__(self) -> None:
        self.pair_to_edges: dict[tuple[int, int], np.ndarray] = {}
        self.edge_to_pairs: dict[int, set[tuple[int, int]]] = {}

    def build(
        self,
        points: list[tuple[float, float]],
        client: OSRMClient,
        osm_node_to_edge: dict[tuple[int, int], int],
    ) -> None:
        """Query `/route` for each ordered pair, translate consecutive OSM node ids into
        internal edge ids via `osm_node_to_edge`, populate both maps (spec: path index).
        """
        raise NotImplementedError

    def edges_on(self, i: int, j: int) -> np.ndarray:
        """Edge ids along path(i, j); empty array if i == j."""
        raise NotImplementedError

    def pairs_using(self, edge_ids: Iterable[int]) -> set[tuple[int, int]]:
        """Union of OD pairs whose path traverses any edge in `edge_ids`."""
        raise NotImplementedError
