"""Deterministic stand-in for OSRMClient: haversine costs, straight-line 'routes'."""

import numpy as np

from backend.road.geometry import haversine_matrix
from backend.road.osrm_client import RouteResult


class FakeOSRM:
    """Duration = distance / 10 m/s. `node_ids` for route(i -> j) come from `paths`,
    a dict {(lat,lon) pair tuple: [osm node ids]}; default is [] (no graph nodes)."""

    def __init__(self, paths=None):
        self.paths = paths or {}
        self.table_calls = 0
        self.route_calls = 0

    def table(self, sources, destinations=None):
        self.table_calls += 1
        pts = np.array(sources)
        dist = haversine_matrix(pts[:, 0], pts[:, 1]) + 1.0
        np.fill_diagonal(dist, 0.0)
        return dist / 10.0, dist

    def route(self, coords, annotations=True):
        self.route_calls += 1
        nodes = self.paths.get(tuple(coords), [])
        return RouteResult(1.0, 10.0, list(coords), nodes)
