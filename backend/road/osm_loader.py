"""Load and cache the city road graph from OpenStreetMap via osmnx."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_city_graph(place: str, cache_dir: Path, network_type: str = "drive") -> Any:
    """Return an osmnx MultiDiGraph for `place`, cached as GraphML in `cache_dir`
    (spec: road model, data acquisition).

    Cache hit -> load from disk; miss -> `osmnx.graph_from_place` then save.
    Edges must carry `length` (m) and `speed_kph`; `osmnx.add_edge_speeds` and
    `add_edge_travel_times` are applied on first load.
    """
    raise NotImplementedError


def graph_to_arrays(graph: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten graph to (node_lat, node_lon, edge_uv[int32, (E,2)], edge_length[float64, (E,)])
    (spec: road model, array representation for numba kernels).

    Node index = position in `graph.nodes` order; the mapping is stable for the session.
    """
    raise NotImplementedError


def free_flow_travel_time(edge_length: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    """Per-edge free-flow time in seconds: t0 = length / (speed_kph / 3.6) (spec: road model)."""
    raise NotImplementedError
