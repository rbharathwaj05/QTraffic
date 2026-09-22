"""Pure geometry helpers. No I/O, no graph dependency beyond node coordinates."""

from __future__ import annotations

import numpy as np


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (spec: road model, distance metric).

    d = 2R * asin(sqrt(sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2))), R = 6371000 m.
    Used only for heuristics / snapping; real distances come from the OSRM table.
    """
    raise NotImplementedError


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorised pairwise haversine, shape (n, n), metres (spec: road model).

    Same formula as `haversine`, broadcast over all pairs. Numba/NumPy target.
    """
    raise NotImplementedError


def snap_to_node(node_lats: np.ndarray, node_lons: np.ndarray, lat: float, lon: float) -> int:
    """Return index of the nearest graph node to (lat, lon) (spec: customer/depot snapping).

    Nearest = argmin haversine over node arrays. Later phases may swap in a KD-tree.
    """
    raise NotImplementedError


def decode_polyline(encoded: str, precision: int = 5) -> list[tuple[float, float]]:
    """Decode a Google/OSRM encoded polyline into [(lat, lon), ...] (spec: OSRM geometry)."""
    raise NotImplementedError
