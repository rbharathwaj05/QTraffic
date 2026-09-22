"""Pure geometry helpers. No I/O, no graph dependency beyond node coordinates."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (spec: road model, distance metric).

    d = 2R * asin(sqrt(sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2))), R = 6371000 m.
    Used only for heuristics / snapping; real distances come from the OSRM table.
    """
    return float(_haversine(np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)))


def _haversine(phi1, lam1, phi2, lam2):
    # [SPEC road model] haversine, inputs in radians, broadcastable
    a = np.sin((phi2 - phi1) / 2) ** 2
    a += np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorised pairwise haversine, shape (n, n), metres (spec: road model).

    Same formula as `haversine`, broadcast over all pairs. Numba/NumPy target.
    """
    phi, lam = np.radians(np.asarray(lats, float)), np.radians(np.asarray(lons, float))
    return _haversine(phi[:, None], lam[:, None], phi[None, :], lam[None, :])


def snap_to_node(node_lats: np.ndarray, node_lons: np.ndarray, lat: float, lon: float) -> int:
    """Return index of the nearest graph node to (lat, lon) (spec: customer/depot snapping).

    Nearest = argmin haversine over node arrays. Later phases may swap in a KD-tree.
    """
    d = _haversine(
        np.radians(np.asarray(node_lats, float)),
        np.radians(np.asarray(node_lons, float)),
        np.radians(lat),
        np.radians(lon),
    )
    return int(np.argmin(d))


def decode_polyline(encoded: str, precision: int = 5) -> list[tuple[float, float]]:
    """Decode a Google/OSRM encoded polyline into [(lat, lon), ...] (spec: OSRM geometry)."""
    scale = 10**precision
    out: list[tuple[float, float]] = []
    lat = lon = idx = 0
    while idx < len(encoded):
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += delta
            else:
                lon += delta
        out.append((lat / scale, lon / scale))
    return out
