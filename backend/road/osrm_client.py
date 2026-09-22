"""Thin HTTP client for a local OSRM instance (see docker-compose `osrm` service)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np


@dataclass(frozen=True)
class RouteResult:
    duration_s: float
    distance_m: float
    geometry: list[tuple[float, float]]  # [(lat, lon), ...]
    node_ids: list[int]  # OSM node ids along the path


def _coords(points: list[tuple[float, float]]) -> str:
    """(lat, lon) list -> OSRM `lon,lat;lon,lat` path segment."""
    return ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)


class OSRMClient:
    """Wraps OSRM `/route` and `/table` endpoints (spec: road model, cost source).

    `transport` is forwarded to `httpx.Client`; tests pass `httpx.MockTransport`.
    """

    def __init__(
        self, base_url: str, timeout_s: float = 5.0, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def _get(self, path: str, params: dict) -> dict:
        resp = self._http.get(path, params=params)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != "Ok":
            raise RuntimeError(f"OSRM {path}: {body.get('code')} {body.get('message', '')}")
        return body

    def route(self, coords: list[tuple[float, float]], annotations: bool = True) -> RouteResult:
        """Single `/route/v1/driving` call over an ordered coordinate list.

        Returns total duration/distance plus geometry and node annotations, which
        `path_index` uses to map routes onto graph edges (spec: path index).
        """
        params = {"overview": "full", "geometries": "geojson"}
        if annotations:
            params["annotations"] = "nodes"
        body = self._get(f"/route/v1/driving/{_coords(coords)}", params)
        r = body["routes"][0]
        geometry = [(lat, lon) for lon, lat in r["geometry"]["coordinates"]]
        node_ids: list[int] = []
        for leg in r["legs"] if annotations else []:
            for n in leg["annotation"]["nodes"]:
                if not node_ids or node_ids[-1] != n:  # legs share their junction node
                    node_ids.append(int(n))
        return RouteResult(float(r["duration"]), float(r["distance"]), geometry, node_ids)

    def table(
        self,
        sources: list[tuple[float, float]],
        destinations: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`/table/v1/driving` -> (durations_s, distances_m), each shape (S, D)
        (spec: cost matrix construction). `destinations=None` means sources x sources.
        Unroutable pairs (OSRM `null`) come back as NaN.
        """
        params: dict = {"annotations": "duration,distance"}
        pts = list(sources)
        if destinations is not None:
            params["sources"] = ";".join(map(str, range(len(sources))))
            params["destinations"] = ";".join(
                map(str, range(len(sources), len(sources) + len(destinations)))
            )
            pts += list(destinations)
        body = self._get(f"/table/v1/driving/{_coords(pts)}", params)
        dur = np.array(body["durations"], dtype=np.float64)  # None -> nan under float dtype
        dist = np.array(body["distances"], dtype=np.float64)
        return dur, dist
