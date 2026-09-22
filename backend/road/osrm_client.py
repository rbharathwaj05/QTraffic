"""Thin HTTP client for a local OSRM instance (docker-compose `osrm` service).

OSRM data build (one-time, documented here because it is the only non-Python step):

  1. Put an OSM extract covering `QTrafficConfig.city_query` at `data/city/city.osm.pbf`
     (Geofabrik regional file; optionally clip with
     `osmium extract -b W,S,E,N in.osm.pbf -o data/city/city.osm.pbf`).
     A standalone .pbf is used rather than re-exporting the osmnx graph: osmnx's XML
     export is lossy for simplified graphs, and node ids are OSM ids in both, so
     `path_index` can map OSRM node annotations onto graph edges directly.
  2. `make osrm-build`  (osrm-extract -> osrm-partition -> osrm-customize, MLD).
  3. `docker compose up osrm`.

Usage boundary [SPEC 10.1-10.2, defect #1]: `table` runs once per scenario inside
`cost_matrix.build_cost_matrices`; `route` runs once per OD pair inside
`PathIndex.build`, and at deployment/display time. Neither is ever called from the
optimiser, fitness, or repair.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteResult:
    duration_s: float
    distance_m: float
    geometry: list[tuple[float, float]]  # [(lat, lon), ...]
    node_ids: list[int]  # OSM node ids along the path


class OSRMClient:
    """Wraps OSRM `/route` and `/table` endpoints (spec: road model, cost source)."""

    def __init__(self, base_url: str, timeout_s: float = 30.0, max_table_size: int = 100) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # osrm-routed --max-table-size: max coordinates per /table request.
        self.max_table_size = max_table_size

    # -- transport -------------------------------------------------------------------
    def _get(self, service: str, coords: list[tuple[float, float]], params: dict) -> dict:
        """GET `/<service>/v1/driving/<lon,lat;...>?params`, raise on non-Ok code."""
        path = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
        url = f"{self.base_url}/{service}/v1/driving/{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
            body = json.load(resp)
        if body.get("code") != "Ok":
            raise RuntimeError(f"OSRM {service}: {body.get('code')} {body.get('message', '')}")
        return body

    # -- endpoints -------------------------------------------------------------------
    def route(self, coords: list[tuple[float, float]], annotations: bool = True) -> RouteResult:
        """Single `/route/v1/driving` call over an ordered coordinate list.

        Returns total duration/distance plus geometry and node annotations, which
        `path_index` uses to map routes onto graph edges (spec: path index).
        """
        params = {
            "overview": "full",
            "geometries": "geojson",
            "annotations": "nodes" if annotations else "false",
        }
        r = self._get("route", coords, params)["routes"][0]
        nodes = [n for leg in r["legs"] for n in leg["annotation"]["nodes"]] if annotations else []
        return RouteResult(
            duration_s=float(r["duration"]),
            distance_m=float(r["distance"]),
            geometry=[(lat, lon) for lon, lat in r["geometry"]["coordinates"]],
            node_ids=[int(n) for n in nodes],
        )

    def table(
        self,
        sources: list[tuple[float, float]],
        destinations: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`/table/v1/driving` -> (durations_s, distances_m), each shape (S, D)
        (spec: cost matrix construction). `destinations=None` means sources x sources.

        Requests are tiled into (row block x column block) so each call carries at most
        `max_table_size` coordinates: ceil(S/b)*ceil(D/b) calls with b = max/2, never
        one call per pair [SPEC 10.2].
        """
        dests = sources if destinations is None else destinations
        S, D = len(sources), len(dests)
        dur = np.full((S, D), np.nan)
        dist = np.full((S, D), np.nan)
        b = max(1, self.max_table_size // 2)
        for r0 in range(0, S, b):
            src = sources[r0 : r0 + b]
            for c0 in range(0, D, b):
                dst = dests[c0 : c0 + b]
                params = {
                    "annotations": "duration,distance",
                    "sources": ";".join(map(str, range(len(src)))),
                    "destinations": ";".join(map(str, range(len(src), len(src) + len(dst)))),
                }
                body = self._get("table", src + dst, params)
                dur[r0 : r0 + len(src), c0 : c0 + len(dst)] = _block(body["durations"])
                dist[r0 : r0 + len(src), c0 : c0 + len(dst)] = _block(body["distances"])
        return dur, dist


def _block(rows: list[list]) -> np.ndarray:
    # OSRM emits null for unroutable pairs; keep as nan so the caller can decide.
    return np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
