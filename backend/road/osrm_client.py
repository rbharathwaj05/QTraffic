"""Thin HTTP client for a local OSRM instance (see docker-compose `osrm` service)."""

from __future__ import annotations

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

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s

    def route(self, coords: list[tuple[float, float]], annotations: bool = True) -> RouteResult:
        """Single `/route/v1/driving` call over an ordered coordinate list.

        Returns total duration/distance plus geometry and node annotations, which
        `path_index` uses to map routes onto graph edges (spec: path index).
        """
        raise NotImplementedError

    def table(
        self,
        sources: list[tuple[float, float]],
        destinations: list[tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`/table/v1/driving` -> (durations_s, distances_m), each shape (S, D)
        (spec: cost matrix construction). `destinations=None` means sources x sources.
        """
        raise NotImplementedError
