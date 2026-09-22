"""Bidirectional index: OD pair -> edges on its path, edge -> OD pairs using it.

THE shared structure of the system [SPEC v3 10.3]. Built exactly once per scenario
(`PathIndex.build`, N*(N+1) OSRM /route calls) and then reused, never rebuilt, by:

  (a) live travel-time inflation   -> `cost_matrix.update_factors` aggregates the
      per-edge congestion factor over `edges_on(i, j)`: T[i,j,t] = T_base[i,j] * inflate.
  (b) incremental cache invalidation -> `invalidate_edge(e)` is an O(1) lookup of the
      (i, j) pairs whose cached time is stale after edge e changes.
  (c) affected-vehicle identification (Phase 9) -> a vehicle is affected iff one of its
      planned legs (i, j) is in `pairs_using(event edges)`.

Do not build a second structure for any of these. Edge ids are positions in
`osm_loader.graph_to_arrays` edge order (see `osm_edge_index`).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from backend.road.osrm_client import OSRMClient


def osm_edge_index(graph: Any) -> dict[tuple[int, int], int]:
    """(u, v) OSM node ids -> edge id, aligned with `graph_to_arrays` edge order.
    Parallel edges (same u, v) collapse onto the last one; a traffic factor applies to
    both in practice, so this loses nothing the index needs."""
    return {(u, v): i for i, (u, v) in enumerate(graph.edges())}


class PathIndex:
    """Maps each (i, j) OD pair to the edge ids on its OSRM path and inverts that map
    so a traffic event on edge e can find every affected pair in O(1)
    (spec: path index / event-to-route impact)."""

    def __init__(self) -> None:
        self.n = 0  # number of points (depot + customers)
        # Forward map: (i, j) -> int64 array of edge ids in path order.
        self.pair_to_edges: dict[tuple[int, int], np.ndarray] = {}
        # Inverse map: edge id -> set of (i, j) pairs whose path uses that edge.
        self.edge_to_pairs: dict[int, set[tuple[int, int]]] = {}
        # Pairs whose OSRM path left the graph somewhere: their edge list is a subset of
        # the truth, so `invalidate_edge` must be conservative for unknown edges.
        self.partial: set[tuple[int, int]] = set()
        self._csr: tuple[np.ndarray, np.ndarray] | None = None  # lazy cache, see csr()

    @property
    def complete(self) -> bool:
        return not self.partial

    def build(
        self,
        points: list[tuple[float, float]],
        client: OSRMClient,
        osm_node_to_edge: dict[tuple[int, int], int],
        workers: int = 8,
    ) -> None:
        """Query `/route` for each ordered pair, translate consecutive OSM node ids into
        internal edge ids via `osm_node_to_edge`, populate both maps (spec: path index).

        OSRM returns every OSM node on the path; the osmnx graph is simplified, so the
        sequence is first filtered to graph nodes and consecutive survivors are looked
        up as (u, v). A missing (u, v) marks the pair `partial`.
        """
        self.n = len(points)
        # Set of OSM node ids that survived osmnx simplification (endpoints of some edge).
        graph_nodes = {u for u, _ in osm_node_to_edge} | {v for _, v in osm_node_to_edge}
        pairs = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]

        def one(pair):
            i, j = pair
            return pair, client.route([points[i], points[j]]).node_ids

        # N*(N-1) HTTP calls are I/O-bound; a thread pool overlaps them. Results are
        # consumed in submission order so the index is deterministic.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (i, j), node_ids in pool.map(one, pairs):
                self.add_path(i, j, node_ids, graph_nodes, osm_node_to_edge)
        for i in range(self.n):  # diagonal: staying put uses no edges
            self.pair_to_edges[(i, i)] = np.empty(0, dtype=np.int64)

    def add_path(
        self,
        i: int,
        i_to: int,
        node_ids: list[int],
        graph_nodes: set[int],
        osm_node_to_edge: dict[tuple[int, int], int],
    ) -> None:
        """Insert one (i, i_to) path given its OSM node sequence."""
        # Drop intermediate OSM nodes that osmnx simplified away; what remains should be
        # consecutive endpoints of graph edges.
        seq = [n for n in node_ids if n in graph_nodes]
        edges = []
        for u, v in zip(seq, seq[1:]):
            e = osm_node_to_edge.get((u, v))
            if e is None:
                # OSRM used a road the graph does not have (different extract / pruning):
                # remember the pair is incomplete so invalidation stays conservative.
                self.partial.add((i, i_to))
            else:
                edges.append(e)
        self.pair_to_edges[(i, i_to)] = np.array(edges, dtype=np.int64)
        for e in edges:  # maintain the inverse map
            self.edge_to_pairs.setdefault(e, set()).add((i, i_to))
        self._csr = None  # invalidate the cached CSR view

    def edges_on(self, i: int, j: int) -> np.ndarray:
        """Edge ids along path(i, j); empty array if i == j."""
        return self.pair_to_edges.get((i, j), np.empty(0, dtype=np.int64))

    def pairs_using(self, edge_ids: Iterable[int]) -> set[tuple[int, int]]:
        """Union of OD pairs whose path traverses any edge in `edge_ids`."""
        out: set[tuple[int, int]] = set()
        for e in edge_ids:
            out |= self.edge_to_pairs.get(e, set())
        return out

    def invalidate_edge(self, edge_id: int) -> set[tuple[int, int]]:
        """Pairs whose cached T is stale after `edge_id` changes: O(1) dict lookup.

        Fallback [SPEC 10.2]: if the edge is unknown to the index AND the index is
        partial, the dependency bound is not confident -> return every partial pair
        (broader rebuild) rather than risk a stale entry. Unknown edge on a complete
        index means no pair uses it, so the empty set is exact.
        """
        if edge_id in self.edge_to_pairs:
            return set(self.edge_to_pairs[edge_id])
        return set(self.partial)

    def csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Flat (ptr, edges) over pairs in row-major (i*n + j) order for vectorised
        aggregation; cached until the next `add_path`."""
        if self._csr is None:
            # Compressed-sparse-row layout: edges of pair k = flat[ptr[k]:ptr[k+1]],
            # k = i*n + j. Lets cost_matrix.inflate_all aggregate with one bincount.
            lists = [self.edges_on(i, j) for i in range(self.n) for j in range(self.n)]
            ptr = np.zeros(len(lists) + 1, dtype=np.int64)
            ptr[1:] = np.cumsum([len(x) for x in lists])
            flat = np.concatenate(lists) if lists else np.empty(0, dtype=np.int64)
            self._csr = (ptr, flat.astype(np.int64))
        return self._csr

    # -- persistence: later phases load, never recompute ----------------------------
    def save(self, path: Path) -> None:
        # Pickle the instance dict (plain dicts/sets/arrays), not the class, so the file
        # survives renames of this module.
        Path(path).write_bytes(pickle.dumps(self.__dict__))

    @classmethod
    def load(cls, path: Path) -> PathIndex:
        obj = cls()
        obj.__dict__.update(pickle.loads(Path(path).read_bytes()))
        return obj
