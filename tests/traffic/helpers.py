"""Tiny hand-built road layer for the Phase 9 tests: 4 nodes, 5 edges, explicit paths.

Small enough that every affected-vehicle answer can be read off by hand, which is the
point -- these tests pin behaviour the spec states, not whatever the code happens to do.
"""

from __future__ import annotations

import numpy as np

from backend.road.cost_matrix import CostMatrix, TrafficVersion
from backend.road.path_index import PathIndex

N_EDGES = 5
# path(i, j) -> edge ids. Symmetric, and deliberately NOT geographic: pair (0, 3) runs
# through edge 0 even though node 3 is far from it, and pair (0, 2) avoids edge 0 even
# though node 2 sits right next to it [SPEC 10.3 over/under-counting].
PATHS = {
    (0, 1): [0],
    (1, 0): [0],
    (0, 2): [2],
    (2, 0): [2],
    (0, 3): [0, 3],
    (3, 0): [0, 3],
    (1, 2): [1],
    (2, 1): [1],
    (1, 3): [1, 3],
    (3, 1): [1, 3],
    (2, 3): [4],
    (3, 2): [4],
}
# node coordinates for the proximity counter-example: 2 is next to edge 0, 3 is far away
COORDS = np.array([[0.0, 0.0], [0.0, 1.0], [0.1, 0.1], [50.0, 50.0]])


def path_index() -> PathIndex:
    idx = PathIndex()
    idx.n = 4
    for i in range(4):
        for j in range(4):
            e = np.array(PATHS.get((i, j), []), dtype=np.int64)
            idx.pair_to_edges[(i, j)] = e
            for k in e:
                idx.edge_to_pairs.setdefault(int(k), set()).add((i, j))
    return idx


def cost_matrix(scenario: str = "test") -> CostMatrix:
    """Uniform 100 s / 1000 m between distinct nodes, zero diagonal, factor 1."""
    dur = np.full((4, 4), 100.0)
    np.fill_diagonal(dur, 0.0)
    dist = np.full((4, 4), 1000.0)
    np.fill_diagonal(dist, 0.0)
    return CostMatrix(dur, dist, np.ones((4, 4)), np.ones(N_EDGES), TrafficVersion(scenario))
