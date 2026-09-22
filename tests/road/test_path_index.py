import numpy as np

from backend.road.path_index import PathIndex, osm_edge_index

from .fake_osrm import FakeOSRM


def synthetic_index() -> PathIndex:
    """3 points, hand-built: 0->1 uses edges [10, 11], 1->2 uses [11, 12], 2->0 uses [13];
    1->0 is partial (walked off-graph); others empty."""
    idx = PathIndex()
    idx.n = 3
    nodes = {100, 101, 102, 103, 104}
    n2e = {(100, 101): 10, (101, 102): 11, (102, 103): 12, (103, 100): 13}
    idx.add_path(0, 1, [100, 101, 102], nodes, n2e)
    idx.add_path(1, 2, [101, 102, 103], nodes, n2e)
    idx.add_path(2, 0, [103, 100], nodes, n2e)
    idx.add_path(1, 0, [102, 104], nodes, n2e)  # (102,104) not an edge -> partial
    idx.add_path(0, 2, [], nodes, n2e)
    idx.add_path(2, 1, [], nodes, n2e)
    for i in range(3):
        idx.pair_to_edges[(i, i)] = np.empty(0, dtype=np.int64)
    return idx


def test_invalidate_edge_returns_exact_subset():
    idx = synthetic_index()
    assert idx.invalidate_edge(11) == {(0, 1), (1, 2)}
    assert idx.invalidate_edge(13) == {(2, 0)}
    assert idx.pairs_using([10, 12]) == {(0, 1), (1, 2)}
    assert list(idx.edges_on(0, 1)) == [10, 11] and idx.edges_on(0, 0).size == 0


def test_invalidate_unknown_edge_falls_back_to_partial_pairs():
    idx = synthetic_index()
    assert not idx.complete and idx.partial == {(1, 0)}
    assert idx.invalidate_edge(999) == {(1, 0)}  # not confident -> broader
    idx.partial.clear()
    assert idx.invalidate_edge(999) == set()  # complete index -> exact empty


def test_csr_matches_pair_lists():
    idx = synthetic_index()
    ptr, flat = idx.csr()
    assert len(ptr) == 10 and ptr[-1] == len(flat) == 5
    assert list(flat[ptr[1] : ptr[2]]) == [10, 11]  # pair (0,1) = row-major index 1


def test_build_filters_simplified_nodes_and_roundtrips(tmp_path):
    pts = [(13.0, 80.0), (13.1, 80.1)]
    # OSRM emits intermediate node 55 that the simplified graph dropped
    client = FakeOSRM({(pts[0], pts[1]): [1, 55, 2], (pts[1], pts[0]): [2, 1]})
    idx = PathIndex()
    idx.build(pts, client, {(1, 2): 0, (2, 1): 1})
    assert client.route_calls == 2 and idx.complete
    assert list(idx.edges_on(0, 1)) == [0] and list(idx.edges_on(1, 0)) == [1]
    idx.save(tmp_path / "pi.pkl")
    back = PathIndex.load(tmp_path / "pi.pkl")
    assert back.n == 2 and back.edge_to_pairs == idx.edge_to_pairs


def test_osm_edge_index_aligns_with_graph_edges():
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    assert osm_edge_index(g) == {(1, 2): 0, (2, 3): 1}
