"""Fake OSRM over a 4-node ring graph: 10 -> 20 -> 30 -> 40 -> 10. Raw OSRM node 25 sits
between 20 and 30 (simplified away in the osmnx graph) to exercise the filtering."""

import json

import httpx
import networkx as nx
import numpy as np
import pytest

from backend.road.osrm_client import OSRMClient
from backend.road.path_index import PathIndex, edge_free_flow_times, osm_node_to_edge_map

RING = [10, 20, 30, 40]
POINTS = [(13.00, 80.00), (13.02, 80.02), (13.03, 80.03)]  # -> nodes 10, 30, 40
POINT_NODE = {f"{lon:.6f},{lat:.6f}": n for (lat, lon), n in zip(POINTS, [10, 30, 40])}


def ring_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n in RING:
        g.add_node(n, x=0.0, y=0.0)
    for i, (u, v) in enumerate(zip(RING, RING[1:] + RING[:1])):
        g.add_edge(u, v, length=100.0 * (i + 1), speed_kph=36.0, travel_time=10.0 * (i + 1))
    return g


def ring_walk(a: int, b: int) -> list[int]:
    i = RING.index(a)
    out = [a]
    while out[-1] != b:
        i = (i + 1) % len(RING)
        out.append(RING[i])
    return out


def fake_osrm(req: httpx.Request) -> httpx.Response:
    coords = req.url.path.split("/")[-1].split(";")
    nodes = [POINT_NODE[c] for c in coords]
    if "/route/" in req.url.path:
        walk = ring_walk(nodes[0], nodes[1])
        raw = [n for n in walk for n in ([n, 25] if n == 20 else [n])]  # inject raw node 25
        body = {
            "code": "Ok",
            "routes": [
                {
                    "duration": 10.0 * (len(walk) - 1),
                    "distance": 100.0 * (len(walk) - 1),
                    "geometry": {"coordinates": []},
                    "legs": [{"annotation": {"nodes": raw}}],
                }
            ],
        }
    else:  # table: duration = 10 s per hop, distance = 100 m per hop
        hops = [[len(ring_walk(a, b)) - 1 for b in nodes] for a in nodes]
        body = {
            "code": "Ok",
            "durations": [[10.0 * h for h in row] for row in hops],
            "distances": [[100.0 * h for h in row] for row in hops],
        }
    return httpx.Response(200, text=json.dumps(body))


@pytest.fixture
def client():
    return OSRMClient("http://osrm", transport=httpx.MockTransport(fake_osrm))


@pytest.fixture
def index(client):
    g = ring_graph()
    idx = PathIndex()
    idx.build(POINTS, client, osm_node_to_edge_map(g), edge_t0=edge_free_flow_times(g))
    return idx


def test_edge_map_and_t0_follow_graph_edge_order():
    g = ring_graph()
    assert osm_node_to_edge_map(g) == {(10, 20): 0, (20, 30): 1, (30, 40): 2, (40, 10): 3}
    np.testing.assert_array_equal(edge_free_flow_times(g), [10.0, 20.0, 30.0, 40.0])


def test_build_maps_pairs_to_edges_skipping_raw_nodes(index):
    # 10 -> 30 crosses edges 0, 1 (raw node 25 filtered out, not a hole)
    np.testing.assert_array_equal(index.edges_on(0, 1), [0, 1])
    np.testing.assert_array_equal(index.edges_on(1, 2), [2])
    np.testing.assert_array_equal(index.edges_on(2, 0), [3])
    np.testing.assert_array_equal(index.edges_on(1, 0), [2, 3])
    assert index.edges_on(1, 1).size == 0
    assert len(index.pair_to_edges) == 6


def test_pairs_using_inverts_map(index):
    assert index.pairs_using([0]) == {(0, 1), (0, 2), (2, 1)}
    assert index.pairs_using([2, 3]) == {(1, 2), (1, 0), (2, 0), (2, 1), (0, 2)}
    assert index.pairs_using([99]) == set()
