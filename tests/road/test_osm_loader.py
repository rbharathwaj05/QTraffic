"""Phase 1: real OSM download of a ~1.5 km T. Nagar bbox (skips offline). Checks the
cleaned graph is strongly connected, every edge has length/speed/travel_time,
batched snapping + depot pick work, GraphML cache round-trips, GeoJSON is LineStrings."""

import networkx as nx
import pytest

from backend.config import QTrafficConfig
from backend.road import osm_loader as mod

# ~1.5 km x 1.5 km around T. Nagar, Chennai: small enough to download in a few seconds.
BBOX = (80.225, 13.035, 80.245, 13.050)


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    cfg = QTrafficConfig(city_bbox=BBOX, city_cache_dir=tmp_path_factory.mktemp("city"))
    try:
        g = mod.load_road_graph(cfg)
    except Exception as exc:  # network / Overpass outage -> skip, not fail
        pytest.skip(f"OSM download unavailable: {exc!r}")
    print(f"\nbbox graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    return g


def test_strongly_connected_and_no_dead_ends(graph):
    assert nx.is_strongly_connected(graph)
    assert all(graph.out_degree(n) >= 1 for n in graph.nodes)


def test_edges_have_length_and_speed(graph):
    for _, _, d in graph.edges(data=True):
        assert d["length"] > 0
        assert d["speed_kph"] > 0
        assert d["travel_time"] > 0


def test_snap_and_depot(graph):
    pts = [(13.040, 80.230), (13.045, 80.240), (13.038, 80.236)]
    ids = mod.snap_points_to_nodes(graph, pts)
    assert len(ids) == len(pts)
    assert all(i in graph.nodes for i in ids)
    assert mod.pick_depot_node(graph) in graph.nodes
    assert mod.pick_depot_node(graph, pts[0]) == ids[0]
    assert mod.snap_points_to_nodes(graph, []) == []


def test_cache_roundtrip(graph, tmp_path):
    cfg = QTrafficConfig(city_bbox=BBOX, city_cache_dir=tmp_path)
    g1 = mod.load_road_graph(cfg)
    assert list(tmp_path.glob("*.graphml"))
    g2 = mod.load_road_graph(cfg)  # cache hit
    assert g1.number_of_edges() == g2.number_of_edges()
    assert nx.is_strongly_connected(g2)


def test_arrays_and_geojson(graph):
    lat, lon, uv, length = mod.graph_to_arrays(graph)
    assert lat.shape == lon.shape == (graph.number_of_nodes(),)
    assert uv.shape == (graph.number_of_edges(), 2) and length.shape == (uv.shape[0],)
    assert uv.max() < len(lat)
    gj = mod.graph_to_geojson(graph)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == graph.number_of_edges()
    assert gj["features"][0]["geometry"]["type"] == "LineString"


def test_free_flow_travel_time():
    import numpy as np

    t = mod.free_flow_travel_time(np.array([1000.0]), np.array([36.0]))
    assert np.allclose(t, [100.0])
