import httpx
import numpy as np
import pytest

from backend.road.cost_matrix import build_cost_matrix, effective_duration, update_factors
from backend.road.osrm_client import OSRMClient
from backend.road.path_index import PathIndex, edge_free_flow_times, osm_node_to_edge_map
from tests.road.test_path_index import POINTS, fake_osrm, ring_graph


@pytest.fixture
def client():
    return OSRMClient("http://osrm", transport=httpx.MockTransport(fake_osrm))


@pytest.fixture
def index(client):
    g = ring_graph()
    idx = PathIndex()
    idx.build(POINTS, client, osm_node_to_edge_map(g), edge_t0=edge_free_flow_times(g))
    return idx


def test_build_cost_matrix(client):
    m = build_cost_matrix(POINTS, client)
    np.testing.assert_array_equal(m.duration_s, [[0, 20, 30], [20, 0, 10], [10, 30, 0]])
    np.testing.assert_array_equal(m.distance_m, m.duration_s * 10)
    np.testing.assert_array_equal(m.factor, np.ones((3, 3)))


def test_update_factors_is_t0_weighted_mean(client, index):
    m = build_cost_matrix(POINTS, client)
    f = np.array([1.0, 3.0, 1.0, 1.0])  # congest edge 1 (20 -> 30, t0 = 20 s)
    changed = update_factors(m, f, index)
    # pair (0, 1) uses edges 0, 1 with t0 10, 20: (10*1 + 20*3) / 30 = 7/3
    assert m.factor[0, 1] == pytest.approx(7 / 3)
    # pair (0, 2) uses edges 0, 1, 2 with t0 10, 20, 30: (10 + 60 + 30) / 60 = 5/3
    assert m.factor[0, 2] == pytest.approx(5 / 3)
    assert m.factor[1, 2] == 1.0 and m.factor[1, 1] == 1.0
    assert {tuple(p) for p in changed} == {(0, 1), (0, 2), (2, 1)}
    assert effective_duration(m)[0, 1] == pytest.approx(20 * 7 / 3)
    # no change -> nothing reported
    assert update_factors(m, f, index).shape == (0, 2)


def test_unroutable_pair_raises():
    def handler(req):
        return httpx.Response(
            200,
            json={"code": "Ok", "durations": [[0, None], [1, 0]], "distances": [[0, 1], [1, 0]]},
        )

    c = OSRMClient("http://osrm", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="unroutable"):
        build_cost_matrix([(13.0, 80.0), (13.1, 80.1)], c)
