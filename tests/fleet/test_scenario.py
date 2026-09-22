"""Phase 3: generate() on the test bbox for S1/S2 — every customer on a graph node,
distinct, not the depot, feasible under rho_max, windows inside the shift and not all
'anytime'; JSON save/load round-trip; infeasible capacity raises. Skips offline."""

import numpy as np
import pytest

from backend.config import QTrafficConfig
from backend.fleet import scenario as mod
from backend.road.osm_loader import load_road_graph
from tests.road.test_osm_loader import BBOX


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    cfg = QTrafficConfig(city_bbox=BBOX, city_cache_dir=tmp_path_factory.mktemp("city"))
    try:
        return load_road_graph(cfg)
    except Exception as exc:
        pytest.skip(f"OSM download unavailable: {exc!r}")


@pytest.mark.parametrize("name", ["S1", "S2"])  # S3+ need more nodes than the test bbox has
def test_generate_snapped_feasible_windowed(graph, name, tmp_path):
    cfg = QTrafficConfig(city_bbox=BBOX)
    n, m = mod.SIZES[name]
    customers, vehicles, depot = mod.generate(graph, n, m, cfg, np.random.default_rng(1))
    assert len(customers) == n and len(vehicles) == m
    nodes = set(graph.nodes)
    assert depot["node_id"] in nodes
    assert all(c.node_id in nodes for c in customers)  # CORRECTNESS BAR
    assert len({c.node_id for c in customers}) == n and depot["node_id"] not in {
        c.node_id for c in customers
    }
    mod.check_feasible(customers, vehicles, cfg.rho_max)  # CORRECTNESS BAR
    for c in customers:
        assert cfg.demand_range[0] <= c.demand <= cfg.demand_range[1]
        assert 0 <= c.time_window_start < c.time_window_end <= cfg.shift_end_s
    widths = [c.time_window_end - c.time_window_start for c in customers]
    assert min(widths) < cfg.shift_end_s  # not all "anytime"
    assert all(
        v.current_node == depot["node_id"] and v.shift_end == cfg.shift_end_s for v in vehicles
    )

    mod.save(name, customers, vehicles, depot, tmp_path)
    c2, v2, d2 = mod.load(name, tmp_path)
    assert c2 == customers and v2 == vehicles and d2 == depot


def test_infeasible_demand_raises(graph):
    cfg = QTrafficConfig(city_bbox=BBOX, vehicle_capacity=1)
    with pytest.raises(ValueError, match="infeasible"):
        mod.generate(graph, 10, 2, cfg, np.random.default_rng(0))


def test_window_spread_across_shift(graph):
    """Tight windows spread over the shift: starts not clustered at 0, ends not at H."""
    cfg = QTrafficConfig(city_bbox=BBOX, tw_anytime_fraction=0.0)
    customers, _, _ = mod.generate(graph, 50, 10, cfg, np.random.default_rng(3))
    starts = np.array([c.time_window_start for c in customers])
    assert starts.std() > 0.1 * cfg.shift_end_s and starts.max() > 0.5 * cfg.shift_end_s
