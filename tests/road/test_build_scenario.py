import time

import numpy as np
import pytest

from backend.config import QTrafficConfig
from backend.road import build_scenario as mod

from .fake_osrm import FakeOSRM
from .test_osm_loader import BBOX


def test_build_and_load_scenario(tmp_path, monkeypatch):
    cfg = QTrafficConfig(city_bbox=BBOX, city_cache_dir=tmp_path / "city")
    try:
        mod.load_road_graph(cfg)
    except Exception as exc:
        pytest.skip(f"OSM download unavailable: {exc!r}")
    client = FakeOSRM()
    monkeypatch.setattr(mod, "OSRMClient", lambda url: client)
    t = time.perf_counter()
    m, idx, pts = mod.build_scenario("t50", 50, cfg, np.random.default_rng(0), tmp_path / "sc")
    print(f"\nbuild_scenario N=50 (fake OSRM): {time.perf_counter() - t:.2f}s")
    assert client.table_calls == 1 and client.route_calls == 50 * 51
    assert m.duration_s.shape == (51, 51) and idx.n == 51 and len(pts) == 51
    m2, idx2, pts2 = mod.load_scenario("t50", tmp_path / "sc")
    assert np.array_equal(m2.distance_m, m.distance_m) and idx2.n == 51
    assert np.array_equal(pts2, np.array(pts)) and m2.edge_t0.shape == m.edge_t0.shape
