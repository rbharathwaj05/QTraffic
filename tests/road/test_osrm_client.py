import numpy as np

from backend.road.osrm_client import OSRMClient


def test_table_tiles_requests_and_reassembles(monkeypatch):
    pts = [(13.0 + i * 0.01, 80.0) for i in range(6)]
    calls = []

    def fake_get(self, service, coords, params):
        assert service == "table" and len(coords) <= 4
        calls.append(coords)
        src = [int(s) for s in params["sources"].split(";")]
        dst = [int(s) for s in params["destinations"].split(";")]
        # cost = 100 * |row_index - col_index| in the original point list
        row = [pts.index(coords[s]) for s in src]
        col = [pts.index(coords[d]) for d in dst]
        m = [[100.0 * abs(r - c) if r != c else None for c in col] for r in row]
        return {"code": "Ok", "durations": m, "distances": m}

    monkeypatch.setattr(OSRMClient, "_get", fake_get)
    dur, dist = OSRMClient("http://x", max_table_size=4).table(pts)
    assert len(calls) == 9  # 3 row blocks x 3 col blocks of size 2
    assert dur.shape == (6, 6)
    expect = 100.0 * np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
    assert np.isnan(dur).sum() == 6  # diagonal null -> nan
    off = ~np.eye(6, dtype=bool)
    assert np.allclose(dur[off], expect[off]) and np.allclose(dist[off], expect[off])


def test_route_parses_geometry_and_nodes(monkeypatch):
    body = {
        "code": "Ok",
        "routes": [
            {
                "duration": 12.5,
                "distance": 300.0,
                "geometry": {"coordinates": [[80.0, 13.0], [80.1, 13.1]]},
                "legs": [{"annotation": {"nodes": [1, 2, 3]}}],
            }
        ],
    }
    monkeypatch.setattr(OSRMClient, "_get", lambda self, s, c, p: body)
    r = OSRMClient("http://x").route([(13.0, 80.0), (13.1, 80.1)])
    assert r.duration_s == 12.5 and r.distance_m == 300.0
    assert r.geometry == [(13.0, 80.0), (13.1, 80.1)]
    assert r.node_ids == [1, 2, 3]
