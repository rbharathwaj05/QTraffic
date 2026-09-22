import json

import httpx
import numpy as np

from backend.road.osrm_client import OSRMClient


def _client(handler):
    return OSRMClient("http://osrm", transport=httpx.MockTransport(handler))


def test_route_parses_geometry_and_dedupes_leg_nodes():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        body = {
            "code": "Ok",
            "routes": [
                {
                    "duration": 120.5,
                    "distance": 900.0,
                    "geometry": {"coordinates": [[80.23, 13.04], [80.24, 13.05]]},
                    "legs": [
                        {"annotation": {"nodes": [1, 2, 3]}},
                        {"annotation": {"nodes": [3, 4]}},
                    ],
                }
            ],
        }
        return httpx.Response(200, text=json.dumps(body))

    r = _client(handler).route([(13.04, 80.23), (13.045, 80.235), (13.05, 80.24)])
    assert r.duration_s == 120.5 and r.distance_m == 900.0
    assert r.geometry == [(13.04, 80.23), (13.05, 80.24)]  # lon,lat -> lat,lon
    assert r.node_ids == [1, 2, 3, 4]
    assert "/route/v1/driving/80.230000,13.040000;" in seen["url"]
    assert "annotations=nodes" in seen["url"]


def test_table_nan_for_unroutable_and_source_dest_params():
    seen = {}

    def handler(req: httpx.Request):
        seen["params"] = dict(req.url.params)
        body = {"code": "Ok", "durations": [[0, None]], "distances": [[0, 5.0]]}
        return httpx.Response(200, text=json.dumps(body))

    dur, dist = _client(handler).table([(13.0, 80.0)], [(13.1, 80.1)])
    assert dur.shape == (1, 2) and np.isnan(dur[0, 1])
    assert dist[0, 1] == 5.0
    assert seen["params"]["sources"] == "0" and seen["params"]["destinations"] == "1"


def test_non_ok_code_raises():
    def handler(req):
        return httpx.Response(200, text=json.dumps({"code": "NoRoute", "message": "x"}))

    try:
        _client(handler).table([(13.0, 80.0)])
    except RuntimeError as exc:
        assert "NoRoute" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
