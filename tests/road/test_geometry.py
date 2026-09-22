"""Phase 1: haversine against a known Chennai distance, matrix == scalar form,
nearest-node snap, and Google's polyline reference vector."""

import numpy as np

from backend.road import geometry as g


def test_haversine_known_distance():
    # Chennai Central -> Chennai Airport, ~14.6 km (dlat 9.85 km, dlon 10.8 km)
    d = g.haversine(13.0827, 80.2707, 12.9941, 80.1709)
    assert 14_500 < d < 14_800
    assert g.haversine(13.0, 80.0, 13.0, 80.0) == 0.0


def test_haversine_matrix_matches_scalar():
    lats = np.array([13.0, 13.1, 12.9])
    lons = np.array([80.0, 80.2, 80.1])
    m = g.haversine_matrix(lats, lons)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0)
    assert np.isclose(m[0, 1], g.haversine(13.0, 80.0, 13.1, 80.2))
    assert np.allclose(m, m.T)


def test_snap_to_node():
    lats = np.array([13.0, 13.1, 12.9])
    lons = np.array([80.0, 80.2, 80.1])
    assert g.snap_to_node(lats, lons, 12.91, 80.09) == 2


def test_decode_polyline():
    # Google's reference example
    pts = g.decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    assert np.allclose(pts, [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)])
