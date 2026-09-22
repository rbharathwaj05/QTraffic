import pytest

import backend.road.osm_loader as mod  # import must succeed even while bodies are stubs


@pytest.mark.skip(reason="phase 0 stub: no behaviour to test yet")
def test_osm_loader():
    assert mod
