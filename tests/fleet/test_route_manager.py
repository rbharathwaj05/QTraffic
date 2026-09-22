import pytest

import backend.fleet.route_manager as mod  # import must succeed even while bodies are stubs


@pytest.mark.skip(reason="phase 0 stub: no behaviour to test yet")
def test_route_manager():
    assert mod
