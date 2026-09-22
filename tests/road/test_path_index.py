import pytest

import backend.road.path_index as mod  # import must succeed even while bodies are stubs


@pytest.mark.skip(reason="phase 0 stub: no behaviour to test yet")
def test_path_index():
    assert mod
