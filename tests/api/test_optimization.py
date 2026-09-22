import pytest

import backend.api.optimization as mod  # import must succeed even while bodies are stubs


@pytest.mark.skip(reason="phase 0 stub: no behaviour to test yet")
def test_optimization():
    assert mod
