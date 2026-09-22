import pytest

import backend.optimization.pso as mod  # import must succeed even while bodies are stubs


@pytest.mark.skip(reason="phase 0 stub: no behaviour to test yet")
def test_pso():
    assert mod
