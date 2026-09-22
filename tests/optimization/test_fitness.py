import pytest

import backend.optimization.fitness as mod  # import must succeed even while bodies are stubs


@pytest.mark.skip(reason="phase 0 stub: no behaviour to test yet")
def test_fitness():
    assert mod
