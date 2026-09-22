import pytest

from backend.config import QTrafficConfig


def test_defaults_valid():
    QTrafficConfig().validate()


@pytest.mark.parametrize(
    "kw",
    [
        {"w_t": 0.5},
        {"eta_a": 0.6},
        {"warm_fraction": 0.3},
        {"r_E": 0.25},
        {"theta_soft": 0.2},
    ],
)
def test_invalid_rejected(kw):
    with pytest.raises(ValueError):
        QTrafficConfig(**kw)
