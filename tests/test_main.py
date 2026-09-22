"""Phase 0: the FastAPI app boots and /health answers."""

from fastapi.testclient import TestClient

from backend.main import app


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
