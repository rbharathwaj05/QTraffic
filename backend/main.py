"""FastAPI entrypoint. Routers are mounted here; no business logic lives in this file."""

from fastapi import FastAPI

from backend.config import QTrafficConfig

app = FastAPI(title="qtraffic")
config = QTrafficConfig()  # module-level singleton; validated on import


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for docker-compose / load balancers; no dependencies touched."""
    return {"status": "ok"}


# Domain routers (backend/api/*.py) are mounted here once they carry endpoints (Phase 7+).
