"""FastAPI entrypoint. Routers are mounted here; no business logic lives in this file."""

from fastapi import FastAPI

from backend.config import QTrafficConfig

app = FastAPI(title="qtraffic")
config = QTrafficConfig()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
