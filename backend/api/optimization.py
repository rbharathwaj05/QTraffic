"""HTTP/WS surface for the optimization domain. Phase 0: router placeholder, no endpoints."""

from fastapi import APIRouter

# Phase 0-6: no routes yet and not mounted in backend/main.py; kept so imports resolve.

router = APIRouter(prefix="/optimization", tags=["optimization"])
