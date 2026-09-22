"""HTTP/WS surface for the optimization domain. Phase 0: router placeholder, no endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/optimization", tags=["optimization"])
