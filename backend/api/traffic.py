"""HTTP/WS surface for the traffic domain. Phase 0: router placeholder, no endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/traffic", tags=["traffic"])
