"""HTTP/WS surface for the fleet domain. Phase 0: router placeholder, no endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/fleet", tags=["fleet"])
