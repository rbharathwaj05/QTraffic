"""HTTP/WS surface for the websocket domain. Phase 0: router placeholder, no endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/websocket", tags=["websocket"])
