"""Administrative API endpoints for LocalAi."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
import uvicorn

router = APIRouter(prefix="/localai", tags=["admin"])


@router.post("/shutdown")
async def shutdown(request: Request) -> dict[str, str]:
    """Trigger graceful server shutdown.

    Schedules uvicorn shutdown after the response is sent so the FastAPI
    lifespan cleanup block can complete fully.

    Args:
        request: Incoming FastAPI request object.

    Returns:
        A shutdown status payload.

    Raises:
        None.
    """
    server: uvicorn.Server = request.app.state.uvicorn_server

    async def _shutdown() -> None:
        await asyncio.sleep(0.1)
        server.should_exit = True

    asyncio.create_task(_shutdown())
    return {"status": "shutting down"}
