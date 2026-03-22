"""Central API router registration for LocalAi."""

from __future__ import annotations

from fastapi import FastAPI

from server.api.admin import router as admin_router
from server.api.openai_compat import router as openai_router


def register_routers(app: FastAPI) -> None:
    """Mount all API routers onto the FastAPI application.

    Args:
        app: FastAPI application instance.

    Returns:
        None.

    Raises:
        None.
    """
    app.include_router(openai_router)
    app.include_router(admin_router)
