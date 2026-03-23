"""Administrative API endpoints for LocalAi."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import uvicorn

from server.config.schemas import LoadModelRequest
from server.core.inference_engine import engine
from server.utils.logger import get_logger

router = APIRouter(prefix="/localai", tags=["admin"])
logger = get_logger(__name__)


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


@router.get("/status")
async def get_status(request: Request) -> dict[str, object]:
    """Return full LocalAi runtime status.

    Args:
        request: Incoming FastAPI request object.

    Returns:
        Current engine, model, VRAM, queue, uptime, and monitor status.

    Raises:
        None.
    """
    status_snapshot = request.app.state.metrics_collector.collect_runtime_status().to_dict()
    status_snapshot["monitor"] = request.app.state.health_monitor.get_status()
    return status_snapshot


@router.post("/models/load", response_model=None)
async def load_model(body: LoadModelRequest, request: Request) -> dict[str, object] | JSONResponse:
    """Load a model into the shared inference engine.

    Args:
        body: Model load request with optional context override.
        request: Incoming FastAPI request object.

    Returns:
        Load result with decision details, or an error response.

    Raises:
        None.
    """
    model_manager = request.app.state.model_manager
    resolved_id = model_manager.resolve_model_id(body.model_id)
    if resolved_id is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Model not found: {body.model_id}",
                "installed_models": sorted(model.model_id for model in model_manager.list_models()),
            },
        )

    installed_model = model_manager.get_model(resolved_id)
    if installed_model is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Resolved model not found: {resolved_id}"},
        )

    original_ctx_size = installed_model.config.ctx_size
    ctx_size_used = body.ctx_size if body.ctx_size is not None else original_ctx_size
    if body.ctx_size is not None:
        installed_model.config.ctx_size = body.ctx_size

    runtime_config = request.app.state.config
    try:
        decision = await model_manager.load_model(
            resolved_id,
            safety_margin_mb=runtime_config.vram.safety_margin_mb,
            runtime_overhead_mb=runtime_config.vram.runtime_overhead_mb,
        )
    except ValueError as error:
        logger.warning("model load rejected", model_id=body.model_id, error=str(error))
        return JSONResponse(status_code=404, content={"error": str(error)})
    except FileNotFoundError as error:
        logger.error("model load file missing", model_id=resolved_id, error=str(error))
        return JSONResponse(status_code=500, content={"error": str(error)})
    except RuntimeError as error:
        logger.warning("model load failed", model_id=resolved_id, error=str(error))
        return JSONResponse(status_code=503, content={"error": str(error)})
    finally:
        installed_model.config.ctx_size = original_ctx_size

    logger.info(
        "admin model load complete",
        requested_model_id=body.model_id,
        resolved_model_id=resolved_id,
        ctx_size_used=ctx_size_used,
        n_gpu_layers=decision.n_gpu_layers,
    )
    return {
        "status": "ok",
        "model_id": resolved_id,
        "display_name": installed_model.config.display_name,
        "ctx_size": ctx_size_used,
        "decision": {
            "can_load": decision.can_load,
            "full_gpu": decision.full_gpu,
            "n_gpu_layers": decision.n_gpu_layers,
            "vram_required_mb": decision.vram_required_mb,
            "vram_free_mb": decision.vram_free_mb,
            "reason": decision.reason,
        },
        "engine": {
            "running": engine.get_status().running,
            "pid": engine.get_status().pid,
            "port": engine.get_status().port,
        },
    }


@router.post("/models/unload")
async def unload_model(request: Request) -> dict[str, object]:
    """Unload the currently loaded model.

    Args:
        request: Incoming FastAPI request object.

    Returns:
        Unload status payload.

    Raises:
        None.
    """
    model_manager = request.app.state.model_manager
    loaded_id = model_manager.get_loaded_model_id()
    if loaded_id is None:
        logger.warning("admin unload requested with no model loaded")
        return {
            "status": "ok",
            "model_unloaded": False,
            "model_id": None,
            "message": "No model is currently loaded.",
        }

    await model_manager.unload_model()
    logger.info("admin model unload complete", model_id=loaded_id)
    return {
        "status": "ok",
        "model_unloaded": True,
        "model_id": loaded_id,
    }
