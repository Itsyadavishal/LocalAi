"""OpenAI-compatible inference endpoints for LocalAi."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from server.utils.logger import get_logger

router = APIRouter(tags=["openai"])
logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """Single chat message in OpenAI-compatible format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request body for OpenAI-compatible chat completions."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    stream: bool = False
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: list[str] | None = None

    model_config = ConfigDict(extra="allow")


class CompletionRequest(BaseModel):
    """Request body for OpenAI-compatible raw completions."""

    model: str
    prompt: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    stream: bool = False

    model_config = ConfigDict(extra="allow")


def error_response(status_code: int, message: str, error_type: str, code: str) -> JSONResponse:
    """Build an OpenAI-compatible JSON error response.

    Args:
        status_code: HTTP status code to return.
        message: Human-readable error message.
        error_type: OpenAI-style error type string.
        code: OpenAI-style error code string.

    Returns:
        JSONResponse using the OpenAI error envelope.

    Raises:
        None.
    """
    logger.warning(
        "openai api error",
        status_code=status_code,
        error_type=error_type,
        code=code,
        message=message,
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


async def _sse_generator(stream_iter: AsyncIterator[str]) -> AsyncIterator[str]:
    """Convert llama-server line streaming into SSE output.

    Args:
        stream_iter: Async line iterator returned from the request handler.

    Returns:
        SSE-formatted line stream.

    Raises:
        None.
    """
    saw_done = False
    try:
        async for line in stream_iter:
            if not line:
                continue
            normalized_line = line if line.startswith("data:") else f"data: {line}"
            if normalized_line.strip() == "data: [DONE]":
                saw_done = True
            yield f"{normalized_line}\n\n"
    except Exception as error:
        yield f"data: {json.dumps({'error': str(error)})}\n\n"
    finally:
        if not saw_done:
            yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, object]:
    """List all installed models in OpenAI-compatible format.

    Args:
        request: Incoming FastAPI request object.

    Returns:
        OpenAI-style model list payload.

    Raises:
        None.
    """
    models = request.app.state.model_manager.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": model.model_id,
                "object": "model",
                "owned_by": "localai",
                "capabilities": [capability.value for capability in model.config.capabilities],
            }
            for model in models
        ],
    }


@router.get("/v1/models/{model_id}", response_model=None)
async def get_model(model_id: str, request: Request) -> dict[str, object] | JSONResponse:
    """Return details for a specific installed model using fuzzy resolution.

    Args:
        model_id: Requested model identifier or partial identifier.
        request: Incoming FastAPI request object.

    Returns:
        OpenAI-style model object, or an OpenAI-style error response.

    Raises:
        None.
    """
    model_manager = request.app.state.model_manager
    resolved_model_id = model_manager.resolve_model_id(model_id)
    if resolved_model_id is None:
        return error_response(
            status_code=404,
            message=f"Model '{model_id}' not found.",
            error_type="invalid_request_error",
            code="model_not_found",
        )

    model = model_manager.get_model(resolved_model_id)
    if model is None:
        return error_response(
            status_code=404,
            message=f"Model '{resolved_model_id}' not found.",
            error_type="invalid_request_error",
            code="model_not_found",
        )

    return {
        "id": model.model_id,
        "object": "model",
        "owned_by": "localai",
        "capabilities": [capability.value for capability in model.config.capabilities],
    }


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> dict[str, Any] | JSONResponse | StreamingResponse:
    """Handle OpenAI-compatible chat completions.

    Args:
        body: Validated chat completions request body.
        request: Incoming FastAPI request object.

    Returns:
        llama-server JSON payload, SSE stream, or an OpenAI-style error response.

    Raises:
        None.
    """
    model_manager = request.app.state.model_manager
    resolved_model_id = model_manager.resolve_model_id(body.model)
    if resolved_model_id is None:
        return error_response(
            status_code=404,
            message=f"Model '{body.model}' not found. Load a model first.",
            error_type="invalid_request_error",
            code="model_not_found",
        )

    if model_manager.get_loaded_model_id() != resolved_model_id:
        return error_response(
            status_code=503,
            message=(
                f"Model '{resolved_model_id}' is installed but not loaded. "
                "POST /localai/models/load first."
            ),
            error_type="service_unavailable",
            code="model_not_loaded",
        )

    payload = body.model_dump(exclude_none=True)
    handler = request.app.state.request_handler

    if body.stream:
        result = await handler.enqueue(
            endpoint="/v1/chat/completions",
            payload=payload,
            stream=True,
        )
        if result.error:
            return error_response(
                status_code=result.status_code,
                message=result.error,
                error_type="server_error",
                code="inference_error",
            )
        return StreamingResponse(
            _sse_generator(result.stream_iter),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = await handler.enqueue(
        endpoint="/v1/chat/completions",
        payload=payload,
        stream=False,
    )

    if result.error:
        return error_response(
            status_code=result.status_code,
            message=result.error,
            error_type="server_error",
            code="inference_error",
        )

    return result.body or {}


@router.post("/v1/completions", response_model=None)
async def completions(
    body: CompletionRequest,
    request: Request,
) -> dict[str, Any] | JSONResponse | StreamingResponse:
    """Handle OpenAI-compatible raw completions.

    Args:
        body: Validated completions request body.
        request: Incoming FastAPI request object.

    Returns:
        llama-server JSON payload, SSE stream, or an OpenAI-style error response.

    Raises:
        None.
    """
    model_manager = request.app.state.model_manager
    resolved_model_id = model_manager.resolve_model_id(body.model)
    if resolved_model_id is None:
        return error_response(
            status_code=404,
            message=f"Model '{body.model}' not found.",
            error_type="invalid_request_error",
            code="model_not_found",
        )

    if model_manager.get_loaded_model_id() != resolved_model_id:
        return error_response(
            status_code=503,
            message=f"Model '{resolved_model_id}' is not loaded.",
            error_type="service_unavailable",
            code="model_not_loaded",
        )

    payload = body.model_dump(exclude_none=True)
    handler = request.app.state.request_handler

    if body.stream:
        result = await handler.enqueue(
            endpoint="/v1/completions",
            payload=payload,
            stream=True,
        )
        if result.error:
            return error_response(
                status_code=result.status_code,
                message=result.error,
                error_type="server_error",
                code="inference_error",
            )
        return StreamingResponse(
            _sse_generator(result.stream_iter),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    result = await handler.enqueue(
        endpoint="/v1/completions",
        payload=payload,
        stream=False,
    )

    if result.error:
        return error_response(
            status_code=result.status_code,
            message=result.error,
            error_type="server_error",
            code="inference_error",
        )

    return result.body or {}
