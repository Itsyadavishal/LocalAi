"""Async request queue and dispatch layer for LocalAi."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any
import uuid

import httpx

from server.core.inference_engine import engine
from server.utils.logger import get_logger

LLAMA_BASE_URL_TEMPLATE: str = "http://127.0.0.1:{port}"
WORKER_SLEEP_SECONDS: float = 0.05
DEFAULT_TIMEOUT_SECONDS: int = 120


@dataclass(slots=True)
class QueuedRequest:
    """Queued request metadata and completion future."""

    request_id: str
    endpoint: str
    payload: dict[str, Any]
    timeout_seconds: int
    stream: bool
    enqueued_at: float
    future: asyncio.Future[RequestResult]


@dataclass(slots=True)
class RequestResult:
    """Result returned from the request queue worker."""

    request_id: str
    status_code: int
    body: dict[str, Any] | None
    stream_iter: Any | None
    error: str | None


class RequestHandler:
    """Manage queued requests and single-worker dispatch to llama-server."""

    def __init__(
        self,
        llama_port: int,
        max_queue_depth: int = 20,
        default_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the request queue, worker state, and counters.

        Args:
            llama_port: Port used by the llama-server subprocess.
            max_queue_depth: Maximum queued requests allowed at once.
            default_timeout_seconds: Default timeout applied to queued requests.

        Returns:
            None.

        Raises:
            None.
        """
        self._llama_port = llama_port
        self._max_queue_depth = max_queue_depth
        self._default_timeout = default_timeout_seconds
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(maxsize=max_queue_depth)
        self._worker_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._logger = get_logger(__name__)
        self._request_count: int = 0
        self._error_count: int = 0

    async def start(self) -> None:
        """Start the HTTP client and background worker task.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if self._worker_task is not None and not self._worker_task.done():
            return

        self._client = httpx.AsyncClient(timeout=None)
        self._worker_task = asyncio.create_task(self._worker_loop(), name="localai-request-worker")
        self._logger.info("request handler started", port=self._llama_port)

    async def stop(self) -> None:
        """Stop the worker loop and close the HTTP client.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        while not self._queue.empty():
            try:
                queued_request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if not queued_request.future.done():
                queued_request.future.set_result(
                    RequestResult(
                        request_id=queued_request.request_id,
                        status_code=503,
                        body={"error": "Request handler stopped before request could be processed."},
                        stream_iter=None,
                        error="request handler stopped",
                    )
                )
            self._queue.task_done()

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        self._logger.info("request handler stopped")

    async def enqueue(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout_seconds: int | None = None,
        stream: bool = False,
    ) -> RequestResult:
        """Add a request to the queue and await the worker result.

        Args:
            endpoint: llama-server endpoint to call.
            payload: JSON payload forwarded to llama-server.
            timeout_seconds: Optional request timeout override.
            stream: Whether the caller requested streaming behavior.

        Returns:
            The completed request result.

        Raises:
            None.
        """
        request_id = str(uuid.uuid4())[:8]
        if not engine.get_status().running:
            self._error_count += 1
            return RequestResult(
                request_id=request_id,
                status_code=503,
                body={"error": "No model loaded. Load a model first."},
                stream_iter=None,
                error="no model loaded",
            )

        effective_timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
        future = asyncio.get_running_loop().create_future()
        queued_request = QueuedRequest(
            request_id=request_id,
            endpoint=endpoint,
            payload=payload,
            timeout_seconds=effective_timeout,
            stream=stream,
            enqueued_at=time.monotonic(),
            future=future,
        )

        try:
            self._queue.put_nowait(queued_request)
        except asyncio.QueueFull:
            self._error_count += 1
            return RequestResult(
                request_id=request_id,
                status_code=429,
                body={"error": "Request queue full. Try again shortly."},
                stream_iter=None,
                error="queue full",
            )

        self._logger.info(
            "request enqueued",
            request_id=request_id,
            endpoint=endpoint,
            queue_depth=self._queue.qsize(),
        )

        try:
            result = await asyncio.wait_for(asyncio.shield(queued_request.future), timeout=effective_timeout)
            return result
        except asyncio.TimeoutError:
            self._error_count += 1
            return RequestResult(
                request_id=request_id,
                status_code=408,
                body={"error": f"Request timed out after {effective_timeout}s"},
                stream_iter=None,
                error="request timeout",
            )

    async def _worker_loop(self) -> None:
        """Continuously process queued requests one at a time.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        try:
            while True:
                request = await self._queue.get()
                try:
                    result = await self._dispatch(request)
                    if not request.future.done():
                        request.future.set_result(result)
                except Exception as error:
                    self._error_count += 1
                    if not request.future.done():
                        request.future.set_result(
                            RequestResult(
                                request_id=request.request_id,
                                status_code=500,
                                body={"error": str(error)},
                                stream_iter=None,
                                error=str(error),
                            )
                        )
                finally:
                    self._queue.task_done()
                    await asyncio.sleep(WORKER_SLEEP_SECONDS)
        except asyncio.CancelledError:
            raise

    async def _dispatch(self, request: QueuedRequest) -> RequestResult:
        """Dispatch a queued request to llama-server and capture the result.

        Args:
            request: Queued request metadata and payload.

        Returns:
            The result of the llama-server request.

        Raises:
            RuntimeError: If the HTTP client has not been started.
        """
        self._logger.info(
            "dispatching request",
            request_id=request.request_id,
            endpoint=request.endpoint,
            stream=request.stream,
        )

        if request.stream:
            self._request_count += 1
            self._error_count += 1
            result = RequestResult(
                request_id=request.request_id,
                status_code=501,
                body={"error": "Streaming not yet implemented"},
                stream_iter=None,
                error="streaming not implemented",
            )
            duration_ms = round((time.monotonic() - request.enqueued_at) * 1000)
            self._logger.info(
                "request complete",
                request_id=request.request_id,
                status_code=result.status_code,
                duration_ms=duration_ms,
            )
            return result

        if self._client is None:
            raise RuntimeError("Request handler HTTP client is not started")

        target_url = f"{LLAMA_BASE_URL_TEMPLATE.format(port=self._llama_port)}{request.endpoint}"
        try:
            response = await self._client.post(
                target_url,
                json=request.payload,
                timeout=request.timeout_seconds,
            )
            try:
                body = response.json()
            except ValueError:
                body = {"error": "Invalid JSON response from llama-server"}
                if response.status_code < 400:
                    response = httpx.Response(status_code=502, request=response.request, json=body)
            result = RequestResult(
                request_id=request.request_id,
                status_code=response.status_code,
                body=body,
                stream_iter=None,
                error=None if response.status_code < 400 else body.get("error", "request failed"),
            )
        except httpx.ConnectError:
            result = RequestResult(
                request_id=request.request_id,
                status_code=503,
                body={"error": "llama-server not reachable"},
                stream_iter=None,
                error="llama-server not reachable",
            )
        except httpx.TimeoutException:
            result = RequestResult(
                request_id=request.request_id,
                status_code=408,
                body={"error": "llama-server timed out"},
                stream_iter=None,
                error="llama-server timed out",
            )
        except Exception as error:
            result = RequestResult(
                request_id=request.request_id,
                status_code=500,
                body={"error": str(error)},
                stream_iter=None,
                error=str(error),
            )

        self._request_count += 1
        if result.error is not None or result.status_code >= 400:
            self._error_count += 1

        duration_ms = round((time.monotonic() - request.enqueued_at) * 1000)
        self._logger.info(
            "request complete",
            request_id=request.request_id,
            status_code=result.status_code,
            duration_ms=duration_ms,
        )
        return result

    def get_stats(self) -> dict[str, int | bool]:
        """Return the current queue and worker statistics.

        Args:
            None.

        Returns:
            Queue depth and aggregate request counters.

        Raises:
            None.
        """
        return {
            "queue_depth": self._queue.qsize(),
            "max_queue_depth": self._max_queue_depth,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "worker_running": self._worker_task is not None and not self._worker_task.done(),
        }
