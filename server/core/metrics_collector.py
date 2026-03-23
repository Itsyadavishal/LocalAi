"""Aggregate runtime metrics for LocalAi observability endpoints and monitors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from server.core.inference_engine import InferenceEngine
from server.core.model_manager import ModelManager
from server.core.request_handler import RequestHandler
from server.utils.gpu_utils import VramInfo, get_vram_info


@dataclass(slots=True)
class HealthSnapshot:
    """Compact health payload for the public /health endpoint."""

    status: str
    version: str
    model_loaded: bool
    vram_used_mb: int
    vram_total_mb: int
    queue_depth: int

    def to_dict(self) -> dict[str, object]:
        """Return the health snapshot as a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class RuntimeStatusSnapshot:
    """Full runtime status payload for administrative inspection."""

    status: str
    version: str
    engine: dict[str, Any]
    model: dict[str, Any]
    vram: dict[str, Any]
    queue: dict[str, int | bool]
    uptime_seconds: int

    def to_dict(self) -> dict[str, object]:
        """Return the runtime status snapshot as a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class ModelMetrics:
    """Aggregate per-model request counters and latency totals."""

    requests_total: int = 0
    errors_total: int = 0
    total_latency_ms: float = 0.0
    last_used: str | None = None


class MetricsCollector:
    """Collect live engine, model, VRAM, queue, and uptime metrics."""

    def __init__(
        self,
        *,
        app_version: str,
        model_manager: ModelManager,
        request_handler: RequestHandler,
        inference_engine: InferenceEngine,
        start_monotonic: float | None = None,
        log_path: str | None = None,
    ) -> None:
        """Initialize runtime metric dependencies.

        Args:
            app_version: Exposed LocalAi application version string.
            model_manager: Shared model manager instance.
            request_handler: Shared queue/request handler instance.
            inference_engine: Shared inference engine instance.
            start_monotonic: Optional monotonic start time override.
            log_path: Optional path to the structured LocalAi log file.

        Returns:
            None.

        Raises:
            None.
        """
        self._app_version = app_version
        self._model_manager = model_manager
        self._request_handler = request_handler
        self._inference_engine = inference_engine
        self._start_monotonic = start_monotonic if start_monotonic is not None else time.monotonic()
        self._model_metrics: dict[str, ModelMetrics] = {}
        self._last_total_requests = 0
        self._last_total_errors = 0
        self._log_path = Path(log_path) if log_path is not None else Path(__file__).resolve().parents[2] / "logs" / "localai.log"
        self._log_offset = self._log_path.stat().st_size if self._log_path.exists() else 0

    def get_uptime_seconds(self) -> int:
        """Return the current service uptime in whole seconds.

        Args:
            None.

        Returns:
            Whole seconds since the collector started.

        Raises:
            None.
        """
        return int(time.monotonic() - self._start_monotonic)

    def get_vram_snapshot(self) -> VramInfo:
        """Return the latest VRAM snapshot from the GPU utility layer.

        Args:
            None.

        Returns:
            Current VRAM information.

        Raises:
            None.
        """
        return get_vram_info()

    def collect_health_snapshot(self) -> HealthSnapshot:
        """Build the compact health payload used by the public API.

        Args:
            None.

        Returns:
            Current public health snapshot.

        Raises:
            None.
        """
        self._refresh_model_metrics()
        vram = self.get_vram_snapshot()
        queue_stats = self._request_handler.get_stats()
        return HealthSnapshot(
            status="ok",
            version=self._app_version,
            model_loaded=self._model_manager.get_loaded_model_id() is not None,
            vram_used_mb=vram.used_mb,
            vram_total_mb=vram.total_mb,
            queue_depth=int(queue_stats["queue_depth"]),
        )

    def collect_runtime_status(self) -> RuntimeStatusSnapshot:
        """Build the full runtime status payload for administrative endpoints.

        Args:
            None.

        Returns:
            Full runtime status snapshot.

        Raises:
            None.
        """
        self._refresh_model_metrics()
        engine_status = self._inference_engine.get_status()
        loaded_model_id = self._model_manager.get_loaded_model_id()
        loaded_model = self._model_manager.get_model(loaded_model_id) if loaded_model_id else None
        vram = self.get_vram_snapshot()
        queue_stats = self._request_handler.get_stats()

        return RuntimeStatusSnapshot(
            status="ok",
            version=self._app_version,
            engine={
                "running": engine_status.running,
                "pid": engine_status.pid,
                "port": engine_status.port,
                "model_path": engine_status.model_path,
                "error": engine_status.error,
            },
            model={
                "loaded": loaded_model_id is not None,
                "model_id": loaded_model_id,
                "display_name": loaded_model.config.display_name if loaded_model else None,
            },
            vram={
                "gpu_name": vram.gpu_name,
                "used_mb": vram.used_mb,
                "free_mb": vram.free_mb,
                "total_mb": vram.total_mb,
            },
            queue=queue_stats,
            uptime_seconds=self.get_uptime_seconds(),
        )

    def get_snapshot(self) -> dict[str, object]:
        """Return the metrics endpoint snapshot in the expected API shape.

        Args:
            None.

        Returns:
            Runtime metrics including uptime, VRAM, queue, per-model stats,
            and the currently loaded model identifier.

        Raises:
            None.
        """
        self._refresh_model_metrics()
        vram = self.get_vram_snapshot()
        queue_stats = self._request_handler.get_stats()
        loaded_model_id = self._model_manager.get_loaded_model_id()
        models_snapshot: dict[str, dict[str, object]] = {}

        for installed_model in self._model_manager.list_models():
            metrics = self._model_metrics.setdefault(installed_model.model_id, ModelMetrics())
            average_latency = 0.0
            if metrics.requests_total > 0:
                average_latency = round(metrics.total_latency_ms / metrics.requests_total, 3)
            models_snapshot[installed_model.model_id] = {
                "requests_total": metrics.requests_total,
                "errors_total": metrics.errors_total,
                "avg_latency_ms": average_latency,
                "last_used": metrics.last_used,
            }

        return {
            "uptime_seconds": round(time.monotonic() - self._start_monotonic, 3),
            "server_version": self._app_version,
            "vram": {
                "used_mb": vram.used_mb,
                "free_mb": vram.free_mb,
                "total_mb": vram.total_mb,
                "gpu_name": vram.gpu_name,
            },
            "queue": {
                "depth": int(queue_stats["queue_depth"]),
                "max_depth": int(queue_stats["max_queue_depth"]),
                "total_requests": int(queue_stats["total_requests"]),
                "total_errors": int(queue_stats["total_errors"]),
                "worker_running": bool(queue_stats["worker_running"]),
            },
            "models": models_snapshot,
            "loaded_model_id": loaded_model_id,
        }

    def _refresh_model_metrics(self) -> None:
        """Refresh per-model counters by attributing new handler deltas.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        queue_stats = self._request_handler.get_stats()
        total_requests = int(queue_stats["total_requests"])
        total_errors = int(queue_stats["total_errors"])
        request_delta = max(0, total_requests - self._last_total_requests)
        error_delta = max(0, total_errors - self._last_total_errors)
        latency_sum_ms = self._consume_new_latency_sum()
        loaded_model_id = self._model_manager.get_loaded_model_id()

        if loaded_model_id is not None and (request_delta > 0 or error_delta > 0 or latency_sum_ms > 0):
            model_metrics = self._model_metrics.setdefault(loaded_model_id, ModelMetrics())
            model_metrics.requests_total += request_delta
            model_metrics.errors_total += error_delta
            model_metrics.total_latency_ms += latency_sum_ms
            model_metrics.last_used = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        self._last_total_requests = total_requests
        self._last_total_errors = total_errors

    def _consume_new_latency_sum(self) -> float:
        """Read newly appended request-complete log entries and sum latencies.

        Args:
            None.

        Returns:
            Sum of new request completion durations in milliseconds.

        Raises:
            None.
        """
        if not self._log_path.exists():
            return 0.0

        latency_sum_ms = 0.0
        current_size = self._log_path.stat().st_size
        if current_size < self._log_offset:
            self._log_offset = 0

        with self._log_path.open("r", encoding="utf-8") as log_file:
            log_file.seek(self._log_offset)
            for line in log_file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "request complete":
                    duration_ms = payload.get("duration_ms")
                    if isinstance(duration_ms, (int, float)):
                        latency_sum_ms += float(duration_ms)
            self._log_offset = log_file.tell()

        return latency_sum_ms
