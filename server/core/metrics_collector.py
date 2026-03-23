"""Aggregate runtime metrics for LocalAi observability endpoints and monitors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    ) -> None:
        """Initialize runtime metric dependencies.

        Args:
            app_version: Exposed LocalAi application version string.
            model_manager: Shared model manager instance.
            request_handler: Shared queue/request handler instance.
            inference_engine: Shared inference engine instance.
            start_monotonic: Optional monotonic start time override.

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
