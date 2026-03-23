"""Monitor LocalAi runtime health and periodically log service snapshots."""

from __future__ import annotations

import asyncio
import time

from server.core.metrics_collector import MetricsCollector, RuntimeStatusSnapshot
from server.core.vram_manager import log_vram_snapshot
from server.utils.logger import get_logger

DEFAULT_MONITOR_INTERVAL_SECONDS: float = 30.0


class HealthMonitor:
    """Run a lightweight background health monitor for the LocalAi service."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        interval_seconds: float = DEFAULT_MONITOR_INTERVAL_SECONDS,
    ) -> None:
        """Initialize monitor state and background task configuration.

        Args:
            metrics_collector: Shared metrics collector used to build snapshots.
            interval_seconds: Seconds between periodic health observations.

        Returns:
            None.

        Raises:
            None.
        """
        self._metrics_collector = metrics_collector
        self._interval_seconds = interval_seconds
        self._logger = get_logger(__name__)
        self._task: asyncio.Task[None] | None = None
        self._last_snapshot: RuntimeStatusSnapshot | None = None
        self._last_poll_monotonic: float | None = None

    async def start(self) -> None:
        """Start the background health monitor loop if it is not already running.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if self._task is not None and not self._task.done():
            return

        self._last_snapshot = self._metrics_collector.collect_runtime_status()
        self._last_poll_monotonic = time.monotonic()
        self._task = asyncio.create_task(self._run_loop(), name="localai-health-monitor")
        self._logger.info("health monitor started", interval_seconds=self._interval_seconds)

    async def stop(self) -> None:
        """Stop the background health monitor loop.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if self._task is None:
            self._logger.warning("health monitor stop requested with no active task")
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

        self._logger.info("health monitor stopped")

    def get_last_snapshot(self) -> dict[str, object] | None:
        """Return the last recorded full runtime snapshot, if available.

        Args:
            None.

        Returns:
            Last runtime status snapshot as a dictionary, or None.

        Raises:
            None.
        """
        if self._last_snapshot is None:
            return None
        return self._last_snapshot.to_dict()

    def get_status(self) -> dict[str, object]:
        """Return monitor status and the latest poll timing.

        Args:
            None.

        Returns:
            Background task status and last poll age.

        Raises:
            None.
        """
        running = self._task is not None and not self._task.done()
        seconds_since_last_poll = None
        if self._last_poll_monotonic is not None:
            seconds_since_last_poll = round(time.monotonic() - self._last_poll_monotonic, 3)

        return {
            "running": running,
            "interval_seconds": self._interval_seconds,
            "seconds_since_last_poll": seconds_since_last_poll,
        }

    async def _run_loop(self) -> None:
        """Continuously refresh health snapshots and emit periodic VRAM logs.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        try:
            while True:
                await asyncio.sleep(self._interval_seconds)
                self._last_snapshot = self._metrics_collector.collect_runtime_status()
                self._last_poll_monotonic = time.monotonic()
                log_vram_snapshot()
                self._logger.debug(
                    "health snapshot updated",
                    uptime_seconds=self._last_snapshot.uptime_seconds,
                    queue_depth=self._last_snapshot.queue.get("queue_depth", 0),
                    model_loaded=self._last_snapshot.model.get("loaded", False),
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._logger.error("health monitor failed", error=str(error))
            raise
