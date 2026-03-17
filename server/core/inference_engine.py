"""llama-server subprocess lifecycle management for LocalAi.

This module manages a single `llama-server.exe` process. It validates launch
inputs, spawns the subprocess, polls its `/health` endpoint until ready, and
stops it using a Windows-safe terminate/kill sequence. Model load decisions and
request routing are intentionally handled elsewhere.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time

import httpx

from server.utils.logger import get_logger

HEALTH_CHECK_INTERVAL_SECONDS: float = 1.0
HEALTH_CHECK_TIMEOUT_SECONDS: float = 60.0
HEALTH_CHECK_ENDPOINT: str = "/health"
LLAMA_HOST: str = "127.0.0.1"
PROCESS_WAIT_TIMEOUT_SECONDS: float = 5.0
READINESS_LOG_INTERVAL_POLLS: int = 10


@dataclass(slots=True)
class EngineConfig:
    """Runtime configuration for the llama-server process."""

    model_path: str
    mmproj_path: str | None
    ctx_size: int
    n_gpu_layers: int
    llama_bin: str
    llama_port: int


@dataclass(slots=True)
class EngineStatus:
    """Observed state of the llama-server process."""

    running: bool
    pid: int | None
    port: int | None
    model_path: str | None
    error: str | None


class InferenceEngine:
    """Manage a single llama-server subprocess for the LocalAi runtime."""

    def __init__(self) -> None:
        """Initialize the engine state.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self._process: subprocess.Popen[str] | None = None
        self._config: EngineConfig | None = None
        self._logger = get_logger(__name__)
        self._last_error: str | None = None

    async def start(self, config: EngineConfig) -> None:
        """Spawn llama-server and wait until its health endpoint is ready.

        Args:
            config: Process launch configuration for llama-server.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the llama binary, model file, or provided mmproj file is missing.
            RuntimeError: If the engine is already running or the subprocess exits during startup.
            TimeoutError: If the health endpoint never becomes ready before timeout.
        """
        resolved_llama_bin = Path(config.llama_bin).expanduser().resolve()
        if not resolved_llama_bin.is_file():
            message = f"llama-server binary not found: {resolved_llama_bin}"
            self._last_error = message
            raise FileNotFoundError(message)

        resolved_model_path = Path(config.model_path).expanduser().resolve()
        if not resolved_model_path.is_file():
            message = f"Model file not found: {resolved_model_path}"
            self._last_error = message
            raise FileNotFoundError(message)

        resolved_mmproj_path: Path | None = None
        if config.mmproj_path is not None:
            resolved_mmproj_path = Path(config.mmproj_path).expanduser().resolve()
            if not resolved_mmproj_path.is_file():
                message = f"mmproj file not found: {resolved_mmproj_path}"
                self._last_error = message
                raise FileNotFoundError(message)

        if self._process is not None and self._process.poll() is None:
            message = "Engine already running. Call stop() before starting again."
            self._last_error = message
            raise RuntimeError(message)

        command = [
            str(resolved_llama_bin),
            "--model",
            str(resolved_model_path),
            "--ctx-size",
            str(config.ctx_size),
            "--n-gpu-layers",
            str(config.n_gpu_layers),
            "--port",
            str(config.llama_port),
            "--host",
            LLAMA_HOST,
            "--parallel",
            "1",
            "--no-mmap",
        ]
        if resolved_mmproj_path is not None:
            command.extend(["--mmproj", str(resolved_mmproj_path)])

        self._logger.debug("spawning llama-server", command=command)

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process_config = EngineConfig(
            model_path=str(resolved_model_path),
            mmproj_path=str(resolved_mmproj_path) if resolved_mmproj_path is not None else None,
            ctx_size=config.ctx_size,
            n_gpu_layers=config.n_gpu_layers,
            llama_bin=str(resolved_llama_bin),
            llama_port=config.llama_port,
        )

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                creationflags=creationflags,
            )
            self._config = process_config
            self._last_error = None
            await self._wait_until_ready(process_config.llama_port)
            self._logger.info(
                "llama-server ready",
                port=process_config.llama_port,
                pid=self._process.pid if self._process is not None else None,
            )
        except Exception as error:
            self._last_error = str(error)
            await self.stop()
            raise

    async def stop(self) -> None:
        """Terminate llama-server using a Windows-safe shutdown sequence.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if self._process is None:
            self._logger.warning("llama-server stop requested with no active process")
            return

        process = self._process
        self._logger.info("stopping llama-server", pid=process.pid)
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=PROCESS_WAIT_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            self._process = None
            self._config = None
        self._logger.info("llama-server stopped")

    def get_status(self) -> EngineStatus:
        """Return the current subprocess status without mutating engine state.

        Args:
            None.

        Returns:
            The current engine status snapshot.

        Raises:
            None.
        """
        if self._process is None:
            return EngineStatus(
                running=False,
                pid=None,
                port=self._config.llama_port if self._config is not None else None,
                model_path=self._config.model_path if self._config is not None else None,
                error=self._last_error,
            )

        return_code = self._process.poll()
        if return_code is not None:
            error = f"Process exited unexpectedly with code {return_code}"
            self._last_error = error
            return EngineStatus(
                running=False,
                pid=self._process.pid,
                port=self._config.llama_port if self._config is not None else None,
                model_path=self._config.model_path if self._config is not None else None,
                error=error,
            )

        return EngineStatus(
            running=True,
            pid=self._process.pid,
            port=self._config.llama_port if self._config is not None else None,
            model_path=self._config.model_path if self._config is not None else None,
            error=None,
        )

    async def _wait_until_ready(self, port: int) -> None:
        """Poll the llama-server health endpoint until it becomes ready.

        Args:
            port: Local port exposed by llama-server.

        Returns:
            None.

        Raises:
            RuntimeError: If the subprocess exits during startup.
            TimeoutError: If readiness is not reached before the timeout.
        """
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT_SECONDS
        health_url = f"http://{LLAMA_HOST}:{port}{HEALTH_CHECK_ENDPOINT}"
        poll_count = 0

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self._process is None or self._process.poll() is not None:
                    message = "llama-server exited during startup. Check model path and GPU layers."
                    self._last_error = message
                    raise RuntimeError(message)

                poll_count += 1
                if poll_count % READINESS_LOG_INTERVAL_POLLS == 0:
                    self._logger.info("waiting for llama-server", port=port, polls=poll_count)

                try:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass

                await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

        message = (
            "llama-server did not become ready within "
            f"{HEALTH_CHECK_TIMEOUT_SECONDS}s"
        )
        self._last_error = message
        raise TimeoutError(message)


engine = InferenceEngine()
