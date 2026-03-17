"""FastAPI application bootstrap for the LocalAi server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, Request
from rich import print as rich_print

from server.config.config_loader import load_config
from server.config.schemas import LocalAiConfig

APP_TITLE = "LocalAi"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Local AI inference infrastructure"
CONFIG_FILE_NAME = "localai.config.json"
STARTUP_BANNER = (
    "┌─────────────────────────────────────┐\n"
    "│  LocalAi v0.1.0                     │\n"
    "│  Status:  Starting...               │\n"
    "│  Host:    {host:<27}│\n"
    "│  Port:    {port:<27}│\n"
    "└─────────────────────────────────────┘"
)
SHUTDOWN_MESSAGE = "LocalAi shutting down...\n"


def resolve_config_path() -> Path:
    """Resolve the absolute path to the LocalAi configuration file."""
    return Path(__file__).resolve().parent.parent / CONFIG_FILE_NAME


def write_error(message: str) -> None:
    """Write a startup error message to stderr."""
    sys.stderr.write(f"{message}\n")


def load_runtime_config() -> LocalAiConfig:
    """Load the application configuration from the project root."""
    return load_config(str(resolve_config_path()))


def load_runtime_config_or_exit() -> LocalAiConfig:
    """Load configuration for process startup or exit with code 1."""
    try:
        return load_runtime_config()
    except (FileNotFoundError, ValueError) as error:
        write_error(str(error))
        raise SystemExit(1) from error


def print_startup_banner(config: LocalAiConfig) -> None:
    """Render the startup banner with the resolved host and port."""
    banner = STARTUP_BANNER.format(host=config.server.host, port=config.server.port)
    rich_print(banner)


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown behavior."""
    try:
        config = load_runtime_config()
    except (FileNotFoundError, ValueError) as error:
        write_error(str(error))
        raise

    app_instance.state.config = config
    print_startup_banner(config)
    try:
        yield
    finally:
        sys.stdout.write(SHUTDOWN_MESSAGE)


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)


@app.get("/health")
async def health_endpoint(request: Request) -> dict[str, object]:
    """Return a placeholder health response for the bootstrap phase."""
    _ = request.app.state.config
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model_loaded": False,
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "queue_depth": 0,
    }


@app.get("/v1/models")
async def list_models_endpoint(request: Request) -> dict[str, object]:
    """Return the placeholder model listing for the bootstrap phase."""
    _ = request.app.state.config
    return {
        "object": "list",
        "data": [],
    }


if __name__ == "__main__":
    import uvicorn

    config = load_runtime_config_or_exit()
    uvicorn.run(
        "server.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=False,
    )
