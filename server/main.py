"""FastAPI application bootstrap for the LocalAi server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, Request, Response
from rich import print as rich_print

from server.api.router import register_routers
from server.config.config_loader import load_config
from server.config.schemas import LocalAiConfig
from server.core.health_monitor import HealthMonitor
from server.core.inference_engine import engine
from server.core.metrics_collector import MetricsCollector
from server.core.model_manager import ModelManager
from server.core.request_handler import RequestHandler
from server.utils.gpu_utils import get_vram_info, init_gpu, shutdown_gpu
from server.utils.logger import get_logger, setup_logging

APP_TITLE = "LocalAi"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Local AI inference infrastructure"
CONFIG_FILE_NAME = "localai.config.json"
BANNER_CONTENT_WIDTH = 33

logger = get_logger(__name__)


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


def build_banner_line(content: str) -> str:
    """Format a single banner line to the fixed output width."""
    trimmed_content = content[:BANNER_CONTENT_WIDTH]
    return f"|  {trimmed_content:<{BANNER_CONTENT_WIDTH}}|"


def print_startup_banner(config: LocalAiConfig) -> None:
    """Render the startup banner with host and GPU information."""
    vram = get_vram_info()
    host_line = build_banner_line(f"Host:  {config.server.host}:{config.server.port}")
    gpu_line = build_banner_line(f"GPU:   {vram.gpu_name}")
    vram_free_line = build_banner_line(f"VRAM:  {vram.free_mb}MB free of")
    vram_total_line = build_banner_line(f"       {vram.total_mb}MB")
    banner = "\n".join(
        [
            "+-------------------------------------+",
            build_banner_line(f"LocalAi v{APP_VERSION}"),
            host_line,
            gpu_line,
            vram_free_line,
            vram_total_line,
            "+-------------------------------------+",
        ]
    )
    rich_print(banner)


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown behavior."""
    try:
        config = load_runtime_config()
    except (FileNotFoundError, ValueError) as error:
        write_error(str(error))
        raise

    setup_logging(log_level=config.server.log_level, log_dir=config.logging.log_dir)
    app_instance.state.config = config
    logger.info(
        "logging initialized",
        log_level=config.server.log_level,
        log_dir=config.logging.log_dir,
    )

    gpu_available = init_gpu()
    app_instance.state.gpu_available = gpu_available
    model_manager = ModelManager(
        models_dir=config.models.models_dir,
        engine=engine,
        llama_server_bin=config.inference.llama_server_bin,
        llama_server_port=config.inference.llama_server_port,
    )
    app_instance.state.model_manager = model_manager
    discovered = model_manager.discover_models()
    logger.info("models discovered", count=len(discovered), models=discovered)

    request_handler = RequestHandler(
        llama_port=config.inference.llama_server_port,
        max_queue_depth=config.inference.max_queue_depth,
        default_timeout_seconds=config.inference.request_timeout_seconds,
    )
    await request_handler.start()
    app_instance.state.request_handler = request_handler

    metrics_collector = MetricsCollector(
        app_version=APP_VERSION,
        model_manager=model_manager,
        request_handler=request_handler,
        inference_engine=engine,
    )
    app_instance.state.metrics_collector = metrics_collector

    health_monitor = HealthMonitor(metrics_collector=metrics_collector)
    await health_monitor.start()
    app_instance.state.health_monitor = health_monitor

    if config.models.auto_load_on_startup and config.models.default_model:
        await model_manager.load_model(
            config.models.default_model,
            safety_margin_mb=config.vram.safety_margin_mb,
            runtime_overhead_mb=config.vram.runtime_overhead_mb,
        )

    print_startup_banner(config)
    vram = get_vram_info()
    logger.info(
        "application startup complete",
        host=config.server.host,
        port=config.server.port,
        gpu_available=gpu_available,
        gpu_name=vram.gpu_name,
        free_mb=vram.free_mb,
        total_mb=vram.total_mb,
    )
    try:
        yield
    finally:
        await app_instance.state.health_monitor.stop()
        await app_instance.state.request_handler.stop()
        if app_instance.state.model_manager.get_loaded_model_id():
            await app_instance.state.model_manager.unload_model()
        else:
            logger.info("no model loaded during shutdown")
        shutdown_gpu()
        logger.info("application shutdown complete")


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)
register_routers(app)


@app.get("/health")
async def health_endpoint(request: Request) -> dict[str, object]:
    """Return the current service health and VRAM snapshot."""
    _ = request.app.state.config
    return request.app.state.metrics_collector.collect_health_snapshot().to_dict()


@app.get("/")
async def root_endpoint() -> dict[str, object]:
    """Return a lightweight root document for browser and probe requests."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "ok",
        "docs": ["/health", "/v1/models", "/localai/status", "/localai/metrics"],
    }


@app.get("/v1")
async def openai_root_endpoint() -> dict[str, object]:
    """Return a lightweight API root document for OpenAI namespace probes."""
    return {
        "object": "api",
        "version": APP_VERSION,
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_endpoint() -> Response:
    """Return an empty favicon response to avoid noisy browser 404 probes."""
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    config = load_runtime_config_or_exit()
    # Uvicorn: always use app=app form. See RULES.md section 10.
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(
            app=app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level,
            reload=False,
        )
    )
    app.state.uvicorn_server = uvicorn_server
    uvicorn_server.run()
