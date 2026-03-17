"""Pydantic schemas for validating LocalAi configuration files."""

from typing import ClassVar, Self

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_LOG_LEVELS: tuple[str, ...] = ("debug", "info", "warning", "error")
ALLOWED_ROTATIONS: tuple[str, ...] = ("daily", "weekly", "hourly")


class ServerConfig(BaseModel):
    """Runtime server settings."""

    host: str = Field(default="127.0.0.1", description="Host interface for the FastAPI server.")
    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="TCP port used by the FastAPI server.",
    )
    log_level: str = Field(default="info", description="Uvicorn log level for the server.")

    _allowed_log_levels: ClassVar[tuple[str, ...]] = ALLOWED_LOG_LEVELS

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate the configured server log level."""
        if value not in cls._allowed_log_levels:
            allowed_levels = ", ".join(cls._allowed_log_levels)
            raise ValueError(f"log_level must be one of: {allowed_levels}")
        return value


class InferenceConfig(BaseModel):
    """Inference subsystem settings."""

    llama_server_port: int = Field(
        default=8081,
        ge=1024,
        le=65535,
        description="Reserved port for the llama.cpp server process.",
    )
    llama_server_bin: str = Field(
        default="bin/llama-server.exe",
        description="Relative path to the llama.cpp server executable.",
    )
    request_timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout in seconds for inference requests.",
    )
    max_queue_depth: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of queued inference requests.",
    )


class VramConfig(BaseModel):
    """VRAM safety settings."""

    safety_margin_mb: int = Field(
        default=300,
        ge=100,
        description="VRAM buffer reserved to avoid over-allocation.",
    )
    runtime_overhead_mb: int = Field(
        default=200,
        ge=100,
        description="Estimated VRAM overhead consumed by the runtime.",
    )


class ModelsConfig(BaseModel):
    """Model directory and startup settings."""

    models_dir: str = Field(default="models", description="Relative directory containing model assets.")
    auto_load_on_startup: bool = Field(
        default=False,
        description="Whether the default model should be auto-loaded during startup.",
    )
    default_model: str = Field(default="", description="Identifier of the default model to load.")


class LoggingConfig(BaseModel):
    """Log storage settings."""

    log_dir: str = Field(default="logs", description="Relative directory used for application logs.")
    rotation: str = Field(default="daily", description="Log rotation schedule.")
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retain rotated log files.",
    )

    _allowed_rotations: ClassVar[tuple[str, ...]] = ALLOWED_ROTATIONS

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, value: str) -> str:
        """Validate the configured log rotation cadence."""
        if value not in cls._allowed_rotations:
            allowed_rotations = ", ".join(cls._allowed_rotations)
            raise ValueError(f"rotation must be one of: {allowed_rotations}")
        return value


class LocalAiConfig(BaseModel):
    """Top-level LocalAi configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig, description="API server configuration.")
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig,
        description="Inference subsystem configuration.",
    )
    vram: VramConfig = Field(default_factory=VramConfig, description="VRAM management configuration.")
    models: ModelsConfig = Field(default_factory=ModelsConfig, description="Model storage configuration.")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration.")

    @model_validator(mode="after")
    def validate_distinct_ports(self) -> Self:
        """Ensure the API server and llama.cpp ports do not conflict."""
        if self.server.port == self.inference.llama_server_port:
            raise ValueError("server.port and inference.llama_server_port must be different")
        return self
