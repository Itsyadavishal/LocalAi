"""Structured logging setup for LocalAi.

Usage example:
    from server.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("model loaded", model_id="qwen3.5-4b", vram_mb=3200)
    logger.error("inference failed", error=str(e), request_id="abc123")
    logger.warning("queue full", depth=20, max_depth=20)
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import sys
from typing import Any

import structlog

DEFAULT_LOG_FILE_NAME = "localai.log"
DEFAULT_BACKUP_COUNT = 30
DEFAULT_ROTATION_WHEN = "midnight"
DEFAULT_ROTATION_INTERVAL = 1
DEFAULT_LOG_LEVEL = "info"
LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_LOGGING_CONFIGURED = False


def _resolve_log_level(log_level: str) -> int:
    """Resolve a config log level string to a standard logging level.

    Args:
        log_level: Configured log level string.

    Returns:
        The numeric logging level.

    Raises:
        ValueError: If the provided level is unsupported.
    """
    normalized_level = log_level.strip().upper()
    if normalized_level not in LOG_LEVEL_MAP:
        raise ValueError(f"Unsupported log level: {log_level}")
    return LOG_LEVEL_MAP[normalized_level]


def _add_logger_field(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure every event dictionary includes a logger field.

    Args:
        _: Bound logger placeholder from structlog.
        __: Method name emitted by structlog.
        event_dict: Event dictionary being processed.

    Returns:
        The event dictionary with a logger field attached.

    Raises:
        None.
    """
    if "logger" in event_dict:
        return event_dict

    record = event_dict.get("_record")
    if record is not None and hasattr(record, "name"):
        event_dict["logger"] = record.name
    else:
        event_dict["logger"] = "localai"
    return event_dict


def _add_message_field(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Mirror the structlog event field into a message field.

    Args:
        _: Bound logger placeholder from structlog.
        __: Method name emitted by structlog.
        event_dict: Event dictionary being processed.

    Returns:
        The event dictionary with a message field attached.

    Raises:
        None.
    """
    if "event" in event_dict and "message" not in event_dict:
        event_dict["message"] = event_dict["event"]
    return event_dict


def _shared_processors() -> list[Any]:
    """Build the shared processor pipeline used by all handlers.

    Args:
        None.

    Returns:
        The ordered structlog processor list.

    Raises:
        None.
    """
    return [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        _add_logger_field,
        _add_message_field,
    ]


def _build_console_formatter() -> structlog.stdlib.ProcessorFormatter:
    """Create the console formatter for human-readable terminal output.

    Args:
        None.

    Returns:
        A configured ProcessorFormatter for console rendering.

    Raises:
        None.
    """
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )


def _build_file_formatter() -> structlog.stdlib.ProcessorFormatter:
    """Create the file formatter for JSON log output.

    Args:
        None.

    Returns:
        A configured ProcessorFormatter for JSON rendering.

    Raises:
        None.
    """
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )


def _configure_structlog() -> None:
    """Configure structlog to route loggers through the standard logging stack.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    structlog.configure(
        processors=_shared_processors() + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _configure_fallback_logging() -> None:
    """Configure a fallback stderr logger before startup logging is initialized.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    resolved_level = _resolve_log_level(DEFAULT_LOG_LEVEL)
    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(_build_console_formatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(console_handler)

    _configure_structlog()


def setup_logging(log_level: str, log_dir: str) -> None:
    """Configure LocalAi logging for console and rotating JSON file output.

    Args:
        log_level: Logging verbosity from config.
        log_dir: Directory where log files should be written.

    Returns:
        None.

    Raises:
        ValueError: If the requested log level is unsupported.
    """
    global _LOGGING_CONFIGURED

    resolved_level = _resolve_log_level(log_level)
    resolved_log_dir = Path(log_dir).expanduser().resolve()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = resolved_log_dir / DEFAULT_LOG_FILE_NAME

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(_build_console_formatter())

    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when=DEFAULT_ROTATION_WHEN,
        interval=DEFAULT_ROTATION_INTERVAL,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(_build_file_formatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configure_structlog()
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a module-scoped LocalAi logger.

    Args:
        name: Logger name to bind into each log entry.

    Returns:
        A bound logger that includes the logger name in each event.

    Raises:
        None.
    """
    return structlog.stdlib.get_logger(name).bind(logger=name)


_configure_fallback_logging()
