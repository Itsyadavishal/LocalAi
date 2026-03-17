"""Helpers for reading and validating the LocalAi configuration file."""

import json
from pathlib import Path

from pydantic import ValidationError

from server.config.schemas import LocalAiConfig


def load_config(config_path: str) -> LocalAiConfig:
    """Load, parse, and validate the LocalAi configuration file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A validated LocalAiConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the JSON is invalid or the parsed data fails validation.
    """
    absolute_path = Path(config_path).expanduser().resolve()
    if not absolute_path.is_file():
        raise FileNotFoundError(f"Config file not found: {absolute_path}")

    try:
        raw_data = absolute_path.read_text(encoding="utf-8")
        parsed_data = json.loads(raw_data)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in config file: {absolute_path} — {error}") from error

    try:
        return LocalAiConfig.model_validate(parsed_data)
    except ValidationError as error:
        raise ValueError(f"Config validation failed:\n{error}") from error
