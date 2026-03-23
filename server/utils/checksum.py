"""Provide SHA256 checksum helpers for LocalAi model artifacts."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from server.utils.logger import get_logger

CHUNK_SIZE: int = 8 * 1024 * 1024
BYTES_PER_MB: int = 1024 * 1024

logger = get_logger(__name__)


def compute_checksum(file_path: str) -> str:
    """Compute the SHA256 checksum of a file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Lowercase hexadecimal SHA256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_size_mb = round(path.stat().st_size / BYTES_PER_MB, 3)
    logger.debug("starting checksum computation", file_path=str(path), file_size_mb=file_size_mb)

    sha256 = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(CHUNK_SIZE):
            sha256.update(chunk)

    logger.debug("checksum computation complete", file_path=str(path), file_size_mb=file_size_mb)
    return sha256.hexdigest()


def write_checksum_file(file_path: str) -> str:
    """Compute and write a SHA256 sidecar file for the target file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        The computed SHA256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the checksum sidecar file cannot be written.
    """
    path = Path(file_path).expanduser().resolve()
    checksum = compute_checksum(str(path))
    checksum_path = Path(f"{path}.sha256")
    checksum_path.write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
    logger.debug("checksum sidecar written", file_path=str(path), checksum_path=str(checksum_path))
    return checksum


def verify_checksum(file_path: str) -> tuple[bool, str]:
    """Verify a file against its SHA256 sidecar file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Tuple containing verification success and a status message.

    Raises:
        None.
    """
    path = Path(file_path).expanduser().resolve()
    checksum_path = Path(f"{path}.sha256")

    if not path.is_file():
        return False, f"File not found: {path}"
    if not checksum_path.is_file():
        return False, f"No checksum file found: {checksum_path}"

    try:
        first_line = checksum_path.read_text(encoding="utf-8").splitlines()[0].strip()
    except IndexError:
        return False, f"Invalid checksum file: {checksum_path}"
    except OSError as error:
        return False, str(error)

    expected_checksum = first_line.split()[0].lower() if first_line else ""
    if not expected_checksum:
        return False, f"Invalid checksum file: {checksum_path}"

    try:
        actual_checksum = compute_checksum(str(path))
    except (FileNotFoundError, OSError) as error:
        return False, str(error)

    if actual_checksum != expected_checksum:
        return False, f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"

    return True, "OK"


async def compute_checksum_async(file_path: str) -> str:
    """Compute a SHA256 checksum without blocking the async event loop.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Lowercase hexadecimal SHA256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, compute_checksum, file_path)

