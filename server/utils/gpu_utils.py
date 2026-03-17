"""GPU VRAM utilities for LocalAi using NVIDIA Management Library."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from server.utils.logger import get_logger

GPU_INDEX: int = 0
_UNAVAILABLE_GPU_NAME = "unavailable"
_BYTES_PER_MB = 1024 * 1024
_GPU_INITIALIZED = False

logger = get_logger(__name__)


@dataclass(slots=True)
class VramInfo:
    """Represents the current VRAM state for a GPU."""

    total_mb: int
    used_mb: int
    free_mb: int
    gpu_name: str
    gpu_index: int


def _unavailable_vram_info(gpu_index: int) -> VramInfo:
    """Return the safe default VRAM payload for unavailable GPUs.

    Args:
        gpu_index: GPU index requested by the caller.

    Returns:
        A zeroed VRAM information object.

    Raises:
        None.
    """
    return VramInfo(
        total_mb=0,
        used_mb=0,
        free_mb=0,
        gpu_name=_UNAVAILABLE_GPU_NAME,
        gpu_index=gpu_index,
    )


def _import_pynvml():
    """Import pynvml while suppressing its deprecation warning.

    Args:
        None.

    Returns:
        The imported pynvml module.

    Raises:
        None.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import pynvml

    return pynvml


def init_gpu() -> bool:
    """Initialize NVML and detect whether an NVIDIA GPU is available.

    Args:
        None.

    Returns:
        True when NVML initializes and at least one GPU is available, otherwise False.

    Raises:
        None.
    """
    global _GPU_INITIALIZED

    try:
        pynvml = _import_pynvml()
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count <= GPU_INDEX:
            pynvml.nvmlShutdown()
            _GPU_INITIALIZED = False
            logger.info("gpu unavailable", gpu_index=GPU_INDEX, device_count=device_count)
            return False

        handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")

        _GPU_INITIALIZED = True
        logger.info("gpu initialized", gpu_index=GPU_INDEX, gpu_name=gpu_name)
        return True
    except Exception as error:
        _GPU_INITIALIZED = False
        logger.info("gpu initialization unavailable", gpu_index=GPU_INDEX, error=str(error))
        return False


def get_vram_info(gpu_index: int = GPU_INDEX) -> VramInfo:
    """Read the current VRAM state for the requested GPU.

    Args:
        gpu_index: GPU index to query.

    Returns:
        A populated VramInfo object, or a safe zeroed fallback on failure.

    Raises:
        None.
    """
    if not _GPU_INITIALIZED:
        return _unavailable_vram_info(gpu_index)

    try:
        pynvml = _import_pynvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")

        return VramInfo(
            total_mb=memory_info.total // _BYTES_PER_MB,
            used_mb=memory_info.used // _BYTES_PER_MB,
            free_mb=memory_info.free // _BYTES_PER_MB,
            gpu_name=str(gpu_name),
            gpu_index=gpu_index,
        )
    except Exception:
        return _unavailable_vram_info(gpu_index)


def get_free_vram_mb(gpu_index: int = GPU_INDEX) -> int:
    """Return only the currently free VRAM for the requested GPU.

    Args:
        gpu_index: GPU index to query.

    Returns:
        The free VRAM in megabytes.

    Raises:
        None.
    """
    return get_vram_info(gpu_index).free_mb


def shutdown_gpu() -> None:
    """Shut down NVML if it was initialized.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    global _GPU_INITIALIZED

    if not _GPU_INITIALIZED:
        return

    try:
        pynvml = _import_pynvml()
        pynvml.nvmlShutdown()
    except Exception:
        pass
    finally:
        _GPU_INITIALIZED = False
