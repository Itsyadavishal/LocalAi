"""VRAM estimation and load-decision utilities for LocalAi.

This module estimates GGUF VRAM usage from file sizes and a conservative KV
cache approximation. The KV cache estimate uses a flat 128 MB per 1K context
window, which intentionally overestimates many 4B-class models. Partial GPU
layer calculations assume 32 transformer layers when the real layer count is
unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.utils.gpu_utils import get_vram_info
from server.utils.logger import get_logger

DEFAULT_MODEL_LAYERS: int = 32
BYTES_PER_MB: int = 1024 * 1024
KV_CACHE_MB_PER_1K_CTX: int = 128

logger = get_logger(__name__)


@dataclass(slots=True)
class ModelVramRequirement:
    """Estimated VRAM requirement for a model load."""

    weights_mb: int
    mmproj_mb: int
    kv_cache_mb: int
    runtime_overhead_mb: int
    total_mb: int


@dataclass(slots=True)
class LoadDecision:
    """Result of a model load feasibility decision."""

    can_load: bool
    full_gpu: bool
    n_gpu_layers: int
    vram_required_mb: int
    vram_free_mb: int
    reason: str


def _bytes_to_mb(size_bytes: int) -> int:
    """Convert a byte count to integer megabytes.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Size converted to megabytes using integer division.

    Raises:
        None.
    """
    return size_bytes // BYTES_PER_MB


def estimate_vram_requirement(
    weights_path: str,
    ctx_size: int,
    mmproj_path: str | None = None,
    runtime_overhead_mb: int = 200,
) -> ModelVramRequirement:
    """Estimate VRAM usage for a GGUF model load.

    The weights estimate uses the `.gguf` file size directly because GGUF files
    are almost entirely model weights. The KV cache estimate uses a conservative
    flat formula of 128 MB per 1K context window, which roughly matches a
    7B-class model and intentionally overestimates many 4B-class models.

    Args:
        weights_path: Path to the primary model weights `.gguf` file.
        ctx_size: Requested context size in tokens.
        mmproj_path: Optional path to a multimodal projector `.gguf` file.
        runtime_overhead_mb: Fixed runtime overhead for llama.cpp in MB.

    Returns:
        The estimated VRAM requirement broken down into major components.

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    resolved_weights_path = Path(weights_path).expanduser().resolve()
    if not resolved_weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {resolved_weights_path}")

    weights_mb = _bytes_to_mb(resolved_weights_path.stat().st_size)

    mmproj_mb = 0
    if mmproj_path is not None:
        resolved_mmproj_path = Path(mmproj_path).expanduser().resolve()
        if resolved_mmproj_path.is_file():
            mmproj_mb = _bytes_to_mb(resolved_mmproj_path.stat().st_size)

    kv_cache_mb = (ctx_size // 1024) * KV_CACHE_MB_PER_1K_CTX
    total_mb = weights_mb + mmproj_mb + kv_cache_mb + runtime_overhead_mb

    requirement = ModelVramRequirement(
        weights_mb=weights_mb,
        mmproj_mb=mmproj_mb,
        kv_cache_mb=kv_cache_mb,
        runtime_overhead_mb=runtime_overhead_mb,
        total_mb=total_mb,
    )
    logger.debug(
        "estimated vram requirement",
        weights_path=str(resolved_weights_path),
        mmproj_path=mmproj_path,
        ctx_size=ctx_size,
        weights_mb=weights_mb,
        mmproj_mb=mmproj_mb,
        kv_cache_mb=kv_cache_mb,
        runtime_overhead_mb=runtime_overhead_mb,
        total_mb=total_mb,
    )
    return requirement


def decide_load(
    requirement: ModelVramRequirement,
    safety_margin_mb: int = 300,
    total_model_layers: int = DEFAULT_MODEL_LAYERS,
) -> LoadDecision:
    """Decide whether a model can be loaded fully or partially on the GPU.

    This decision uses the current free VRAM reported by `get_vram_info()` and
    assumes 32 total transformer layers when the true architecture is unknown.
    Partial offload estimates divide the model weight size evenly across those
    layers to approximate how many GPU-offloaded layers can fit.

    Args:
        requirement: Estimated VRAM requirement for the model.
        safety_margin_mb: VRAM buffer that must remain free.
        total_model_layers: Estimated total layer count for the model.

    Returns:
        A load decision containing feasibility, GPU layer count, and reason.

    Raises:
        None.
    """
    vram = get_vram_info()
    usable_vram = vram.free_mb - safety_margin_mb
    effective_total_layers = total_model_layers if total_model_layers > 0 else DEFAULT_MODEL_LAYERS

    if usable_vram <= 0:
        decision = LoadDecision(
            can_load=False,
            full_gpu=False,
            n_gpu_layers=0,
            vram_required_mb=requirement.total_mb,
            vram_free_mb=vram.free_mb,
            reason=(
                "Insufficient VRAM after safety margin. "
                f"Required: {requirement.total_mb}MB, Available: {usable_vram}MB"
            ),
        )
        logger.info(
            "vram load decision",
            can_load=decision.can_load,
            full_gpu=decision.full_gpu,
            n_gpu_layers=decision.n_gpu_layers,
            vram_required_mb=decision.vram_required_mb,
            vram_free_mb=decision.vram_free_mb,
            reason=decision.reason,
        )
        return decision

    if requirement.total_mb <= usable_vram:
        decision = LoadDecision(
            can_load=True,
            full_gpu=True,
            n_gpu_layers=effective_total_layers,
            vram_required_mb=requirement.total_mb,
            vram_free_mb=vram.free_mb,
            reason="Model fits fully in VRAM",
        )
        logger.info(
            "vram load decision",
            can_load=decision.can_load,
            full_gpu=decision.full_gpu,
            n_gpu_layers=decision.n_gpu_layers,
            vram_required_mb=decision.vram_required_mb,
            vram_free_mb=decision.vram_free_mb,
            reason=decision.reason,
        )
        return decision

    available_for_weights_mb = usable_vram - requirement.mmproj_mb - requirement.runtime_overhead_mb
    bytes_per_layer_mb = max(1, requirement.weights_mb // effective_total_layers)
    n_gpu_layers = available_for_weights_mb // bytes_per_layer_mb

    if n_gpu_layers <= 0:
        decision = LoadDecision(
            can_load=False,
            full_gpu=False,
            n_gpu_layers=0,
            vram_required_mb=requirement.total_mb,
            vram_free_mb=vram.free_mb,
            reason=(
                "Insufficient VRAM even for partial offload. "
                f"Required: {requirement.total_mb}MB, Available: {usable_vram}MB"
            ),
        )
        logger.info(
            "vram load decision",
            can_load=decision.can_load,
            full_gpu=decision.full_gpu,
            n_gpu_layers=decision.n_gpu_layers,
            vram_required_mb=decision.vram_required_mb,
            vram_free_mb=decision.vram_free_mb,
            reason=decision.reason,
        )
        return decision

    partial_vram_required_mb = (
        requirement.mmproj_mb
        + requirement.kv_cache_mb
        + requirement.runtime_overhead_mb
        + (n_gpu_layers * bytes_per_layer_mb)
    )
    decision = LoadDecision(
        can_load=True,
        full_gpu=False,
        n_gpu_layers=n_gpu_layers,
        vram_required_mb=partial_vram_required_mb,
        vram_free_mb=vram.free_mb,
        reason=f"Partial offload: {n_gpu_layers}/{effective_total_layers} layers on GPU",
    )
    logger.info(
        "vram load decision",
        can_load=decision.can_load,
        full_gpu=decision.full_gpu,
        n_gpu_layers=decision.n_gpu_layers,
        vram_required_mb=decision.vram_required_mb,
        vram_free_mb=decision.vram_free_mb,
        reason=decision.reason,
    )
    return decision


def log_vram_snapshot() -> None:
    """Log the current VRAM snapshot for health monitoring.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    try:
        vram = get_vram_info()
        logger.info(
            "vram snapshot",
            used_mb=vram.used_mb,
            free_mb=vram.free_mb,
            total_mb=vram.total_mb,
            gpu_name=vram.gpu_name,
            gpu_index=vram.gpu_index,
        )
    except Exception:
        return
