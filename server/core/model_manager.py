"""Model discovery and load lifecycle management for LocalAi."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import ValidationError

from server.config.schemas import ModelConfig
from server.core.inference_engine import EngineConfig, InferenceEngine
from server.core.vram_manager import LoadDecision, decide_load, estimate_vram_requirement
from server.utils.gpu_utils import get_vram_info
from server.utils.logger import get_logger

MODEL_CONFIG_FILENAME: str = "model.config.json"
WEIGHTS_SUBDIR: str = "weights"
VISION_SUBDIR: str = "vision"


@dataclass(slots=True)
class InstalledModel:
    """Resolved metadata for an installed model on disk."""

    model_id: str
    config: ModelConfig
    weights_path: str
    mmproj_path: str | None
    config_path: str


class ModelManager:
    """Track installed models and coordinate engine load and unload operations."""

    def __init__(
        self,
        models_dir: str,
        engine: InferenceEngine,
        llama_server_bin: str,
        llama_server_port: int,
    ) -> None:
        """Initialize the model manager with resolved runtime paths.

        Args:
            models_dir: Root directory containing per-model folders.
            engine: Shared inference engine instance used for loading models.
            llama_server_bin: Path to the llama-server executable.
            llama_server_port: Port used by the llama-server subprocess.

        Returns:
            None.

        Raises:
            None.
        """
        self._project_root = Path(__file__).resolve().parents[2]
        models_root = Path(models_dir)
        self._models_dir = models_root if models_root.is_absolute() else self._project_root / models_root
        self._models_dir = self._models_dir.resolve()
        self._engine = engine
        self._installed: dict[str, InstalledModel] = {}
        self._loaded_model_id: str | None = None
        self._logger = get_logger(__name__)

        llama_bin_path = Path(llama_server_bin)
        self._llama_bin = (
            llama_bin_path if llama_bin_path.is_absolute() else self._project_root / llama_bin_path
        ).resolve()
        self._llama_port = llama_server_port

    def discover_models(self) -> list[str]:
        """Scan the models directory for valid model installations.

        Args:
            None.

        Returns:
            A list of discovered model identifiers.

        Raises:
            None.
        """
        discovered: dict[str, InstalledModel] = {}
        if not self._models_dir.is_dir():
            self._installed = discovered
            self._logger.warning("models directory not found", models_dir=str(self._models_dir))
            return []

        for model_dir in sorted(path for path in self._models_dir.iterdir() if path.is_dir()):
            config_path = model_dir / MODEL_CONFIG_FILENAME
            if not config_path.is_file():
                continue

            try:
                raw_config = json.loads(config_path.read_text(encoding="utf-8"))
                model_config = ModelConfig.model_validate(raw_config)
            except (OSError, json.JSONDecodeError, ValidationError, ValueError) as error:
                self._logger.warning(
                    "skipping model with invalid config",
                    model_dir=str(model_dir),
                    config_path=str(config_path),
                    error=str(error),
                )
                continue

            if model_config.model_id != model_dir.name:
                self._logger.warning(
                    "skipping model with mismatched model_id",
                    model_dir=str(model_dir),
                    model_id=model_config.model_id,
                )
                continue

            weights_path = model_dir / WEIGHTS_SUBDIR / model_config.gguf_filename
            if not weights_path.is_file():
                self._logger.warning(
                    "skipping model with missing weights",
                    model_id=model_config.model_id,
                    weights_path=str(weights_path),
                )
                continue

            mmproj_path: Path | None = None
            if model_config.mmproj_filename is not None:
                mmproj_path = model_dir / VISION_SUBDIR / model_config.mmproj_filename
                if not mmproj_path.is_file():
                    self._logger.warning(
                        "skipping model with missing mmproj",
                        model_id=model_config.model_id,
                        mmproj_path=str(mmproj_path),
                    )
                    continue

            installed_model = InstalledModel(
                model_id=model_config.model_id,
                config=model_config,
                weights_path=str(weights_path.resolve()),
                mmproj_path=str(mmproj_path.resolve()) if mmproj_path is not None else None,
                config_path=str(config_path.resolve()),
            )
            discovered[installed_model.model_id] = installed_model
            self._logger.info(
                "discovered model",
                model_id=installed_model.model_id,
                config_path=installed_model.config_path,
                weights_path=installed_model.weights_path,
                mmproj_path=installed_model.mmproj_path,
            )

        self._installed = discovered
        if self._loaded_model_id is not None and self._loaded_model_id not in self._installed:
            self._loaded_model_id = None
        return list(self._installed.keys())

    def list_models(self) -> list[InstalledModel]:
        """Return all installed models.

        Args:
            None.

        Returns:
            Installed models as a list.

        Raises:
            None.
        """
        return list(self._installed.values())

    def get_model(self, model_id: str) -> InstalledModel | None:
        """Return a discovered model by identifier.

        Args:
            model_id: Requested model identifier.

        Returns:
            The installed model, or None if not found.

        Raises:
            None.
        """
        return self._installed.get(model_id)

    def get_loaded_model_id(self) -> str | None:
        """Return the currently loaded model identifier.

        Args:
            None.

        Returns:
            The loaded model identifier, or None.

        Raises:
            None.
        """
        return self._loaded_model_id

    def resolve_model_id(self, requested: str) -> str | None:
        """Resolve a requested model name using exact, prefix, then substring matching.

        Matching rules (in order):
        1. Exact match.
        2. Prefix match.
        3. Substring match.

        When multiple candidates exist, prefer the currently loaded model if it is
        among the candidates. Otherwise return the shortest candidate.

        Args:
            requested: Requested model identifier fragment.

        Returns:
            The resolved model identifier, or None when no match exists.

        Raises:
            None.
        """
        normalized_requested = requested.strip().lower()
        if not normalized_requested:
            return None

        for model_id in self._installed:
            if model_id.lower() == normalized_requested:
                return model_id

        prefix_matches = [
            model_id for model_id in self._installed if model_id.lower().startswith(normalized_requested)
        ]
        substring_matches = [] if prefix_matches else [
            model_id for model_id in self._installed if normalized_requested in model_id.lower()
        ]
        candidates = prefix_matches or substring_matches
        if not candidates:
            return None

        if self._loaded_model_id in candidates:
            return self._loaded_model_id

        return min(candidates, key=len)


    async def load_model(
        self,
        model_id: str,
        safety_margin_mb: int = 300,
        runtime_overhead_mb: int = 200,
    ) -> LoadDecision:
        """Load a discovered model into the shared inference engine.

        Args:
            model_id: Requested model identifier or fuzzy match fragment.
            safety_margin_mb: Free VRAM margin that must remain available.
            runtime_overhead_mb: Fixed llama runtime overhead used in VRAM estimation.

        Returns:
            The VRAM load decision used for the engine start.

        Raises:
            ValueError: If the model cannot be resolved to an installed model.
            RuntimeError: If the model cannot fit or the engine start fails.
        """
        resolved_id = self.resolve_model_id(model_id)
        if resolved_id is None:
            installed_models = sorted(self._installed.keys())
            raise ValueError(f"Model not found: {model_id}. Installed models: {installed_models}")

        installed_model = self._installed[resolved_id]
        if self._loaded_model_id == resolved_id:
            self._logger.info("model already loaded", model_id=resolved_id)
            return LoadDecision(
                can_load=True,
                full_gpu=True,
                n_gpu_layers=installed_model.config.gpu_layers,
                vram_required_mb=0,
                vram_free_mb=get_vram_info().free_mb,
                reason="Already loaded",
            )

        if self._loaded_model_id is not None:
            await self.unload_model()

        requirement = estimate_vram_requirement(
            weights_path=installed_model.weights_path,
            ctx_size=installed_model.config.ctx_size,
            mmproj_path=installed_model.mmproj_path,
            runtime_overhead_mb=runtime_overhead_mb,
        )
        decision = decide_load(
            requirement=requirement,
            safety_margin_mb=safety_margin_mb,
            total_model_layers=installed_model.config.gpu_layers,
        )
        if not decision.can_load:
            raise RuntimeError(decision.reason)

        engine_config = EngineConfig(
            model_path=installed_model.weights_path,
            mmproj_path=installed_model.mmproj_path,
            ctx_size=installed_model.config.ctx_size,
            n_gpu_layers=decision.n_gpu_layers,
            llama_bin=str(self._llama_bin),
            llama_port=self._llama_port,
        )
        await self._engine.start(engine_config)
        self._loaded_model_id = resolved_id
        self._logger.info(
            "model loaded",
            model_id=resolved_id,
            n_gpu_layers=decision.n_gpu_layers,
            vram_required_mb=decision.vram_required_mb,
        )
        return decision

    async def unload_model(self) -> None:
        """Unload the currently running model from the inference engine.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if self._loaded_model_id is None:
            self._logger.warning("unload requested with no model loaded")
            return

        model_id = self._loaded_model_id
        self._logger.info("unloading model", model_id=model_id)
        await self._engine.stop()
        self._loaded_model_id = None
        self._logger.info("model unloaded", model_id=model_id)
