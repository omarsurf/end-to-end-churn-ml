from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import ChurnConfig, resolve_path


@dataclass
class StageResult:
    stage: str
    success: bool
    detail: str


class PipelineOrchestrator:
    """Minimal orchestrator for sequential stage execution."""

    def __init__(self, config: ChurnConfig):
        self.config = config

    def health_snapshot(self, root: Path) -> dict[str, bool]:
        models_dir = resolve_path(root, self.config.paths.models)
        return {
            "preprocessor_exists": (models_dir / self.config.artifacts.preprocessor_file).exists(),
            "model_exists": (models_dir / self.config.artifacts.model_file).exists(),
            "results_exists": (models_dir / self.config.artifacts.final_results_file).exists(),
        }

    def run_stage(self, stage_name: str, fn: Callable[[], None]) -> StageResult:
        try:
            fn()
            return StageResult(stage=stage_name, success=True, detail="ok")
        except Exception as exc:  # pragma: no cover - pass-through helper
            return StageResult(stage=stage_name, success=False, detail=str(exc))
