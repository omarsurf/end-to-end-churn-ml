from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import ModelNotFoundError

ModelStatus = Literal["training", "validation", "production", "deprecated"]


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_path: str
    config_hash: str
    metrics: dict[str, float] = Field(default_factory=dict)
    status: ModelStatus = "training"
    input_features: list[str] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    notes: str | None = None


class RegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelMetadata] = Field(default_factory=list)
    current_production_model_id: str | None = None
    previous_production_model_id: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRegistry:
    """JSON-backed model registry with promotion and rollback operations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _normalize_model_path(self, model_path: str | Path) -> str:
        path = Path(model_path)
        if not path.is_absolute():
            return path.as_posix()

        resolved = path.resolve()
        # Prefer repository-relative paths when model artifacts live under project root.
        project_root = self.path.resolve().parent.parent
        try:
            return resolved.relative_to(project_root).as_posix()
        except ValueError:
            return str(resolved)

    def _read_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _load(self) -> RegistryDocument:
        raw = self._read_raw()
        if not raw:
            return RegistryDocument()

        if "models" in raw:
            normalized = {
                key: raw[key]
                for key in [
                    "models",
                    "current_production_model_id",
                    "previous_production_model_id",
                    "updated_at",
                ]
                if key in raw
            }
            return RegistryDocument.model_validate(normalized)

        # Backward-compatible migration from legacy structure.
        legacy_runs = raw.get("runs", [])
        current_path = raw.get("current_model_path")
        if current_path is not None:
            current_path = self._normalize_model_path(str(current_path))
        models: list[ModelMetadata] = []
        production_id: str | None = None
        now = datetime.now(timezone.utc)

        for idx, run in enumerate(legacy_runs):
            model_id = str(run.get("model_id") or f"legacy-model-{idx + 1}")
            model_path = self._normalize_model_path(str(run.get("model_path", "")))
            status: ModelStatus = "validation"
            if current_path and model_path == current_path:
                status = "production"
                production_id = model_id
            created_at = run.get("created_at", now.isoformat())
            models.append(
                ModelMetadata(
                    model_id=model_id,
                    created_at=created_at,
                    model_path=model_path,
                    config_hash=str(run.get("config_hash", "legacy")),
                    metrics=run.get("metrics", {}),
                    status=status,
                    input_features=run.get("input_features", []),
                    feature_importance=run.get("feature_importance", {}),
                    notes=run.get("notes"),
                )
            )

        if production_id is None and models:
            production_id = models[-1].model_id
            models[-1].status = "production"

        return RegistryDocument(
            models=models,
            current_production_model_id=production_id,
            previous_production_model_id=None,
            updated_at=now,
        )

    def _serialize(self, doc: RegistryDocument) -> dict[str, Any]:
        payload = doc.model_dump(mode="json")

        production = None
        if doc.current_production_model_id:
            production = next(
                (m for m in doc.models if m.model_id == doc.current_production_model_id),
                None,
            )

        latest = doc.models[-1] if doc.models else None
        payload["current_model_path"] = (
            production.model_path if production else (latest.model_path if latest else None)
        )
        payload["runs"] = [
            {
                "model_id": model.model_id,
                "created_at": model.created_at.isoformat(),
                "model_path": model.model_path,
                "metrics": model.metrics,
                "status": model.status,
                "config_hash": model.config_hash,
                "input_features": model.input_features,
                "feature_importance": model.feature_importance,
            }
            for model in doc.models
        ]
        return payload

    def _save(self, doc: RegistryDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize(doc)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_models(self, *, status: ModelStatus | None = None) -> list[ModelMetadata]:
        doc = self._load()
        models = doc.models
        if status is not None:
            models = [m for m in models if m.status == status]
        return models

    def get_model(self, model_id: str) -> ModelMetadata:
        doc = self._load()
        for model in doc.models:
            if model.model_id == model_id:
                return model
        raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

    def get_model_by_path(self, model_path: str | Path) -> ModelMetadata | None:
        raw_model_path = str(model_path)
        normalized_model_path = self._normalize_model_path(model_path)
        doc = self._load()
        for model in doc.models:
            if model.model_path in {raw_model_path, normalized_model_path}:
                return model
        return None

    def get_latest_model(self) -> ModelMetadata:
        doc = self._load()
        if not doc.models:
            raise ModelNotFoundError("Registry is empty.")
        return doc.models[-1]

    def get_production_model(self) -> ModelMetadata:
        doc = self._load()
        if not doc.current_production_model_id:
            raise ModelNotFoundError("No production model set in registry.")
        return self.get_model(doc.current_production_model_id)

    def register(self, model_path: str | Path, metadata: ModelMetadata) -> None:
        doc = self._load()
        if any(m.model_id == metadata.model_id for m in doc.models):
            raise ValueError(f"Model id '{metadata.model_id}' already exists in registry.")
        metadata.model_path = self._normalize_model_path(model_path)
        doc.models.append(metadata)
        doc.updated_at = datetime.now(timezone.utc)
        self._save(doc)

    def update_status(
        self,
        model_id: str,
        *,
        status: ModelStatus | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        doc = self._load()
        found = False
        for idx, model in enumerate(doc.models):
            if model.model_id != model_id:
                continue
            found = True
            payload = model.model_dump(mode="python")
            if status is not None:
                payload["status"] = status
            if metrics:
                payload["metrics"] = {**payload.get("metrics", {}), **metrics}
            doc.models[idx] = ModelMetadata.model_validate(payload)
            break

        if not found:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        doc.updated_at = datetime.now(timezone.utc)
        self._save(doc)

    def promote(self, model_id: str) -> ModelMetadata:
        doc = self._load()
        target_idx = next((i for i, m in enumerate(doc.models) if m.model_id == model_id), None)
        if target_idx is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")

        previous = doc.current_production_model_id
        doc.previous_production_model_id = previous
        doc.current_production_model_id = model_id

        for i, model in enumerate(doc.models):
            payload = model.model_dump(mode="python")
            if i == target_idx:
                payload["status"] = "production"
            elif model.status == "production":
                payload["status"] = "deprecated"
            doc.models[i] = ModelMetadata.model_validate(payload)

        doc.updated_at = datetime.now(timezone.utc)
        self._save(doc)
        return doc.models[target_idx]

    def rollback(self, model_id: str | None = None) -> ModelMetadata:
        doc = self._load()
        if not doc.models:
            raise ModelNotFoundError("Registry is empty; rollback unavailable.")

        if model_id is not None:
            return self.promote(model_id)

        if doc.previous_production_model_id:
            return self.promote(doc.previous_production_model_id)

        non_production = [m for m in doc.models if m.model_id != doc.current_production_model_id]
        if not non_production:
            raise ModelNotFoundError("No rollback candidate found.")

        return self.promote(non_production[-1].model_id)
