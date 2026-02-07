from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_registry import ModelMetadata, ModelRegistry

_VALID_STATUSES = {"training", "validation", "production", "deprecated"}


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": [], "current_model_path": None}
    return json.loads(path.read_text(encoding="utf-8"))


def update_registry(path: Path, entry: dict[str, Any]) -> None:
    model_path = str(entry.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("entry.model_path is required.")

    model_id = str(entry.get("model_id") or f"legacy-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}")
    raw_status = str(entry.get("status") or "training")
    status = raw_status if raw_status in _VALID_STATUSES else "training"
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    input_features = entry.get("input_features")
    if not isinstance(input_features, list):
        input_features = []
    feature_importance = entry.get("feature_importance")
    if not isinstance(feature_importance, dict):
        feature_importance = {}

    metadata = ModelMetadata(
        model_id=model_id,
        model_path=model_path,
        config_hash=str(entry.get("config_hash") or "legacy"),
        metrics=metrics,
        status=status,
        input_features=input_features,
        feature_importance=feature_importance,
        notes=entry.get("notes"),
    )
    registry = ModelRegistry(path)
    registry.register(model_path, metadata)
    # Legacy contract: latest update becomes current model path.
    registry.promote(model_id)


def current_model_path(path: Path) -> str | None:
    data = load_registry(path)
    return data.get("current_model_path")
