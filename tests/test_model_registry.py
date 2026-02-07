from pathlib import Path

import pytest

from churn_ml_decision.exceptions import ModelNotFoundError
from churn_ml_decision.model_registry import ModelMetadata, ModelRegistry


def _metadata(model_id: str, model_path: str) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_path=model_path,
        config_hash="abc123",
        metrics={"auc_roc": 0.84},
        status="training",
        input_features=["tenure", "MonthlyCharges"],
        feature_importance={"tenure": 0.2},
    )


def test_register_and_get_model(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("models/model_v1.joblib", _metadata("v1", "models/model_v1.joblib"))

    model = registry.get_model("v1")
    assert model.model_id == "v1"
    assert model.model_path == "models/model_v1.joblib"


def test_promote_model_to_production(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("models/model_v1.joblib", _metadata("v1", "models/model_v1.joblib"))
    registry.register("models/model_v2.joblib", _metadata("v2", "models/model_v2.joblib"))

    promoted = registry.promote("v2")
    assert promoted.status == "production"
    assert registry.get_production_model().model_id == "v2"


def test_rollback_to_previous_version(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("models/model_v1.joblib", _metadata("v1", "models/model_v1.joblib"))
    registry.register("models/model_v2.joblib", _metadata("v2", "models/model_v2.joblib"))
    registry.promote("v1")
    registry.promote("v2")

    rolled_back = registry.rollback()
    assert rolled_back.model_id == "v1"
    assert registry.get_production_model().model_id == "v1"


def test_get_production_model_raises_when_absent(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    with pytest.raises(ModelNotFoundError):
        registry.get_production_model()


# =============================================================================
# Additional tests for improved coverage
# =============================================================================


def test_legacy_migration_with_runs_format(tmp_path: Path):
    """Test loading registry from legacy 'runs' format with current_model_path."""
    import json

    legacy_data = {
        "runs": [
            {
                "model_id": "legacy-v1",
                "model_path": "models/old_v1.joblib",
                "metrics": {"roc_auc": 0.80},
                "config_hash": "hash1",
            },
            {
                "model_id": "legacy-v2",
                "model_path": "models/old_v2.joblib",
                "metrics": {"roc_auc": 0.85},
                "config_hash": "hash2",
            },
        ],
        "current_model_path": "models/old_v2.joblib",
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(legacy_data))

    registry = ModelRegistry(registry_path)
    models = registry.list_models()

    assert len(models) == 2
    assert models[0].model_id == "legacy-v1"
    assert models[1].model_id == "legacy-v2"
    # v2 should be production (matches current_model_path)
    assert registry.get_production_model().model_id == "legacy-v2"


def test_models_format_normalizes_absolute_paths(tmp_path: Path):
    """Existing models-format registries should be path-normalized on load."""
    import json
    from datetime import datetime, timezone

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    model_file = models_dir / "model_v1.joblib"
    model_file.write_text("placeholder")
    registry_path = models_dir / "registry.json"
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "models": [
            {
                "model_id": "v1",
                "created_at": now,
                "model_path": str(model_file.resolve()),
                "config_hash": "abc123",
                "metrics": {},
                "status": "production",
                "input_features": [],
                "feature_importance": {},
                "notes": None,
            }
        ],
        "current_production_model_id": "v1",
        "previous_production_model_id": None,
        "updated_at": now,
    }
    registry_path.write_text(json.dumps(payload))

    registry = ModelRegistry(registry_path)
    model = registry.get_model("v1")

    assert model.model_path == "models/model_v1.joblib"
    saved = json.loads(registry_path.read_text())
    assert saved["models"][0]["model_path"] == "models/model_v1.joblib"


def test_legacy_migration_no_matching_production(tmp_path: Path):
    """Test legacy migration when no model matches current_model_path."""
    import json

    legacy_data = {
        "runs": [
            {
                "model_id": "legacy-v1",
                "model_path": "models/old_v1.joblib",
                "metrics": {},
            },
        ],
        "current_model_path": "models/nonexistent.joblib",
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(legacy_data))

    registry = ModelRegistry(registry_path)
    # Last model should become production
    production = registry.get_production_model()
    assert production.model_id == "legacy-v1"


def test_list_models_with_status_filter(tmp_path: Path):
    """Test filtering models by status."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("m1.joblib", _metadata("v1", "m1.joblib"))
    registry.register("m2.joblib", _metadata("v2", "m2.joblib"))
    registry.register("m3.joblib", _metadata("v3", "m3.joblib"))
    registry.promote("v2")

    production_models = registry.list_models(status="production")
    training_models = registry.list_models(status="training")

    assert len(production_models) == 1
    assert production_models[0].model_id == "v2"
    assert len(training_models) == 2  # v1 and v3 remain training


def test_get_model_by_path_found(tmp_path: Path):
    """Test retrieving model by its file path."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("models/special.joblib", _metadata("v1", "models/special.joblib"))

    model = registry.get_model_by_path("models/special.joblib")

    assert model is not None
    assert model.model_id == "v1"


def test_get_model_by_path_not_found(tmp_path: Path):
    """Test get_model_by_path returns None for missing path."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("models/v1.joblib", _metadata("v1", "models/v1.joblib"))

    model = registry.get_model_by_path("models/nonexistent.joblib")

    assert model is None


def test_get_latest_model_empty_registry(tmp_path: Path):
    """Test get_latest_model raises error on empty registry."""
    registry = ModelRegistry(tmp_path / "registry.json")

    with pytest.raises(ModelNotFoundError, match="Registry is empty"):
        registry.get_latest_model()


def test_update_status_with_metrics(tmp_path: Path):
    """Test updating model status and metrics."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("m1.joblib", _metadata("v1", "m1.joblib"))

    registry.update_status("v1", status="validation", metrics={"precision": 0.9})

    model = registry.get_model("v1")
    assert model.status == "validation"
    assert model.metrics["precision"] == 0.9
    assert model.metrics["auc_roc"] == 0.84  # Original metric preserved


def test_update_status_model_not_found(tmp_path: Path):
    """Test update_status raises error for missing model."""
    registry = ModelRegistry(tmp_path / "registry.json")

    with pytest.raises(ModelNotFoundError, match="not found"):
        registry.update_status("nonexistent", status="production")


def test_promote_error_model_not_found(tmp_path: Path):
    """Test promote raises error for missing model."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("m1.joblib", _metadata("v1", "m1.joblib"))

    with pytest.raises(ModelNotFoundError, match="not found"):
        registry.promote("nonexistent")


def test_rollback_no_candidate(tmp_path: Path):
    """Test rollback raises error when only one model exists and it's production."""
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("m1.joblib", _metadata("v1", "m1.joblib"))
    registry.promote("v1")

    with pytest.raises(ModelNotFoundError, match="No rollback candidate"):
        registry.rollback()


def test_rollback_empty_registry(tmp_path: Path):
    """Test rollback raises error on empty registry."""
    registry = ModelRegistry(tmp_path / "registry.json")

    with pytest.raises(ModelNotFoundError, match="Registry is empty"):
        registry.rollback()
