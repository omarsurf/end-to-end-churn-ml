from pathlib import Path

from churn_ml_decision.registry import current_model_path, load_registry, update_registry


def test_registry_update_and_current_path(tmp_path: Path):
    registry = tmp_path / "registry.json"

    update_registry(registry, {"model_path": "models/model_v1.joblib"})
    data = load_registry(registry)

    assert data["current_model_path"] == "models/model_v1.joblib"
    assert current_model_path(registry) == "models/model_v1.joblib"
