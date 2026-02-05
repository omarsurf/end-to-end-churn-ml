from pathlib import Path

from churn_ml_decision.registry import current_model_path, load_registry, update_registry


def test_load_registry_missing_file(tmp_path: Path):
    data = load_registry(tmp_path / "does_not_exist.json")
    assert data == {"runs": [], "current_model_path": None}


def test_registry_update_and_current_path(tmp_path: Path):
    registry = tmp_path / "registry.json"

    update_registry(registry, {"model_path": "models/model_v1.joblib"})
    data = load_registry(registry)

    assert data["current_model_path"] == "models/model_v1.joblib"
    assert current_model_path(registry) == "models/model_v1.joblib"


def test_registry_appends_runs(tmp_path: Path):
    registry = tmp_path / "registry.json"

    update_registry(registry, {"model_path": "models/v1.joblib"})
    update_registry(registry, {"model_path": "models/v2.joblib"})

    data = load_registry(registry)
    assert len(data["runs"]) == 2
    assert data["current_model_path"] == "models/v2.joblib"


def test_current_model_path_missing_file(tmp_path: Path):
    assert current_model_path(tmp_path / "nope.json") is None
