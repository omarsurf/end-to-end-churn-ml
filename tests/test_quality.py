import json
import pandas as pd
import pytest

from churn_ml_decision.config import load_config, project_root, resolve_path


def test_quality_thresholds():
    cfg = load_config(project_root() / "config" / "default.yaml")
    models_dir = resolve_path(project_root(), cfg["paths"]["models"])
    results_file = models_dir / cfg["artifacts"]["final_results_file"]

    if not results_file.exists():
        pytest.skip("final_test_results.csv not found (run churn-evaluate first).")

    results = pd.read_csv(results_file).tail(1).iloc[0]
    quality = cfg["quality"]

    assert results["roc_auc"] >= quality["min_roc_auc"]
    assert results["recall"] >= quality["min_recall"]
    assert results["precision"] >= quality["min_precision"]


def test_registry_current_model_path():
    cfg = load_config(project_root() / "config" / "default.yaml")
    registry_path = resolve_path(project_root(), cfg["registry"]["file"])
    if not registry_path.exists():
        pytest.skip("registry.json not found (run churn-train first).")
    data = json.loads(registry_path.read_text())
    assert "current_model_path" in data
