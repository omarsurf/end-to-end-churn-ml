import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn_ml_decision.evaluate import (
    check_quality_gates,
    main,
    select_threshold,
    threshold_analysis,
)
from churn_ml_decision.model_registry import ModelMetadata, ModelRegistry


def test_threshold_analysis_computes_business_metrics():
    y_true = np.array([1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.7, 0.4, 0.2, 0.1])
    thresholds = np.array([0.5])

    df = threshold_analysis(y_true, y_proba, thresholds, retained_value=600.0, contact_cost=50.0)

    assert "Net_Value" in df.columns
    assert "Net_per_Flagged" in df.columns

    row = df.iloc[0]
    # At threshold 0.5: predictions are [1, 1, 0, 0, 0]
    assert row["True_Positives"] == 2
    assert row["False_Positives"] == 0
    # Net = TP * retained - flagged * cost = 2*600 - 2*50 = 1100
    assert row["Net_Value"] == 2 * 600.0 - 2 * 50.0


def test_threshold_analysis_without_business_metrics():
    y_true = np.array([1, 0])
    y_proba = np.array([0.8, 0.2])
    thresholds = np.array([0.5])

    df = threshold_analysis(y_true, y_proba, thresholds)

    assert "Net_Value" not in df.columns
    assert "Precision" in df.columns
    assert "Recall" in df.columns


def test_threshold_analysis_multiple_thresholds():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.4, 0.1])
    thresholds = np.array([0.3, 0.5, 0.7])

    df = threshold_analysis(y_true, y_proba, thresholds)

    assert len(df) == 3
    # Lower threshold = more flagged = higher recall
    assert df.iloc[0]["Recall"] >= df.iloc[2]["Recall"]


def test_threshold_analysis_high_threshold_zero_flagged():
    """When threshold is so high nobody is flagged, zero_division is handled."""
    y_true = np.array([1, 0, 0])
    y_proba = np.array([0.6, 0.3, 0.1])
    thresholds = np.array([0.99])

    df = threshold_analysis(y_true, y_proba, thresholds, retained_value=600.0, contact_cost=50.0)

    row = df.iloc[0]
    assert row["Total_Flagged"] == 0
    assert row["Precision"] == 0.0
    assert row["Recall"] == 0.0
    assert row["Net_per_Flagged"] == 0.0


def test_quality_gates_all_pass():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    failures = check_quality_gates(roc_auc=0.88, recall=0.75, precision=0.55, quality=quality)
    assert failures == []


def test_quality_gates_all_fail():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    failures = check_quality_gates(roc_auc=0.70, recall=0.50, precision=0.30, quality=quality)
    assert set(failures) == {"roc_auc", "recall", "precision"}


def test_quality_gates_partial_fail():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    failures = check_quality_gates(roc_auc=0.90, recall=0.60, precision=0.55, quality=quality)
    assert failures == ["recall"]


def test_quality_gates_boundary_values():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    # Exactly at threshold should pass (not strictly less than)
    failures = check_quality_gates(roc_auc=0.83, recall=0.70, precision=0.50, quality=quality)
    assert failures == []


# ---------- select_threshold tests ----------


def test_select_threshold_high_recall_exists():
    """When rows meet min_recall with optimize_for=precision, pick best precision."""
    df = pd.DataFrame(
        {
            "Threshold": [0.3, 0.5, 0.7],
            "Precision": [0.60, 0.75, 0.90],
            "Recall": [0.95, 0.80, 0.50],
            "F1_Score": [0.73, 0.77, 0.63],
        }
    )
    selected, reason = select_threshold(df, min_recall=0.70, optimize_for="precision")
    assert selected["Threshold"] == 0.5
    assert "Precision" in reason


def test_select_threshold_fallback_to_f1():
    """When no row meets min_recall, fallback to best F1."""
    df = pd.DataFrame(
        {
            "Threshold": [0.6, 0.7, 0.8],
            "Precision": [0.70, 0.80, 0.95],
            "Recall": [0.50, 0.40, 0.20],
            "F1_Score": [0.58, 0.53, 0.33],
        }
    )
    selected, reason = select_threshold(df, min_recall=0.70)
    assert selected["Threshold"] == 0.6
    assert "Best F1" in reason


def test_select_threshold_single_row():
    """Single row edge case."""
    df = pd.DataFrame(
        {
            "Threshold": [0.5],
            "Precision": [0.80],
            "Recall": [0.75],
            "F1_Score": [0.77],
        }
    )
    selected, reason = select_threshold(df, min_recall=0.70, optimize_for="precision")
    assert selected["Threshold"] == 0.5


def test_select_threshold_net_value_optimization():
    """Net_Value optimization selects most profitable threshold."""
    df = pd.DataFrame(
        {
            "Threshold": [0.3, 0.5, 0.7],
            "Recall": [0.92, 0.82, 0.65],
            "Precision": [0.45, 0.55, 0.70],
            "F1_Score": [0.60, 0.66, 0.67],
            "Net_Value": [168000, 155000, 130000],
        }
    )
    row, reason = select_threshold(df, min_recall=0.70, optimize_for="net_value")
    assert row["Threshold"] == 0.3  # Highest Net_Value with recall >= 0.70
    assert "Net_Value" in reason


def test_select_threshold_net_value_fallback_to_f1():
    """When Net_Value column missing, net_value mode falls back to F1."""
    df = pd.DataFrame(
        {
            "Threshold": [0.3, 0.5, 0.7],
            "Recall": [0.92, 0.82, 0.75],
            "Precision": [0.45, 0.55, 0.70],
            "F1_Score": [0.60, 0.66, 0.72],
        }
    )
    row, reason = select_threshold(df, min_recall=0.70, optimize_for="net_value")
    assert row["Threshold"] == 0.7  # Best F1 among those with recall >= 0.70
    assert "F1" in reason


# ---------- main() integration tests ----------


@pytest.fixture
def evaluate_artifacts(tmp_path):
    """Create minimal artifacts needed for evaluate main()."""
    # Create directory structure
    data_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    # Create synthetic data
    np.random.seed(42)
    n_val, n_test = 100, 100
    n_features = 5

    X_val = np.random.randn(n_val, n_features)
    y_val = np.random.randint(0, 2, n_val)
    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, 2, n_test)

    np.save(data_dir / "X_val_processed.npy", X_val)
    np.save(data_dir / "y_val.npy", y_val)
    np.save(data_dir / "X_test_processed.npy", X_test)
    np.save(data_dir / "y_test.npy", y_test)

    # Train and save a simple model
    model = LogisticRegression(random_state=42, max_iter=200)
    X_train = np.random.randn(200, n_features)
    y_train = np.random.randint(0, 2, 200)
    model.fit(X_train, y_train)

    import joblib

    joblib.dump(model, models_dir / "best_model.joblib")

    # Write train summary
    train_summary = {"model_type": "logistic_regression"}
    (models_dir / "train_summary.json").write_text(json.dumps(train_summary))

    # Create config
    config = {
        "paths": {
            "data_raw": "data/raw/test.csv",
            "data_processed": "data/processed",
            "models": "models",
        },
        "model": {"name": "logistic_regression"},
        "evaluation": {
            "min_recall": 0.50,
            "threshold_min": 0.30,
            "threshold_max": 0.70,
            "threshold_step": 0.10,
        },
        "artifacts": {
            "model_file": "best_model.joblib",
            "threshold_analysis_file": "threshold_analysis_val.csv",
            "final_results_file": "final_test_results.csv",
        },
        "business": {"clv": 2000, "success_rate": 0.30, "contact_cost": 50},
        "registry": {"enabled": False},
        "tracking": {"enabled": False},
        "mlflow": {"enabled": False},
        "quality": {"min_roc_auc": 0.40, "min_recall": 0.30, "min_precision": 0.30},
    }
    config_path = config_dir / "test.yaml"
    import yaml

    config_path.write_text(yaml.dump(config))

    return tmp_path, config_path


def test_main_runs_without_error(evaluate_artifacts, monkeypatch):
    """Integration test for evaluate main() with mocked project_root."""
    tmp_path, config_path = evaluate_artifacts

    monkeypatch.setattr("churn_ml_decision.evaluate.project_root", lambda: tmp_path)

    with patch("sys.argv", ["evaluate", "--config", str(config_path)]):
        main()

    # Verify outputs were created
    models_dir = tmp_path / "models"
    assert (models_dir / "threshold_analysis_val.csv").exists()
    assert (models_dir / "final_test_results.csv").exists()

    # Verify content
    results_df = pd.read_csv(models_dir / "final_test_results.csv")
    assert "roc_auc" in results_df.columns
    assert "final_threshold" in results_df.columns


def test_main_registry_missing_artifact_falls_back_to_canonical(evaluate_artifacts, monkeypatch):
    """Registry can reference stale paths; evaluation should still run via canonical model."""
    tmp_path, config_path = evaluate_artifacts
    monkeypatch.setattr("churn_ml_decision.evaluate.project_root", lambda: tmp_path)

    import yaml

    config = yaml.safe_load(config_path.read_text())
    registry_path = tmp_path / "models" / "registry.json"
    config["registry"] = {"enabled": True, "use_current": True, "file": str(registry_path)}
    config_path.write_text(yaml.dump(config))

    registry = ModelRegistry(registry_path)
    registry.register(
        str(tmp_path / "models" / "missing.joblib"),
        ModelMetadata(
            model_id="missing-v1",
            model_path=str(tmp_path / "models" / "missing.joblib"),
            config_hash="abc",
        ),
    )
    registry.promote("missing-v1")

    with patch("sys.argv", ["evaluate", "--config", str(config_path)]):
        main()

    results_df = pd.read_csv(tmp_path / "models" / "final_test_results.csv")
    assert "model_id" in results_df.columns
    assert pd.isna(results_df.iloc[0]["model_id"])


def test_main_with_mlflow_tracking(evaluate_artifacts, monkeypatch, tmp_path):
    """Test that MLflow tracking works when enabled."""
    artifact_path, config_path = evaluate_artifacts

    monkeypatch.setattr("churn_ml_decision.evaluate.project_root", lambda: artifact_path)

    # Update config to enable MLflow
    import yaml

    config = yaml.safe_load(config_path.read_text())
    config["mlflow"] = {
        "enabled": True,
        "tracking_uri": str(tmp_path / "mlruns"),
        "experiment_name": "test-evaluate",
        "register_model": False,
    }
    config_path.write_text(yaml.dump(config))

    with patch("sys.argv", ["evaluate", "--config", str(config_path)]):
        main()

    # MLflow directory should be created
    assert (tmp_path / "mlruns").exists()


def test_main_quality_gate_failure(evaluate_artifacts, monkeypatch):
    """Test that quality gate failures raise SystemExit."""
    tmp_path, config_path = evaluate_artifacts

    monkeypatch.setattr("churn_ml_decision.evaluate.project_root", lambda: tmp_path)

    # Update config with impossible quality gates
    import yaml

    config = yaml.safe_load(config_path.read_text())
    config["quality"] = {"min_roc_auc": 0.99, "min_recall": 0.99, "min_precision": 0.99}
    config_path.write_text(yaml.dump(config))

    with patch("sys.argv", ["evaluate", "--config", str(config_path)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert "Quality gates failed" in str(exc_info.value)
