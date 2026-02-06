from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_ml_decision.config import load_typed_config
from churn_ml_decision.model_registry import ModelMetadata, ModelRegistry
from churn_ml_decision.predict import (
    _prepare_features_for_prediction,
    _resolve_model_path,
    load_threshold,
    main as predict_main,
)


def test_load_threshold_from_csv(tmp_path: Path):
    results = pd.DataFrame([{"final_threshold": 0.42, "roc_auc": 0.85}])
    path = tmp_path / "results.csv"
    results.to_csv(path, index=False)

    assert load_threshold(path) == 0.42


def test_load_threshold_missing_file(tmp_path: Path):
    assert load_threshold(tmp_path / "nope.csv") is None


def test_predict_preserves_customer_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    features = ["MonthlyCharges", "TotalCharges"]
    X = pd.DataFrame({"MonthlyCharges": [20.0, 50.0, 80.0], "TotalCharges": [20.0, 500.0, 2000.0]})
    y = np.array([0, 1, 0])

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([("scaler", StandardScaler())]), features)],
        remainder="drop",
    )
    Xp = preprocessor.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(Xp, y)

    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "best_model.joblib")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  models: {models_dir}",
                "artifacts:",
                "  preprocessor_file: preprocessor.joblib",
                "  model_file: best_model.joblib",
                "  final_results_file: final_test_results.csv",
                "engineering:",
                "  enabled: false",
                "registry:",
                "  enabled: false",
            ]
        )
    )

    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "customerID": ["C-1", "C-2"],
            "MonthlyCharges": [40.0, 90.0],
            "TotalCharges": [200.0, 3000.0],
        }
    ).to_csv(input_path, index=False)
    output_path = tmp_path / "output.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "churn-predict",
            "--config",
            str(cfg),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            "0.5",
        ],
    )
    predict_main()

    out = pd.read_csv(output_path)
    assert "customerID" in out.columns
    assert list(out["customerID"]) == ["C-1", "C-2"]
    assert "churn_probability" in out.columns
    assert out["churn_probability"].between(0, 1).all()


def test_predict_outputs_probabilities_in_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create simple training data
    X = pd.DataFrame(
        {
            "MonthlyCharges": [20.0, 50.0, 80.0, 30.0],
            "TotalCharges": [20.0, 500.0, 2000.0, 60.0],
        }
    )
    y = np.array([0, 0, 1, 0])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), ["MonthlyCharges", "TotalCharges"])
        ],
        remainder="drop",
    )
    Xp = preprocessor.fit_transform(X)

    model = LogisticRegression(max_iter=200)
    model.fit(Xp, y)

    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "best_model.joblib")

    # final results with threshold
    pd.DataFrame(
        [
            {
                "final_threshold": 0.5,
                "roc_auc": 0.8,
                "precision": 0.5,
                "recall": 0.7,
                "f1_score": 0.6,
            }
        ]
    ).to_csv(models_dir / "final_test_results.csv", index=False)

    # Config for predict
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  models: {models_dir}",
                "artifacts:",
                "  preprocessor_file: preprocessor.joblib",
                "  model_file: best_model.joblib",
                "  final_results_file: final_test_results.csv",
                "engineering:",
                "  enabled: false",
                "registry:",
                "  enabled: false",
            ]
        )
    )

    # Input file
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "customerID": ["A-1", "A-2"],
            "MonthlyCharges": [40.0, 90.0],
            "TotalCharges": [200.0, 3000.0],
        }
    ).to_csv(input_path, index=False)

    output_path = tmp_path / "output.csv"

    monkeypatch.setenv("PYTHONPATH", "src")
    monkeypatch.setattr(
        "sys.argv",
        [
            "churn-predict",
            "--config",
            str(cfg),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    predict_main()

    out = pd.read_csv(output_path)
    assert "churn_probability" in out.columns
    assert out["churn_probability"].between(0, 1).all()
    assert "churn_prediction" in out.columns
    assert set(out["churn_prediction"].unique()).issubset({0, 1})


# =============================================================================
# Additional tests for helper functions
# =============================================================================


def test_load_threshold_missing_column(tmp_path: Path):
    """Test load_threshold returns None when column is missing."""
    results = pd.DataFrame([{"roc_auc": 0.85, "precision": 0.7}])
    path = tmp_path / "results.csv"
    results.to_csv(path, index=False)

    assert load_threshold(path) is None


def test_load_threshold_empty_csv(tmp_path: Path):
    """Test load_threshold handles empty results gracefully."""
    path = tmp_path / "results.csv"
    path.write_text("final_threshold\n")  # Header only, no data

    # This should handle empty data or return None/raise gracefully
    try:
        result = load_threshold(path)
        # If it returns, check it handles the empty case
        assert result is None or isinstance(result, float)
    except (IndexError, KeyError):
        # Expected for truly empty CSV
        pass


def _make_predict_config(tmp_path: Path, registry_enabled: bool = False) -> Path:
    """Create minimal config for predict tests."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "paths": {"models": str(models_dir)},
        "artifacts": {
            "preprocessor_file": "preprocessor.joblib",
            "model_file": "best_model.joblib",
            "final_results_file": "final_test_results.csv",
            "train_medians_file": "train_medians.json",
        },
        "registry": {
            "enabled": registry_enabled,
            "file": str(models_dir / "registry.json"),
            "use_current": True,
        },
        "engineering": {"enabled": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def test_resolve_model_path_registry_disabled(tmp_path: Path):
    """Test _resolve_model_path uses default when registry disabled."""
    config_path = _make_predict_config(tmp_path, registry_enabled=False)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    path, model_id = _resolve_model_path(tmp_path, models_dir, cfg)

    assert path == models_dir / "best_model.joblib"
    assert model_id is None


def test_resolve_model_path_production_registry(tmp_path: Path):
    """Test _resolve_model_path returns production model from registry."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # Setup registry with production model
    registry = ModelRegistry(models_dir / "registry.json")
    registry.register(
        str(models_dir / "prod_model.joblib"),
        ModelMetadata(
            model_id="prod-v1",
            model_path=str(models_dir / "prod_model.joblib"),
            config_hash="abc",
        ),
    )
    registry.promote("prod-v1")

    path, model_id = _resolve_model_path(tmp_path, models_dir, cfg)

    assert path == models_dir / "prod_model.joblib"
    assert model_id == "prod-v1"


def test_resolve_model_path_fallback_latest(tmp_path: Path):
    """Test _resolve_model_path falls back to latest when no production."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # Setup registry with non-production model
    registry = ModelRegistry(models_dir / "registry.json")
    registry.register(
        str(models_dir / "latest.joblib"),
        ModelMetadata(
            model_id="latest-v1",
            model_path=str(models_dir / "latest.joblib"),
            config_hash="abc",
            status="training",
        ),
    )

    path, model_id = _resolve_model_path(tmp_path, models_dir, cfg)

    assert path == models_dir / "latest.joblib"
    assert model_id == "latest-v1"


def test_resolve_model_path_empty_registry_fallback(tmp_path: Path):
    """Test _resolve_model_path falls back to default with empty registry."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # Empty registry
    ModelRegistry(models_dir / "registry.json")

    path, model_id = _resolve_model_path(tmp_path, models_dir, cfg)

    assert path == models_dir / "best_model.joblib"
    assert model_id is None


def test_prepare_features_no_engineering(tmp_path: Path):
    """Test _prepare_features when engineering disabled."""
    config_path = _make_predict_config(tmp_path, registry_enabled=False)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    df = pd.DataFrame({"MonthlyCharges": [50.0], "TotalCharges": ["100.0"]})
    result = _prepare_features_for_prediction(df, cfg, models_dir)

    assert "MonthlyCharges" in result.columns
    # TotalCharges should be cleaned
    assert result["TotalCharges"].iloc[0] == 100.0


def test_prepare_features_missing_medians(tmp_path: Path):
    """Test _prepare_features raises when medians file missing with engineering enabled."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "paths": {"models": str(models_dir)},
        "artifacts": {
            "preprocessor_file": "preprocessor.joblib",
            "model_file": "best_model.joblib",
            "train_medians_file": "train_medians.json",
        },
        "engineering": {"enabled": True},
        "registry": {"enabled": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    cfg = load_typed_config(config_path)

    df = pd.DataFrame({"MonthlyCharges": [50.0]})

    with pytest.raises(FileNotFoundError, match="Train medians not found"):
        _prepare_features_for_prediction(df, cfg, models_dir)
