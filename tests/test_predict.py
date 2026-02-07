import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ml_decision.config import load_typed_config
from churn_ml_decision.model_registry import ModelMetadata, ModelRegistry
from churn_ml_decision.prepare import engineer_features
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


def test_load_threshold_prefers_matching_model_id(tmp_path: Path):
    results = pd.DataFrame(
        [
            {"model_id": "model-a", "final_threshold": 0.42},
            {"model_id": "model-b", "final_threshold": 0.61},
            {"model_id": "model-a", "final_threshold": 0.55},
        ]
    )
    path = tmp_path / "results.csv"
    results.to_csv(path, index=False)

    assert load_threshold(path, model_id="model-a") == 0.55
    assert load_threshold(path, model_id="model-b") == 0.61


def test_load_threshold_model_id_fallbacks_to_latest_row(tmp_path: Path):
    results = pd.DataFrame(
        [
            {"model_id": "model-a", "final_threshold": 0.42},
            {"model_id": "model-b", "final_threshold": 0.61},
        ]
    )
    path = tmp_path / "results.csv"
    results.to_csv(path, index=False)

    assert load_threshold(path, model_id="unknown-model") == 0.61


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
            "--allow-unregistered",
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
            "--allow-unregistered",
        ],
    )

    predict_main()

    out = pd.read_csv(output_path)
    assert "churn_probability" in out.columns
    assert out["churn_probability"].between(0, 1).all()
    assert "churn_prediction" in out.columns
    assert set(out["churn_prediction"].unique()).issubset({0, 1})


def test_predict_with_engineering_enabled_uses_model_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test: do not neutral-fallback when engineered columns are expected."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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
                "  train_medians_file: train_medians.json",
                "features:",
                "  numeric:",
                "    - tenure",
                "    - MonthlyCharges",
                "    - TotalCharges",
                "    - avg_monthly_spend",
                "    - charge_tenure_ratio",
                "    - charge_deviation",
                "  categorical:",
                "    - Contract",
                "    - InternetService",
                "    - OnlineSecurity",
                "    - TechSupport",
                "    - PaymentMethod",
                "    - Dependents",
                "    - PaperlessBilling",
                "engineering:",
                "  enabled: true",
                "registry:",
                "  enabled: false",
                "monitoring:",
                "  enabled: false",
                "tracking:",
                "  enabled: false",
            ]
        )
    )
    typed_cfg = load_typed_config(cfg)

    train_df = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 36, 48, 60],
            "MonthlyCharges": [95.0, 80.0, 60.0, 55.0, 45.0, 40.0],
            "TotalCharges": [95.0, 960.0, 1440.0, 1980.0, 2160.0, 2400.0],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "One year",
                "Two year",
                "Two year",
            ],
            "InternetService": ["Fiber optic", "Fiber optic", "DSL", "DSL", "DSL", "No"],
            "OnlineSecurity": ["No", "No", "Yes", "Yes", "Yes", "No"],
            "TechSupport": ["No", "No", "Yes", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Electronic check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
                "Mailed check",
                "Mailed check",
            ],
            "Dependents": ["No", "No", "Yes", "Yes", "Yes", "No"],
            "PaperlessBilling": ["Yes", "Yes", "No", "No", "No", "Yes"],
        }
    )
    y = np.array([1, 1, 0, 0, 0, 0])

    engineered_train, medians = engineer_features(train_df, fit=True, cfg=typed_cfg)
    (models_dir / "train_medians.json").write_text(json.dumps(medians), encoding="utf-8")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), typed_cfg.features.numeric),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        )
                    ]
                ),
                typed_cfg.features.categorical,
            ),
        ],
        remainder="drop",
    )
    x_train = preprocessor.fit_transform(engineered_train)
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y)

    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "best_model.joblib")
    pd.DataFrame([{"final_threshold": 0.5}]).to_csv(
        models_dir / "final_test_results.csv",
        index=False,
    )

    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "customerID": ["ENG-1", "ENG-2"],
            "tenure": [2, 50],
            "MonthlyCharges": [99.0, 42.0],
            "TotalCharges": [198.0, 2100.0],
            "Contract": ["Month-to-month", "Two year"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
            "Dependents": ["No", "Yes"],
            "PaperlessBilling": ["Yes", "No"],
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
            "--allow-unregistered",
        ],
    )
    predict_main()

    out = pd.read_csv(output_path)
    assert "churn_probability" in out.columns
    assert (out["churn_probability"] != 0.5).any()
    assert out["decision"].isin(["contact", "no_contact"]).all()


def test_predict_non_strict_marks_failed_rows_when_feature_preparation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X = pd.DataFrame(
        {
            "MonthlyCharges": [20.0, 50.0, 80.0],
            "TotalCharges": [20.0, 500.0, 2000.0],
        }
    )
    y = np.array([0, 1, 0])
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
                "  enabled: true",
                "registry:",
                "  enabled: false",
                "monitoring:",
                "  enabled: false",
            ]
        )
    )

    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "customerID": ["P-1", "P-2"],
            "MonthlyCharges": [40.0, 90.0],
            "TotalCharges": [200.0, 3000.0],
        }
    ).to_csv(input_path, index=False)
    output_path = tmp_path / "output.csv"

    def _raise_prepare(*_args, **_kwargs):
        raise ValueError("simulated feature preparation failure")

    monkeypatch.setattr(
        "churn_ml_decision.predict._prepare_features_for_prediction",
        _raise_prepare,
    )
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
            "--allow-unregistered",
        ],
    )

    predict_main()

    out = pd.read_csv(output_path)
    assert "prediction_status" in out.columns
    assert set(out["prediction_status"]) == {"failed"}
    assert out["churn_probability"].isna().all()


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

    assert load_threshold(path) is None


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


def test_resolve_model_path_registry_disabled_fails_without_flag(tmp_path: Path):
    """Test _resolve_model_path fails when registry disabled without --allow-unregistered."""
    config_path = _make_predict_config(tmp_path, registry_enabled=False)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    with pytest.raises(SystemExit) as exc_info:
        _resolve_model_path(tmp_path, models_dir, cfg)

    assert "Registry disabled" in str(exc_info.value)


def test_resolve_model_path_registry_disabled_with_allow_unregistered(tmp_path: Path):
    """Test _resolve_model_path uses fallback when --allow-unregistered is set."""
    config_path = _make_predict_config(tmp_path, registry_enabled=False)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    path, model_id = _resolve_model_path(tmp_path, models_dir, cfg, allow_unregistered=True)

    assert path == models_dir / "best_model.joblib"
    assert model_id is None


def test_resolve_model_path_registry_enabled_allows_unregistered_override(tmp_path: Path):
    """Allow-unregistered should bypass production lookup even when registry is enabled."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # No production model registered.
    ModelRegistry(models_dir / "registry.json")

    path, model_id = _resolve_model_path(tmp_path, models_dir, cfg, allow_unregistered=True)

    assert path == models_dir / "best_model.joblib"
    assert model_id is None


def test_resolve_model_path_production_registry(tmp_path: Path):
    """Test _resolve_model_path returns production model from registry."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # Setup registry with production model
    registry = ModelRegistry(models_dir / "registry.json")
    (models_dir / "prod_model.joblib").write_text("fake")
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


def test_resolve_model_path_without_production_fails_fast(tmp_path: Path):
    """Registry-enabled scoring must require an explicit production model."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # Setup registry with non-production model
    registry = ModelRegistry(models_dir / "registry.json")
    (models_dir / "latest.joblib").write_text("fake")
    registry.register(
        str(models_dir / "latest.joblib"),
        ModelMetadata(
            model_id="latest-v1",
            model_path=str(models_dir / "latest.joblib"),
            config_hash="abc",
            status="training",
        ),
    )

    with pytest.raises(SystemExit, match="No production model found in registry"):
        _resolve_model_path(tmp_path, models_dir, cfg)


def test_resolve_model_path_missing_registry_artifact_fails_fast(tmp_path: Path):
    """Registry-enabled scoring must fail when production artifact is missing."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"
    (models_dir / "best_model.joblib").write_text("canonical")

    registry = ModelRegistry(models_dir / "registry.json")
    registry.register(
        str(models_dir / "missing.joblib"),
        ModelMetadata(
            model_id="missing-v1",
            model_path=str(models_dir / "missing.joblib"),
            config_hash="abc",
        ),
    )
    registry.promote("missing-v1")

    with pytest.raises(SystemExit, match="Production model artifact is missing"):
        _resolve_model_path(tmp_path, models_dir, cfg)


def test_resolve_model_path_empty_registry_fails_fast(tmp_path: Path):
    """Registry-enabled scoring must fail with empty registry."""
    config_path = _make_predict_config(tmp_path, registry_enabled=True)
    cfg = load_typed_config(config_path)
    models_dir = tmp_path / "models"

    # Empty registry
    ModelRegistry(models_dir / "registry.json")

    with pytest.raises(SystemExit, match="No production model found in registry"):
        _resolve_model_path(tmp_path, models_dir, cfg)


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
