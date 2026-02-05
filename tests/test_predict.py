from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_ml_decision.predict import load_threshold, main as predict_main


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
        "\n".join([
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
        ])
    )

    input_path = tmp_path / "input.csv"
    pd.DataFrame({
        "customerID": ["C-1", "C-2"],
        "MonthlyCharges": [40.0, 90.0],
        "TotalCharges": [200.0, 3000.0],
    }).to_csv(input_path, index=False)
    output_path = tmp_path / "output.csv"

    monkeypatch.setattr(
        "sys.argv",
        ["churn-predict", "--config", str(cfg), "--input", str(input_path),
         "--output", str(output_path), "--threshold", "0.5"],
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
        transformers=[("num", Pipeline([("scaler", StandardScaler())]), ["MonthlyCharges", "TotalCharges"])],
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
