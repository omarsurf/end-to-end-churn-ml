from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from churn_ml_decision.config import load_config, project_root, resolve_path
from churn_ml_decision.exceptions import DataValidationError
from churn_ml_decision.prepare import clean_total_charges, engineer_features, main


def test_clean_total_charges_handles_blanks():
    df = pd.DataFrame({"TotalCharges": ["", "  ", "10.5", None]})
    cleaned = clean_total_charges(df)
    assert cleaned.isna().sum() == 0
    assert cleaned.iloc[2] == 10.5


def test_engineer_features_outputs_expected_columns():
    df = pd.DataFrame(
        {
            "tenure": [1, 10],
            "MonthlyCharges": [50.0, 70.0],
            "TotalCharges": [50.0, 700.0],
            "Contract": ["Month-to-month", "One year"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        }
    )
    engineered, _ = engineer_features(df, fit=True, cfg={"engineering": {}})
    expected_cols = [
        "avg_monthly_spend",
        "charge_tenure_ratio",
        "charge_deviation",
        "expected_lifetime_value",
        "overpay_indicator",
        "tenure_group",
        "num_support_services",
        "num_streaming_services",
        "is_mtm_fiber",
        "is_mtm_no_support",
        "is_echeck_mtm",
        "tenure_x_contract",
        "is_auto_pay",
        "has_internet",
    ]
    for col in expected_cols:
        assert col in engineered.columns


def test_engineer_features_with_precomputed_medians():
    """fit=False uses provided train_medians instead of computing them."""
    df = pd.DataFrame(
        {
            "tenure": [5, 20],
            "MonthlyCharges": [60.0, 80.0],
            "TotalCharges": [300.0, 1600.0],
            "Contract": ["Month-to-month", "Two year"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "TechSupport": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["Yes", "No"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["No", "Yes"],
            "PaymentMethod": ["Bank transfer (automatic)", "Electronic check"],
        }
    )
    precomputed_medians = {"Fiber optic": 65.0, "DSL": 45.0}
    engineered, returned_medians = engineer_features(
        df, train_medians=precomputed_medians, fit=False, cfg={"engineering": {}}
    )
    # Should return the same medians passed in
    assert returned_medians == precomputed_medians
    # Overpay indicator: 60 - 65 = -5, 80 - 45 = 35
    assert engineered.iloc[0]["overpay_indicator"] == pytest.approx(-5.0)
    assert engineered.iloc[1]["overpay_indicator"] == pytest.approx(35.0)


def test_clean_total_charges_valid_values():
    df = pd.DataFrame({"TotalCharges": ["100.50", "200.75", "0"]})
    cleaned = clean_total_charges(df)
    assert cleaned.iloc[0] == pytest.approx(100.50)
    assert cleaned.iloc[2] == pytest.approx(0.0)


def test_processed_paths_exist_after_prepare():
    cfg = load_config(project_root() / "config" / "default.yaml")
    processed_dir = resolve_path(project_root(), cfg["paths"]["data_processed"])
    # Skip if prepare has not been run
    if not processed_dir.exists():
        pytest.skip("data/processed does not exist (run churn-prepare first).")
    expected = [
        processed_dir / "X_train_processed.npy",
        processed_dir / "X_val_processed.npy",
        processed_dir / "X_test_processed.npy",
        processed_dir / "y_train.npy",
        processed_dir / "y_val.npy",
        processed_dir / "y_test.npy",
    ]
    for p in expected:
        assert p.exists()
        arr = np.load(p)
        assert arr.size > 0


# ---------- main() integration tests ----------


def create_synthetic_raw_data(n_samples: int = 500) -> pd.DataFrame:
    """Create synthetic Telco churn-like data."""
    np.random.seed(42)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet_services = ["DSL", "Fiber optic", "No"]
    yes_no = ["Yes", "No"]
    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:04d}" for i in range(n_samples)],
            "tenure": np.random.randint(0, 72, n_samples),
            "MonthlyCharges": np.random.uniform(20, 100, n_samples).round(2),
            "TotalCharges": np.random.uniform(0, 5000, n_samples).round(2),
            "Contract": np.random.choice(contracts, n_samples),
            "InternetService": np.random.choice(internet_services, n_samples),
            "OnlineSecurity": np.random.choice(yes_no, n_samples),
            "TechSupport": np.random.choice(yes_no, n_samples),
            "OnlineBackup": np.random.choice(yes_no, n_samples),
            "DeviceProtection": np.random.choice(yes_no, n_samples),
            "StreamingTV": np.random.choice(yes_no, n_samples),
            "StreamingMovies": np.random.choice(yes_no, n_samples),
            "PaymentMethod": np.random.choice(payment_methods, n_samples),
            "Dependents": np.random.choice(yes_no, n_samples),
            "PaperlessBilling": np.random.choice(yes_no, n_samples),
            "SeniorCitizen": np.random.randint(0, 2, n_samples),
            "Churn": np.random.choice(["Yes", "No"], n_samples, p=[0.26, 0.74]),
        }
    )
    # Some rows have blank TotalCharges (like real data)
    # Convert to string first to allow blank values
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    blank_indices = np.random.choice(n_samples, size=5, replace=False)
    df.loc[blank_indices, "TotalCharges"] = " "
    return df


@pytest.fixture
def prepare_artifacts(tmp_path):
    """Create minimal artifacts needed for prepare main()."""
    # Create directory structure
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    config_dir = tmp_path / "config"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    # Create synthetic raw data
    raw_df = create_synthetic_raw_data(500)
    raw_path = raw_dir / "telco_churn.csv"
    raw_df.to_csv(raw_path, index=False)

    # Create config
    config = {
        "paths": {
            "data_raw": "data/raw/telco_churn.csv",
            "data_processed": "data/processed",
            "models": "models",
        },
        "split": {
            "test_size": 0.20,
            "val_size": 0.20,
            "random_state": 42,
        },
        "features": {
            "numeric": [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "avg_monthly_spend",
                "charge_tenure_ratio",
                "charge_deviation",
                "expected_lifetime_value",
                "overpay_indicator",
                "tenure_group",
                "num_support_services",
                "num_streaming_services",
                "tenure_x_contract",
                "is_mtm_fiber",
                "is_mtm_no_support",
                "is_echeck_mtm",
                "is_auto_pay",
                "has_internet",
                "SeniorCitizen",
            ],
            "categorical": [
                "Contract",
                "InternetService",
                "OnlineSecurity",
                "TechSupport",
                "PaymentMethod",
                "Dependents",
                "PaperlessBilling",
            ],
        },
        "engineering": {
            "enabled": True,
            "tenure_bins": [0, 6, 12, 24, 48, 72],
            "tenure_labels": [0, 1, 2, 3, 4],
            "support_cols": ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"],
            "streaming_cols": ["StreamingTV", "StreamingMovies"],
            "overpay_group_col": "InternetService",
        },
        "artifacts": {
            "preprocessor_file": "preprocessor.joblib",
            "train_medians_file": "train_medians.json",
        },
    }
    config_path = config_dir / "test.yaml"
    config_path.write_text(yaml.dump(config))

    return tmp_path, config_path


def test_main_runs_full_pipeline(prepare_artifacts, monkeypatch):
    """Integration test for prepare main() with synthetic data."""
    tmp_path, config_path = prepare_artifacts

    monkeypatch.setattr("churn_ml_decision.prepare.project_root", lambda: tmp_path)

    with patch("sys.argv", ["prepare", "--config", str(config_path)]):
        main()

    # Verify outputs were created
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"

    expected_arrays = [
        "X_train_processed.npy",
        "X_val_processed.npy",
        "X_test_processed.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
    ]
    for fname in expected_arrays:
        fpath = processed_dir / fname
        assert fpath.exists(), f"Missing {fname}"
        arr = np.load(fpath)
        assert arr.size > 0, f"{fname} is empty"

    # Verify preprocessor was saved
    assert (models_dir / "preprocessor.joblib").exists()
    assert (models_dir / "train_medians.json").exists()
    assert (models_dir / "final_feature_names.csv").exists()


def test_main_without_feature_engineering(prepare_artifacts, monkeypatch):
    """Test prepare main() with feature engineering disabled."""
    tmp_path, config_path = prepare_artifacts

    monkeypatch.setattr("churn_ml_decision.prepare.project_root", lambda: tmp_path)

    # Update config to disable engineering and use only base features
    config = yaml.safe_load(config_path.read_text())
    config["engineering"]["enabled"] = False
    config["features"]["numeric"] = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    config_path.write_text(yaml.dump(config))

    with patch("sys.argv", ["prepare", "--config", str(config_path)]):
        main()

    # Verify outputs were created
    processed_dir = tmp_path / "data" / "processed"
    assert (processed_dir / "X_train_processed.npy").exists()

    # train_medians should NOT exist when engineering is disabled
    models_dir = tmp_path / "models"
    assert not (models_dir / "train_medians.json").exists()


def test_main_validates_feature_columns(prepare_artifacts, monkeypatch):
    """Test that main() raises when configured features are missing."""
    tmp_path, config_path = prepare_artifacts

    monkeypatch.setattr("churn_ml_decision.prepare.project_root", lambda: tmp_path)

    # Add a non-existent feature to config
    config = yaml.safe_load(config_path.read_text())
    config["features"]["numeric"].append("nonexistent_feature")
    config_path.write_text(yaml.dump(config))

    with patch("sys.argv", ["prepare", "--config", str(config_path)]):
        with pytest.raises(DataValidationError, match="Missing features"):
            main()


def test_main_split_ratios(prepare_artifacts, monkeypatch):
    """Test that train/val/test splits have correct proportions."""
    tmp_path, config_path = prepare_artifacts

    monkeypatch.setattr("churn_ml_decision.prepare.project_root", lambda: tmp_path)

    with patch("sys.argv", ["prepare", "--config", str(config_path)]):
        main()

    processed_dir = tmp_path / "data" / "processed"
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    total = len(y_train) + len(y_val) + len(y_test)
    # With 500 samples, 20% test, 20% val -> 60% train
    assert 0.55 <= len(y_train) / total <= 0.65
    assert 0.15 <= len(y_val) / total <= 0.25
    assert 0.15 <= len(y_test) / total <= 0.25
