import numpy as np
import pandas as pd

import pytest

from churn_ml_decision.config import load_config, project_root, resolve_path
from churn_ml_decision.prepare import clean_total_charges, engineer_features


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
