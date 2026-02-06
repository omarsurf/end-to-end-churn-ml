from pathlib import Path

import pandas as pd
import pytest

from churn_ml_decision.config import load_typed_config, project_root
from churn_ml_decision.exceptions import DataValidationError
from churn_ml_decision.validators import validate_raw_data


def _sample_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "tenure": [1, 12, 24, 36],
            "MonthlyCharges": [50.0, 60.0, 70.0, 80.0],
            "TotalCharges": [50.0, 720.0, 1680.0, 2880.0],
            "Contract": ["Month-to-month", "One year", "Two year", "One year"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "OnlineSecurity": ["Yes", "No", "No", "Yes"],
            "TechSupport": ["No", "Yes", "No", "Yes"],
            "Dependents": ["No", "Yes", "No", "Yes"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )


def test_validate_raw_data_success(tmp_path: Path):
    cfg = load_typed_config(project_root() / "config" / "default.yaml")
    df = _sample_valid_df()

    report_path = tmp_path / "data_quality_report.json"
    report = validate_raw_data(df, cfg, strict=True, report_path=report_path)

    assert report["passed"] is True
    assert report["critical_issues"] == []
    assert report_path.exists()


def test_validate_raw_data_fails_on_negative_tenure():
    cfg = load_typed_config(project_root() / "config" / "default.yaml")
    df = _sample_valid_df()
    df.loc[0, "tenure"] = -1

    with pytest.raises(DataValidationError, match="Numeric range violations"):
        validate_raw_data(df, cfg, strict=True)


def test_validate_raw_data_allows_non_strict_and_reports_issues():
    cfg = load_typed_config(project_root() / "config" / "default.yaml")
    df = _sample_valid_df()
    df["MonthlyCharges"] = [None, None, None, None]

    report = validate_raw_data(df, cfg, strict=False)
    assert report["passed"] is False
    assert len(report["critical_issues"]) >= 1


@pytest.mark.parametrize("tenure,valid", [(0, True), (100, True), (-1, False), (None, False)])
def test_tenure_validation_with_raw_validator(tenure: int | None, valid: bool):
    cfg = load_typed_config(project_root() / "config" / "default.yaml")
    df = _sample_valid_df()
    df.loc[0, "tenure"] = tenure

    report = validate_raw_data(df, cfg, strict=False)
    passed = report["passed"]
    assert passed is valid
