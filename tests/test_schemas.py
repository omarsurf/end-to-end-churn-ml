import pandas as pd
import pytest

from churn_ml_decision.exceptions import DataValidationError
from churn_ml_decision.schemas import (
    CustomerInput,
    validate_batch_input,
    validate_prediction_outputs,
)


def test_customer_input_validation_success():
    payload = {
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "InternetService": "DSL",
        "Contract": "One year",
        "PaymentMethod": "Mailed check",
    }
    model = CustomerInput.model_validate(payload)
    assert model.tenure == 12


def test_customer_input_validation_fails_on_invalid_category():
    with pytest.raises(ValueError):
        CustomerInput.model_validate(
            {
                "tenure": 12,
                "MonthlyCharges": 70.5,
                "TotalCharges": 846.0,
                "InternetService": "Satellite",
            }
        )


def test_customer_input_accepts_no_internet_service_values():
    payload = {
        "tenure": 24,
        "MonthlyCharges": 65.0,
        "TotalCharges": 1560.0,
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
    }
    model = CustomerInput.model_validate(payload)
    assert model.OnlineSecurity == "No internet service"


def test_customer_input_accepts_no_phone_service_multiple_lines():
    payload = {
        "tenure": 10,
        "MonthlyCharges": 30.0,
        "TotalCharges": 300.0,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
    }
    model = CustomerInput.model_validate(payload)
    assert model.MultipleLines == "No phone service"


def test_validate_batch_input_rejects_invalid_rows_non_strict():
    df = pd.DataFrame(
        [
            {"MonthlyCharges": 10, "TotalCharges": 20, "InternetService": "DSL"},
            {"MonthlyCharges": -1, "TotalCharges": 0, "InternetService": "DSL"},
            {"MonthlyCharges": 25, "TotalCharges": 40, "InternetService": "bad"},
        ]
    )
    valid_df, issues = validate_batch_input(
        df,
        required_columns=["MonthlyCharges", "TotalCharges"],
        strict=False,
    )
    assert len(valid_df) == 1
    assert len(issues) == 2


def test_validate_batch_input_strict_mode_raises():
    df = pd.DataFrame([{"MonthlyCharges": -10, "TotalCharges": 0}])
    with pytest.raises(DataValidationError):
        validate_batch_input(df, required_columns=["MonthlyCharges", "TotalCharges"], strict=True)


def test_validate_prediction_outputs_success():
    output = pd.DataFrame(
        [
            {"churn_probability": 0.8, "churn_prediction": 1, "decision": "contact"},
            {"churn_probability": 0.2, "churn_prediction": 0, "decision": "no_contact"},
        ]
    )
    issues = validate_prediction_outputs(output, threshold=0.5, strict=True)
    assert issues == []


def test_validate_prediction_outputs_fails_probability_range():
    output = pd.DataFrame(
        [{"churn_probability": 1.2, "churn_prediction": 1, "decision": "contact"}]
    )
    with pytest.raises(DataValidationError):
        validate_prediction_outputs(output, strict=True)


def test_validate_prediction_outputs_without_threshold_skips_decision_consistency():
    output = pd.DataFrame(
        [
            {"churn_probability": 0.8, "churn_prediction": 1, "decision": "no_contact"},
            {"churn_probability": 0.2, "churn_prediction": 0, "decision": "contact"},
        ]
    )
    issues = validate_prediction_outputs(output, threshold=None, strict=False)
    assert issues == []
