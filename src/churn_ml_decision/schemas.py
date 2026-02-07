from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import DataValidationError

YES_NO_ALLOWED = {"Yes", "No"}
YES_NO_OR_NO_INTERNET_ALLOWED = {"Yes", "No", "No internet service"}
YES_NO_OR_NO_PHONE_ALLOWED = {"Yes", "No", "No phone service"}
INTERNET_SERVICE_ALLOWED = {"DSL", "Fiber optic", "No"}
CONTRACT_ALLOWED = {"Month-to-month", "One year", "Two year"}
PAYMENT_METHOD_ALLOWED = {
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


class CustomerInput(BaseModel):
    """Schema for a single customer prediction record."""

    model_config = ConfigDict(extra="allow")

    customerID: str | None = None
    gender: str | None = None
    SeniorCitizen: int | None = Field(default=None, ge=0, le=1)
    Partner: str | None = None
    Dependents: str | None = None
    tenure: int | None = Field(default=None, ge=0)
    PhoneService: str | None = None
    MultipleLines: str | None = None
    InternetService: str | None = None
    OnlineSecurity: str | None = None
    OnlineBackup: str | None = None
    DeviceProtection: str | None = None
    TechSupport: str | None = None
    StreamingTV: str | None = None
    StreamingMovies: str | None = None
    Contract: str | None = None
    PaperlessBilling: str | None = None
    PaymentMethod: str | None = None
    MonthlyCharges: float | None = Field(default=None, ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_categories(self) -> "CustomerInput":
        yes_no_fields = [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
        ]
        internet_dependent_fields = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for field in yes_no_fields:
            value = getattr(self, field)
            if value is not None and value not in YES_NO_ALLOWED:
                raise ValueError(f"Invalid value '{value}' for {field}.")

        if self.MultipleLines is not None and self.MultipleLines not in YES_NO_OR_NO_PHONE_ALLOWED:
            raise ValueError(f"Invalid value '{self.MultipleLines}' for MultipleLines.")

        for field in internet_dependent_fields:
            value = getattr(self, field)
            if value is not None and value not in YES_NO_OR_NO_INTERNET_ALLOWED:
                raise ValueError(f"Invalid value '{value}' for {field}.")

        if (
            self.InternetService is not None
            and self.InternetService not in INTERNET_SERVICE_ALLOWED
        ):
            raise ValueError(f"Invalid internet service: {self.InternetService}")
        if self.Contract is not None and self.Contract not in CONTRACT_ALLOWED:
            raise ValueError(f"Invalid contract: {self.Contract}")
        if self.PaymentMethod is not None and self.PaymentMethod not in PAYMENT_METHOD_ALLOWED:
            raise ValueError(f"Invalid payment method: {self.PaymentMethod}")
        return self


class PredictionOutput(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    decision: Literal["contact", "no_contact"]
    expected_value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    threshold: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_coherence(self) -> "PredictionOutput":
        if self.threshold is None:
            return self
        should_contact = self.churn_probability >= self.threshold
        if should_contact and self.decision != "contact":
            raise ValueError("Decision must be 'contact' when probability >= threshold.")
        if not should_contact and self.decision != "no_contact":
            raise ValueError("Decision must be 'no_contact' when probability < threshold.")
        return self


def validate_batch_input(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate a dataframe batch against ``CustomerInput``.

    Returns validated rows and row-level issues.
    """
    issues: list[dict[str, Any]] = []
    required = required_columns or []
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        issues.append({"row": None, "errors": [f"Missing required columns: {missing_required}"]})
        if strict:
            raise DataValidationError(f"Batch input validation failed: missing {missing_required}")
        return df.iloc[0:0].copy(), issues

    valid_indices: list[int] = []
    for idx, row in df.iterrows():
        payload = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        try:
            CustomerInput.model_validate(payload)
            if required:
                null_required = [col for col in required if pd.isna(row.get(col))]
                if null_required:
                    raise ValueError(f"Null values in required columns: {null_required}")
            valid_indices.append(idx)
        except (ValidationError, ValueError) as exc:
            error_payload: list[str]
            if isinstance(exc, ValidationError):
                error_payload = [e["msg"] for e in exc.errors()]
            else:
                error_payload = [str(exc)]
            issues.append({"row": int(idx), "errors": error_payload})

    if issues and strict:
        raise DataValidationError(f"Batch input validation failed for {len(issues)} rows.")

    return df.loc[valid_indices].copy(), issues


def validate_prediction_outputs(
    df: pd.DataFrame,
    *,
    threshold: float | None = None,
    strict: bool = True,
) -> list[str]:
    """Validate prediction outputs before writing downstream artifacts."""
    issues: list[str] = []

    if "churn_probability" not in df.columns:
        issues.append("Missing 'churn_probability' column.")
    else:
        invalid_prob = int((~df["churn_probability"].between(0, 1)).sum())
        if invalid_prob > 0:
            issues.append(f"{invalid_prob} rows have probabilities outside [0, 1].")

    if "churn_prediction" in df.columns:
        invalid_pred = int((~df["churn_prediction"].isin([0, 1])).sum())
        if invalid_pred > 0:
            issues.append(f"{invalid_pred} rows have invalid churn_prediction values.")

    if "decision" in df.columns and "churn_probability" in df.columns:
        invalid_decision = int((~df["decision"].isin(["contact", "no_contact"])).sum())
        if invalid_decision > 0:
            issues.append(f"{invalid_decision} rows have invalid decision values.")
        if threshold is not None:
            expected_decision = (
                df["churn_probability"].ge(threshold).map({True: "contact", False: "no_contact"})
            )
            mismatches = int((df["decision"] != expected_decision).sum())
            if mismatches > 0:
                issues.append(f"{mismatches} rows have inconsistent decision vs probability.")

    if strict and issues:
        raise DataValidationError(f"Prediction output validation failed: {issues}")

    return issues
