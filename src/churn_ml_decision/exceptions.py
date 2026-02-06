from __future__ import annotations


class ChurnMLException(Exception):
    """Base exception for churn ML pipeline."""


class ConfigValidationError(ChurnMLException):
    """Raised when configuration validation fails."""


class DataValidationError(ChurnMLException):
    """Raised when input data fails quality checks."""


class ModelNotFoundError(ChurnMLException):
    """Raised when no model can be resolved from registry or artifacts."""


class FeatureEngineeringError(ChurnMLException):
    """Raised when feature engineering fails critically."""
