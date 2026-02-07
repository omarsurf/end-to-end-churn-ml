from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import ConfigValidationError


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NumericRangeRule(StrictBaseModel):
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "NumericRangeRule":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("numeric range 'min' must be <= 'max'.")
        return self


class PathsConfig(StrictBaseModel):
    data_raw: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_processed: str = "data/processed"
    models: str = "models"


class ModelCandidateConfig(StrictBaseModel):
    name: str
    type: Literal["logistic_regression", "xgboost", "lightgbm"]
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictBaseModel):
    name: str = "logistic_regression"
    version: int = Field(default=1, ge=1)
    selection_metric: Literal["roc_auc", "precision", "recall", "f1"] = "roc_auc"
    notes: str | None = None
    candidates: list[ModelCandidateConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_candidates(self) -> "ModelConfig":
        if not self.candidates:
            model_type = self.name
            if model_type not in {"logistic_regression", "xgboost", "lightgbm"}:
                model_type = "logistic_regression"
            self.candidates = [
                ModelCandidateConfig(name=self.name, type=model_type, enabled=True, params={})
            ]
        return self


class EvaluationConfig(StrictBaseModel):
    min_recall: float = Field(default=0.70, ge=0.0, le=1.0)
    min_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold_min: float = Field(default=0.20, ge=0.0, lt=1.0)
    threshold_max: float = Field(default=0.85, gt=0.0, le=1.0)
    threshold_step: float = Field(default=0.05, gt=0.0, le=1.0)
    optimize_for: str = Field(
        default="net_value",
        pattern=r"^(net_value|precision|f1)$",
        description="Threshold selection strategy: net_value, precision, or f1",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "EvaluationConfig":
        if self.threshold_min >= self.threshold_max:
            raise ValueError("'threshold_min' must be lower than 'threshold_max'.")
        if self.threshold_step > (self.threshold_max - self.threshold_min):
            raise ValueError("'threshold_step' is too large for the configured threshold interval.")
        if self.threshold_step > 0.25:
            warnings.warn(
                "Evaluation threshold_step is unusually high; threshold search may be too coarse.",
                stacklevel=2,
            )
        return self


class ArtifactsConfig(StrictBaseModel):
    model_file: str = "best_model.joblib"
    preprocessor_file: str = "preprocessor.joblib"
    threshold_analysis_file: str = "threshold_analysis_val.csv"
    final_results_file: str = "final_test_results.csv"
    train_medians_file: str = "train_medians.json"
    train_summary_file: str = "train_summary.json"
    data_quality_report_file: str = "data_quality_report.json"
    drift_reference_file: str = "drift_reference.json"


class SplitConfig(StrictBaseModel):
    test_size: float = Field(default=0.20, ge=0.01, le=0.49)
    val_size: float = Field(default=0.20, ge=0.01, le=0.49)
    random_state: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_split(self) -> "SplitConfig":
        total = self.test_size + self.val_size
        if total >= 0.80:
            raise ValueError("'test_size + val_size' must be < 0.80.")
        if total > 0.50:
            warnings.warn(
                "Validation+test split is > 50%; this may underfit training.",
                stacklevel=2,
            )
        return self


class FeaturesConfig(StrictBaseModel):
    numeric: list[str] = Field(
        default_factory=lambda: [
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
        ]
    )
    categorical: list[str] = Field(
        default_factory=lambda: [
            "Contract",
            "InternetService",
            "OnlineSecurity",
            "TechSupport",
            "PaymentMethod",
            "Dependents",
            "PaperlessBilling",
        ]
    )


class EngineeringConfig(StrictBaseModel):
    enabled: bool = True
    tenure_bins: list[float] = Field(default_factory=lambda: [0, 6, 12, 24, 48, 72])
    tenure_labels: list[float] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    support_cols: list[str] = Field(
        default_factory=lambda: [
            "OnlineSecurity",
            "TechSupport",
            "OnlineBackup",
            "DeviceProtection",
        ]
    )
    streaming_cols: list[str] = Field(default_factory=lambda: ["StreamingTV", "StreamingMovies"])
    overpay_group_col: str = "InternetService"

    @model_validator(mode="after")
    def validate_bins(self) -> "EngineeringConfig":
        if len(self.tenure_labels) != max(len(self.tenure_bins) - 1, 0):
            raise ValueError("'tenure_labels' length must equal len(tenure_bins) - 1.")
        return self


class BusinessConfig(StrictBaseModel):
    clv: float = Field(default=2000.0, gt=0.0)
    success_rate: float = Field(default=0.30, ge=0.0, le=1.0)
    retained_value: float | None = Field(default=None, gt=0.0)
    contact_cost: float = Field(default=50.0, gt=0.0)

    @model_validator(mode="after")
    def enrich_and_warn(self) -> "BusinessConfig":
        if self.retained_value is None:
            self.retained_value = float(self.clv) * float(self.success_rate)
        if self.success_rate < 0.05 or self.success_rate > 0.95:
            warnings.warn(
                "business.success_rate is extreme; business-impact estimates may be unstable.",
                stacklevel=2,
            )
        if self.contact_cost > self.clv:
            warnings.warn(
                "business.contact_cost is higher than CLV; campaign profitability may be unlikely.",
                stacklevel=2,
            )
        return self


class TrackingConfig(StrictBaseModel):
    enabled: bool = True
    file: str = "models/experiments.jsonl"


class RegistryConfig(StrictBaseModel):
    enabled: bool = True
    file: str = "models/registry.json"
    use_current: bool = True
    template: str = "{name}_v{version}.joblib"
    auto_promote_first_model: bool = True


class MLflowConfig(StrictBaseModel):
    enabled: bool = False
    tracking_uri: str = "mlruns"
    experiment_name: str = "churn-prediction"
    register_model: bool = False
    model_name: str = "churn-classifier"


class QualityConfig(StrictBaseModel):
    min_roc_auc: float = Field(default=0.83, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.70, ge=0.0, le=1.0)
    min_precision: float = Field(default=0.50, ge=0.0, le=1.0)


class DataValidationConfig(StrictBaseModel):
    required_columns: list[str] = Field(
        default_factory=lambda: [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "Churn",
        ]
    )
    target_column: str = "Churn"
    max_missing_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    max_duplicate_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    min_target_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    max_target_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    max_invalid_rows_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    allow_empty_dataframe: bool = False
    numeric_ranges: dict[str, NumericRangeRule] = Field(
        default_factory=lambda: {
            "tenure": NumericRangeRule(min=0),
            "MonthlyCharges": NumericRangeRule(min=0),
            "TotalCharges": NumericRangeRule(min=0),
            "SeniorCitizen": NumericRangeRule(min=0, max=1),
        }
    )


class LoggingSettings(StrictBaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: str = "logs/pipeline.log"
    logger_name: str = "churn_ml_decision"


class MonitoringConfig(StrictBaseModel):
    enabled: bool = True
    drift_p_value_threshold: float = Field(default=0.05, gt=0.0, lt=1.0)
    reference_file: str = "models/drift_reference.json"
    drift_report_file: str = "metrics/data_drift_report.json"
    metrics_file: str = "metrics/production_metrics.json"


class ChurnConfig(StrictBaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    engineering: EngineeringConfig = Field(default_factory=EngineeringConfig)
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    validation: DataValidationConfig = Field(default_factory=DataValidationConfig)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, *, env_prefix: str = "CHURN__") -> "ChurnConfig":
        return load_typed_config(path, env_prefix=env_prefix)


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigValidationError("Top-level YAML config must be a mapping/object.")
    return data


def _parse_env_value(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def _set_nested_value(payload: dict[str, Any], keys: list[str], value: Any) -> None:
    current = payload
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def collect_env_overrides(*, env_prefix: str = "CHURN__") -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        raw_path = key[len(env_prefix) :]
        path_parts = [part.lower() for part in raw_path.split("__") if part]
        if not path_parts:
            continue
        _set_nested_value(overrides, path_parts, _parse_env_value(value))
    return overrides


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "validation error")
        lines.append(f"{loc}: {msg}")
    return "Invalid configuration:\n" + "\n".join(lines)


def load_typed_config(path: str | Path, *, env_prefix: str = "CHURN__") -> ChurnConfig:
    path = Path(path)
    raw = load_yaml_config(path)
    env_overrides = collect_env_overrides(env_prefix=env_prefix)
    if env_overrides:
        raw = deep_merge(raw, env_overrides)
    try:
        return ChurnConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(format_validation_error(exc)) from exc


def load_config(path: Path) -> dict[str, Any]:
    """Compatibility loader returning plain dict for legacy code/tests."""
    return load_typed_config(path).model_dump(mode="python")


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]


def resolve_path(base: Path, value: str | Path) -> Path:
    if isinstance(value, Path):
        return value if value.is_absolute() else base / value
    path = Path(value)
    return path if path.is_absolute() else base / path


def config_hash_from_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(config: ChurnConfig | dict[str, Any]) -> str:
    payload = config if isinstance(config, dict) else config.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def log_loaded_config(logger: Any, cfg: ChurnConfig, config_path: str | Path) -> None:
    logger.info(
        "Configuration loaded",
        extra={
            "config_path": str(config_path),
            "config_hash": config_hash_from_file(config_path),
            "config": cfg.model_dump(mode="json"),
        },
    )
