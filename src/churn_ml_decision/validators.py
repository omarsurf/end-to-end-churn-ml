from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ChurnConfig, DataValidationConfig
from .exceptions import DataValidationError

logger = logging.getLogger(__name__)


def _normalize_config(config: ChurnConfig | dict[str, Any]) -> DataValidationConfig:
    if isinstance(config, ChurnConfig):
        return config.validation
    payload = config.get("validation", {})
    return DataValidationConfig.model_validate(payload)


def _extract_feature_groups(config: ChurnConfig | dict[str, Any]) -> tuple[list[str], list[str]]:
    if isinstance(config, ChurnConfig):
        return config.features.numeric, config.features.categorical
    features = config.get("features", {})
    numeric = list(features.get("numeric", []))
    categorical = list(features.get("categorical", []))
    return numeric, categorical


def _extract_issue_lists(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    critical = report.setdefault("critical_issues", [])
    warnings_ = report.setdefault("warnings", [])
    return critical, warnings_


def _normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        mapped = series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
        if mapped.notna().any():
            return mapped
    return pd.to_numeric(series, errors="coerce")


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    normalized = series
    if normalized.dtype == "O":
        normalized = normalized.replace(r"^\s*$", np.nan, regex=True)
    return pd.to_numeric(normalized, errors="coerce")


def validate_raw_data(
    df: pd.DataFrame,
    config: ChurnConfig | dict[str, Any],
    *,
    strict: bool = True,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate raw input data before processing.

    Returns a structured report and raises ``DataValidationError`` in strict mode on
    critical issues.
    """
    validation_cfg = _normalize_config(config)
    numeric_features, categorical_features = _extract_feature_groups(config)

    report: dict[str, Any] = {
        "passed": True,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "critical_issues": [],
        "warnings": [],
        "stats": {},
    }
    critical, warnings_ = _extract_issue_lists(report)

    if df.empty and not validation_cfg.allow_empty_dataframe:
        critical.append("Input dataframe is empty.")

    missing_cols = [col for col in validation_cfg.required_columns if col not in df.columns]
    if missing_cols:
        critical.append(f"Missing required columns: {missing_cols}")

    total_cells = max(df.shape[0] * max(df.shape[1], 1), 1)
    total_missing_ratio = float(df.isna().sum().sum() / total_cells)
    report["stats"]["missing_ratio_total"] = total_missing_ratio
    report["stats"]["missing_ratio_per_column"] = {
        col: float(value) for col, value in df.isna().mean().to_dict().items()
    }
    if total_missing_ratio > validation_cfg.max_missing_ratio:
        critical.append(
            f"Missing ratio {total_missing_ratio:.4f} exceeds "
            f"{validation_cfg.max_missing_ratio:.4f}."
        )

    duplicate_ratio = float(df.duplicated().mean()) if len(df) > 0 else 0.0
    duplicate_count = int(df.duplicated().sum()) if len(df) > 0 else 0
    report["stats"]["duplicate_count"] = duplicate_count
    report["stats"]["duplicate_ratio"] = duplicate_ratio
    if duplicate_ratio > validation_cfg.max_duplicate_ratio:
        critical.append(
            f"Duplicate ratio {duplicate_ratio:.4f} exceeds "
            f"{validation_cfg.max_duplicate_ratio:.4f}."
        )

    present_numeric = [col for col in numeric_features if col in df.columns]
    non_numeric = [col for col in present_numeric if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        coerced_failures: dict[str, int] = {}
        for col in non_numeric:
            normalized = df[col]
            if normalized.dtype == "O":
                normalized = normalized.replace(r"^\s*$", np.nan, regex=True)
            coerced = pd.to_numeric(normalized, errors="coerce")
            invalid_casts = int((coerced.isna() & normalized.notna()).sum())
            if invalid_casts > 0:
                coerced_failures[col] = invalid_casts
        if coerced_failures:
            critical.append(
                f"Numeric type violations detected (non-convertible values): {coerced_failures}"
            )

    present_categorical = [col for col in categorical_features if col in df.columns]
    numeric_like_cats = [
        col for col in present_categorical if pd.api.types.is_numeric_dtype(df[col])
    ]
    if numeric_like_cats:
        warnings_.append(
            "Categorical columns detected with numeric dtype: "
            f"{numeric_like_cats}. Verify preprocessing assumptions."
        )

    range_violations: dict[str, dict[str, int]] = {}
    for col, rule in validation_cfg.numeric_ranges.items():
        if col not in df.columns:
            continue
        values = _normalize_numeric_series(df[col])
        issues: dict[str, int] = {}
        if rule.min is not None:
            issues["below_min"] = int((values < rule.min).sum())
        if rule.max is not None:
            issues["above_max"] = int((values > rule.max).sum())
        issues = {k: v for k, v in issues.items() if v > 0}
        if issues:
            range_violations[col] = issues
    if range_violations:
        critical.append(f"Numeric range violations: {range_violations}")

    target_col = validation_cfg.target_column
    if target_col in df.columns:
        target = _normalize_target(df[target_col])
        valid_target = target.dropna()
        if not valid_target.empty:
            invalid_target = int((~valid_target.isin([0, 1])).sum())
            churn_rate = float(valid_target.mean())
            report["stats"]["target_rate"] = churn_rate
            if invalid_target > 0:
                critical.append(f"Target column '{target_col}' contains non-binary values.")
            if (
                churn_rate < validation_cfg.min_target_rate
                or churn_rate > validation_cfg.max_target_rate
            ):
                critical.append(
                    f"Target rate {churn_rate:.4f} outside "
                    f"[{validation_cfg.min_target_rate:.4f}, {validation_cfg.max_target_rate:.4f}]."
                )
        else:
            critical.append(f"Target column '{target_col}' has no valid non-null values.")

    existing_required = [col for col in validation_cfg.required_columns if col in df.columns]
    if existing_required:
        invalid_rows_ratio = float(df[existing_required].isna().any(axis=1).mean())
        report["stats"]["invalid_rows_ratio"] = invalid_rows_ratio
        if invalid_rows_ratio > validation_cfg.max_invalid_rows_ratio:
            critical.append(
                f"Invalid rows ratio {invalid_rows_ratio:.4f} exceeds "
                f"{validation_cfg.max_invalid_rows_ratio:.4f}."
            )

    report["passed"] = not critical

    if report_path is not None:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if critical:
        logger.error(
            "Raw data validation failed", extra={"issues": critical, "warnings": warnings_}
        )
        if strict:
            raise DataValidationError(f"Data validation failed: {critical}")
    elif warnings_:
        logger.warning("Raw data validation warnings", extra={"warnings": warnings_})
    else:
        logger.info(
            "Raw data validation passed", extra={"rows": len(df), "columns": len(df.columns)}
        )

    return report
