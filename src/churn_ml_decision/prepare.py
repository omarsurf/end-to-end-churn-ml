from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ChurnConfig, load_typed_config, log_loaded_config, project_root, resolve_path
from .exceptions import DataValidationError
from .logging_config import setup_logging
from .monitoring import DataDriftDetector
from .validators import validate_raw_data

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data/processed artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on data validation issues.",
    )
    return parser.parse_args()


def clean_total_charges(df: pd.DataFrame) -> pd.Series:
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return tc.fillna(0)


def _series_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        logger.warning("Missing source column for feature engineering", extra={"column": col})
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _extract_engineering_cfg(cfg: ChurnConfig | dict[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, ChurnConfig):
        return cfg.engineering.model_dump(mode="python")
    return cfg.get("engineering", {})


def _safe_assign(
    df: pd.DataFrame,
    *,
    feature_name: str,
    compute_fn,
    default: float | int = 0.0,
) -> None:
    try:
        df[feature_name] = compute_fn()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Feature engineering failed; fallback applied",
            extra={"feature": feature_name, "error": str(exc)},
        )
        df[feature_name] = default


def engineer_features(
    df: pd.DataFrame,
    train_medians: dict[str, float] | None = None,
    fit: bool = False,
    cfg: ChurnConfig | dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    df_eng = df.copy()
    eng_cfg = _extract_engineering_cfg(cfg)

    tenure = _series_or_default(df_eng, "tenure", 0.0)
    monthly = _series_or_default(df_eng, "MonthlyCharges", 0.0)
    total = _series_or_default(df_eng, "TotalCharges", 0.0)
    safe_tenure = tenure.clip(lower=1)

    _safe_assign(
        df_eng,
        feature_name="avg_monthly_spend",
        compute_fn=lambda: total / safe_tenure,
        default=0.0,
    )
    _safe_assign(
        df_eng,
        feature_name="charge_tenure_ratio",
        compute_fn=lambda: monthly / safe_tenure,
        default=0.0,
    )
    _safe_assign(
        df_eng,
        feature_name="charge_deviation",
        compute_fn=lambda: monthly - df_eng["avg_monthly_spend"],
        default=0.0,
    )
    _safe_assign(
        df_eng,
        feature_name="expected_lifetime_value",
        compute_fn=lambda: monthly * tenure,
        default=0.0,
    )

    group_col = eng_cfg.get("overpay_group_col", "InternetService")
    if fit:
        if group_col not in df_eng.columns:
            logger.warning(
                "overpay_group_col missing, using global median",
                extra={"overpay_group_col": group_col},
            )
            global_median = float(monthly.median()) if len(monthly) else 0.0
            train_medians = {"__global__": global_median}
        else:
            train_medians = df_eng.groupby(group_col)["MonthlyCharges"].median().to_dict()

    if not train_medians:
        train_medians = {"__global__": float(monthly.median()) if len(monthly) else 0.0}

    def _compute_overpay() -> pd.Series:
        if group_col in df_eng.columns:
            mapped = df_eng[group_col].map(train_medians)
            mapped = mapped.fillna(train_medians.get("__global__", 0.0))
        else:
            mapped = pd.Series(train_medians.get("__global__", 0.0), index=df_eng.index)
        return monthly - mapped

    _safe_assign(
        df_eng,
        feature_name="overpay_indicator",
        compute_fn=_compute_overpay,
        default=0.0,
    )

    bins = eng_cfg.get("tenure_bins", [0, 6, 12, 24, 48, 72])
    labels = eng_cfg.get("tenure_labels", [0, 1, 2, 3, 4])

    def _compute_tenure_group() -> pd.Series:
        return pd.cut(tenure, bins=bins, labels=labels, include_lowest=True).astype(float)

    _safe_assign(
        df_eng,
        feature_name="tenure_group",
        compute_fn=_compute_tenure_group,
        default=0.0,
    )

    support_cols = eng_cfg.get(
        "support_cols",
        ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"],
    )
    streaming_cols = eng_cfg.get("streaming_cols", ["StreamingTV", "StreamingMovies"])

    _safe_assign(
        df_eng,
        feature_name="num_support_services",
        compute_fn=lambda: sum(
            (df_eng[col] == "Yes").astype(int) if col in df_eng.columns else 0
            for col in support_cols
        ),
        default=0,
    )
    _safe_assign(
        df_eng,
        feature_name="num_streaming_services",
        compute_fn=lambda: sum(
            (df_eng[col] == "Yes").astype(int) if col in df_eng.columns else 0
            for col in streaming_cols
        ),
        default=0,
    )

    _safe_assign(
        df_eng,
        feature_name="is_mtm_fiber",
        compute_fn=lambda: (
            (df_eng["Contract"] == "Month-to-month") & (df_eng["InternetService"] == "Fiber optic")
        ).astype(int),
        default=0,
    )
    _safe_assign(
        df_eng,
        feature_name="is_mtm_no_support",
        compute_fn=lambda: (
            (df_eng["Contract"] == "Month-to-month") & (df_eng["num_support_services"] == 0)
        ).astype(int),
        default=0,
    )
    _safe_assign(
        df_eng,
        feature_name="is_echeck_mtm",
        compute_fn=lambda: (
            (df_eng["PaymentMethod"] == "Electronic check")
            & (df_eng["Contract"] == "Month-to-month")
        ).astype(int),
        default=0,
    )

    contract_ord = df_eng.get("Contract", pd.Series(index=df_eng.index)).map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    contract_ord = pd.to_numeric(contract_ord, errors="coerce").fillna(0)
    _safe_assign(
        df_eng,
        feature_name="tenure_x_contract",
        compute_fn=lambda: tenure * contract_ord,
        default=0.0,
    )

    _safe_assign(
        df_eng,
        feature_name="is_auto_pay",
        compute_fn=lambda: (
            df_eng["PaymentMethod"]
            .isin(["Bank transfer (automatic)", "Credit card (automatic)"])
            .astype(int)
        ),
        default=0,
    )
    _safe_assign(
        df_eng,
        feature_name="has_internet",
        compute_fn=lambda: (df_eng["InternetService"] != "No").astype(int),
        default=0,
    )

    return df_eng, train_medians


def _save_drift_reference(
    *,
    cfg: ChurnConfig,
    x_train: pd.DataFrame,
    models_dir: Path,
    numeric_features: list[str],
) -> None:
    if not cfg.monitoring.enabled:
        return

    numeric_cols = [col for col in numeric_features if col in x_train.columns]
    if not numeric_cols:
        logger.warning("Skipping drift reference export: no numeric features available.")
        return

    detector = DataDriftDetector(p_value_threshold=cfg.monitoring.drift_p_value_threshold)
    detector.fit(x_train[numeric_cols])
    detector.save(models_dir / cfg.artifacts.drift_reference_file)
    logger.info(
        "Saved drift reference for monitoring",
        extra={
            "drift_reference_file": str(models_dir / cfg.artifacts.drift_reference_file),
            "numeric_columns": len(numeric_cols),
        },
    )


def main() -> None:
    args = parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)
    pipeline_logger = setup_logging(
        log_file=resolve_path(root, cfg.logging.file),
        level=cfg.logging.level,
        logger_name=cfg.logging.logger_name,
    )
    log_loaded_config(pipeline_logger, cfg, args.config)

    try:
        raw_path = resolve_path(root, cfg.paths.data_raw)
        processed_dir = resolve_path(root, cfg.paths.data_processed)
        models_dir = resolve_path(root, cfg.paths.models)
        processed_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Prepare stage started", extra={"raw_path": str(raw_path), "strict": args.strict}
        )
        df = pd.read_csv(raw_path)
        logger.info(
            "Raw data loaded", extra={"rows": int(len(df)), "columns": int(len(df.columns))}
        )

        quality_report_path = models_dir / cfg.artifacts.data_quality_report_file
        validation_report = validate_raw_data(
            df,
            cfg,
            strict=args.strict,
            report_path=quality_report_path,
        )
        logger.info(
            "Data quality report generated",
            extra={
                "data_quality_report": str(quality_report_path),
                "passed": validation_report["passed"],
                "critical_issues": len(validation_report["critical_issues"]),
            },
        )

        # Target encoding
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # Clean TotalCharges
        df["TotalCharges"] = clean_total_charges(df)

        # Drop ID
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        y = df["Churn"].astype(int)
        x = df.drop(columns=["Churn"])

        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=cfg.split.test_size + cfg.split.val_size,
            random_state=cfg.split.random_state,
            stratify=y,
        )
        relative_val_size = cfg.split.val_size / (cfg.split.test_size + cfg.split.val_size)
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=1 - relative_val_size,
            random_state=cfg.split.random_state,
            stratify=y_temp,
        )
        logger.info(
            "Data split completed",
            extra={
                "train_rows": int(len(x_train)),
                "val_rows": int(len(x_val)),
                "test_rows": int(len(x_test)),
            },
        )

        if cfg.engineering.enabled:
            x_train, train_medians = engineer_features(x_train, fit=True, cfg=cfg)
            x_val, _ = engineer_features(x_val, train_medians=train_medians, cfg=cfg)
            x_test, _ = engineer_features(x_test, train_medians=train_medians, cfg=cfg)
            medians_file = models_dir / cfg.artifacts.train_medians_file
            medians_file.write_text(json.dumps(train_medians), encoding="utf-8")
            logger.info("Saved train medians", extra={"path": str(medians_file)})

        numeric_features = list(cfg.features.numeric)
        categorical_features = list(cfg.features.categorical)

        missing_num = [col for col in numeric_features if col not in x_train.columns]
        missing_cat = [col for col in categorical_features if col not in x_train.columns]
        if missing_num or missing_cat:
            missing = missing_num + missing_cat
            raise DataValidationError(f"Missing features in prepared data: {missing}")

        _save_drift_reference(
            cfg=cfg,
            x_train=x_train,
            models_dir=models_dir,
            numeric_features=numeric_features,
        )

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        x_train_processed = preprocessor.fit_transform(x_train)
        x_val_processed = preprocessor.transform(x_val)
        x_test_processed = preprocessor.transform(x_test)

        np.save(processed_dir / "X_train_processed.npy", x_train_processed)
        np.save(processed_dir / "X_val_processed.npy", x_val_processed)
        np.save(processed_dir / "X_test_processed.npy", x_test_processed)
        np.save(processed_dir / "y_train.npy", y_train.to_numpy())
        np.save(processed_dir / "y_val.npy", y_val.to_numpy())
        np.save(processed_dir / "y_test.npy", y_test.to_numpy())

        feature_names = preprocessor.get_feature_names_out()
        pd.DataFrame({"feature_name": feature_names}).to_csv(
            models_dir / "final_feature_names.csv",
            index=False,
        )

        from joblib import dump

        dump(preprocessor, models_dir / cfg.artifacts.preprocessor_file)
        logger.info(
            "Prepare stage finished",
            extra={
                "processed_dir": str(processed_dir),
                "models_dir": str(models_dir),
                "features_generated": int(len(feature_names)),
            },
        )
    except Exception:
        logger.exception("Prepare stage failed")
        raise


if __name__ == "__main__":
    main()
