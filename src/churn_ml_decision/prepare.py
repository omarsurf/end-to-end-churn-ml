from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_config, project_root, resolve_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data/processed artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def clean_total_charges(df: pd.DataFrame) -> pd.Series:
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fill missing with 0 (customers with zero tenure)
    return tc.fillna(0)


def engineer_features(
    df: pd.DataFrame,
    train_medians: dict[str, float] | None = None,
    fit: bool = False,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    df_eng = df.copy()

    # Safe tenure (avoid division by zero)
    safe_tenure = df_eng["tenure"].clip(lower=1)

    # Ratios
    df_eng["avg_monthly_spend"] = df_eng["TotalCharges"] / safe_tenure
    df_eng["charge_tenure_ratio"] = df_eng["MonthlyCharges"] / safe_tenure
    df_eng["charge_deviation"] = df_eng["MonthlyCharges"] - df_eng["avg_monthly_spend"]
    df_eng["expected_lifetime_value"] = df_eng["MonthlyCharges"] * df_eng["tenure"]

    # Overpay indicator (vs peer median)
    group_col = (cfg or {}).get("engineering", {}).get("overpay_group_col", "InternetService")
    if fit:
        train_medians = df_eng.groupby(group_col)["MonthlyCharges"].median().to_dict()
    df_eng["overpay_indicator"] = df_eng["MonthlyCharges"] - df_eng[group_col].map(train_medians)

    # Tenure binning
    bins = (cfg or {}).get("engineering", {}).get("tenure_bins", [0, 6, 12, 24, 48, 72])
    labels = (cfg or {}).get("engineering", {}).get("tenure_labels", [0, 1, 2, 3, 4])
    df_eng["tenure_group"] = pd.cut(
        df_eng["tenure"], bins=bins, labels=labels, include_lowest=True
    ).astype(float)

    # Service counts
    support_cols = (cfg or {}).get("engineering", {}).get(
        "support_cols", ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]
    )
    streaming_cols = (cfg or {}).get("engineering", {}).get(
        "streaming_cols", ["StreamingTV", "StreamingMovies"]
    )

    df_eng["num_support_services"] = sum(
        (df_eng[col] == "Yes").astype(int) if col in df_eng.columns else 0
        for col in support_cols
    )
    df_eng["num_streaming_services"] = sum(
        (df_eng[col] == "Yes").astype(int) if col in df_eng.columns else 0
        for col in streaming_cols
    )

    # Interactions
    df_eng["is_mtm_fiber"] = (
        (df_eng["Contract"] == "Month-to-month") & (df_eng["InternetService"] == "Fiber optic")
    ).astype(int)
    df_eng["is_mtm_no_support"] = (
        (df_eng["Contract"] == "Month-to-month") & (df_eng["num_support_services"] == 0)
    ).astype(int)
    df_eng["is_echeck_mtm"] = (
        (df_eng["PaymentMethod"] == "Electronic check")
        & (df_eng["Contract"] == "Month-to-month")
    ).astype(int)

    contract_ord = df_eng["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    df_eng["tenure_x_contract"] = df_eng["tenure"] * contract_ord

    # Binary simplifications
    df_eng["is_auto_pay"] = df_eng["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    df_eng["has_internet"] = (df_eng["InternetService"] != "No").astype(int)

    return df_eng, train_medians


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    raw_path = resolve_path(root, cfg["paths"]["data_raw"])
    processed_dir = resolve_path(root, cfg["paths"]["data_processed"])
    models_dir = resolve_path(root, cfg["paths"]["models"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)

    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Clean TotalCharges
    df["TotalCharges"] = clean_total_charges(df)

    # Drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    target = "Churn"
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Train/val/test split
    split_cfg = cfg["split"]
    x_train, x_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"] + split_cfg["val_size"],
        random_state=split_cfg["random_state"],
        stratify=y,
    )
    relative_val_size = split_cfg["val_size"] / (split_cfg["test_size"] + split_cfg["val_size"])
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1 - relative_val_size,
        random_state=split_cfg["random_state"],
        stratify=y_temp,
    )

    # Feature engineering
    if cfg.get("engineering", {}).get("enabled", False):
        x_train, train_medians = engineer_features(x_train, fit=True, cfg=cfg)
        x_val, _ = engineer_features(x_val, train_medians=train_medians, cfg=cfg)
        x_test, _ = engineer_features(x_test, train_medians=train_medians, cfg=cfg)
        train_medians_file = cfg["artifacts"].get("train_medians_file", "train_medians.json")
        Path(models_dir / train_medians_file).write_text(pd.Series(train_medians).to_json())
        logger.info("Saved train medians to %s", models_dir / train_medians_file)

    # Feature groups (final)
    numeric_features = cfg["features"]["numeric"]
    categorical_features = cfg["features"]["categorical"]

    # Validate features exist
    missing_num = [c for c in numeric_features if c not in x_train.columns]
    missing_cat = [c for c in categorical_features if c not in x_train.columns]
    if missing_num or missing_cat:
        missing = missing_num + missing_cat
        raise ValueError(f"Missing features in prepared data: {missing}")

    # Preprocess
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
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

    # Save arrays
    np.save(processed_dir / "X_train_processed.npy", x_train_processed)
    np.save(processed_dir / "X_val_processed.npy", x_val_processed)
    np.save(processed_dir / "X_test_processed.npy", x_test_processed)
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_val.npy", y_val.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())

    # Save feature names
    feature_names = preprocessor.get_feature_names_out()
    pd.DataFrame({"feature_name": feature_names}).to_csv(models_dir / "final_feature_names.csv", index=False)

    # Save preprocessor
    from joblib import dump

    dump(preprocessor, models_dir / cfg["artifacts"]["preprocessor_file"])

    logger.info("Prepared data saved to %s", processed_dir)
    logger.info("Saved preprocessor and feature names to %s", models_dir)


if __name__ == "__main__":
    main()
