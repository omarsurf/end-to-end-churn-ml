from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config, project_root, resolve_path
from .prepare import engineer_features, clean_total_charges
from .registry import current_model_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score new customers.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV of customers.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with scores.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold for label.")
    return parser.parse_args()


def load_threshold(results_path: Path) -> float | None:
    if not results_path.exists():
        return None
    latest = pd.read_csv(results_path).tail(1).iloc[0]
    return float(latest["final_threshold"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    models_dir = resolve_path(root, cfg["paths"]["models"])
    preprocessor_path = models_dir / cfg["artifacts"]["preprocessor_file"]
    if not preprocessor_path.exists():
        raise SystemExit("Preprocessor not found. Run churn-prepare first.")

    registry_cfg = cfg.get("registry", {})
    model_path = None
    if registry_cfg.get("enabled", False) and registry_cfg.get("use_current", False):
        current = current_model_path(resolve_path(root, registry_cfg["file"]))
        if current:
            model_path = Path(current)
    if model_path is None:
        model_path = models_dir / cfg["artifacts"]["model_file"]

    if not model_path.exists():
        raise SystemExit("Model not found. Run churn-train first.")

    df = pd.read_csv(args.input)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    customer_ids = df["customerID"] if "customerID" in df.columns else None
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = clean_total_charges(df)

    train_medians_file = cfg["artifacts"].get("train_medians_file", "train_medians.json")
    medians_path = models_dir / train_medians_file
    train_medians = None
    if medians_path.exists():
        train_medians = pd.read_json(medians_path, typ="series").to_dict()
    if cfg.get("engineering", {}).get("enabled", False) and train_medians is None:
        raise SystemExit("Train medians not found. Run churn-prepare first.")

    if cfg.get("engineering", {}).get("enabled", False):
        df, _ = engineer_features(df, train_medians=train_medians, cfg=cfg)

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    x = preprocessor.transform(df)
    proba = model.predict_proba(x)[:, 1]

    threshold = args.threshold
    if threshold is None:
        threshold = load_threshold(models_dir / cfg["artifacts"]["final_results_file"])

    output = pd.DataFrame({"churn_probability": proba})
    if threshold is not None:
        output["churn_prediction"] = (proba >= threshold).astype(int)

    if customer_ids is not None:
        output.insert(0, "customerID", customer_ids)

    output.to_csv(args.output, index=False)
    logger.info("Wrote predictions to %s", args.output)


if __name__ == "__main__":
    main()
