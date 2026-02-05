from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import load_config, project_root, resolve_path
from .io import load_test_arrays, load_val_arrays
from .registry import current_model_path
from .track import file_sha256, log_run

logger = logging.getLogger(__name__)

def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    retained_value: float | None = None,
    contact_cost: float | None = None,
) -> pd.DataFrame:
    rows = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        total_churners = tp + fn
        total_non_churners = tn + fp
        churners_captured_pct = (tp / total_churners * 100) if total_churners > 0 else 0.0
        false_positive_rate = (fp / total_non_churners * 100) if total_non_churners > 0 else 0.0
        total_flagged = tp + fp

        rows.append(
            {
                "Threshold": float(thresh),
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "True_Positives": int(tp),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "True_Negatives": int(tn),
                "Churners_Captured_%": churners_captured_pct,
                "False_Positive_Rate_%": false_positive_rate,
                "Total_Flagged": int(total_flagged),
            }
        )

    df = pd.DataFrame(rows)
    if retained_value is not None and contact_cost is not None:
        df["Net_Value"] = df["True_Positives"] * retained_value - df["Total_Flagged"] * contact_cost
        df["Net_per_Flagged"] = df["Net_Value"] / df["Total_Flagged"].replace(0, np.nan)
        df["Net_per_Flagged"] = df["Net_per_Flagged"].fillna(0.0)
    return df


def select_threshold(df: pd.DataFrame, min_recall: float) -> tuple[pd.Series, str]:
    high_recall = df[df["Recall"] >= min_recall]
    if not high_recall.empty:
        idx = high_recall["Precision"].idxmax()
        reason = f"Recall >= {min_recall:.2f} with best precision"
    else:
        idx = df["F1_Score"].idxmax()
        reason = "Best F1 (fallback)"
    return df.loc[idx], reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select threshold on validation and evaluate on test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--min-recall", type=float, default=None, help="Override min recall.")
    parser.add_argument("--threshold-min", type=float, default=None, help="Override threshold min.")
    parser.add_argument("--threshold-max", type=float, default=None, help="Override threshold max.")
    parser.add_argument("--threshold-step", type=float, default=None, help="Override threshold step.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    data_dir = resolve_path(root, cfg["paths"]["data_processed"])
    models_dir = resolve_path(root, cfg["paths"]["models"])
    model_file = cfg["artifacts"]["model_file"]
    threshold_file = cfg["artifacts"]["threshold_analysis_file"]
    results_file = cfg["artifacts"]["final_results_file"]

    min_recall = args.min_recall if args.min_recall is not None else cfg["evaluation"]["min_recall"]
    t_min = args.threshold_min if args.threshold_min is not None else cfg["evaluation"]["threshold_min"]
    t_max = args.threshold_max if args.threshold_max is not None else cfg["evaluation"]["threshold_max"]
    t_step = args.threshold_step if args.threshold_step is not None else cfg["evaluation"]["threshold_step"]

    x_val, y_val = load_val_arrays(data_dir)
    x_test, y_test = load_test_arrays(data_dir)

    registry_cfg = cfg.get("registry", {})
    model_path = None
    if registry_cfg.get("enabled", False) and registry_cfg.get("use_current", False):
        registry_path = resolve_path(root, registry_cfg["file"])
        current = current_model_path(registry_path)
        if current:
            model_path = Path(current)
    if model_path is None:
        model_path = models_dir / model_file

    model = joblib.load(model_path)

    # Validation baseline
    y_val_proba = model.predict_proba(x_val)[:, 1]
    roc_auc_val = roc_auc_score(y_val, y_val_proba)
    logger.info("Validation ROC-AUC: %.4f", roc_auc_val)

    thresholds = np.arange(t_min, t_max + 1e-9, t_step)
    thresholds = np.round(thresholds, 2)
    business = cfg.get("business", {})
    retained_value = business.get("retained_value")
    if retained_value is None:
        clv = business.get("clv")
        success_rate = business.get("success_rate")
        if clv is not None and success_rate is not None:
            retained_value = float(clv) * float(success_rate)
    contact_cost = business.get("contact_cost")
    df_threshold = threshold_analysis(
        y_val, y_val_proba, thresholds, retained_value=retained_value, contact_cost=contact_cost
    )

    selected_row, reason = select_threshold(df_threshold, min_recall)
    final_threshold = float(selected_row["Threshold"])

    logger.info("Selected threshold (validation): %s", final_threshold)
    logger.info("Selection rule: %s", reason)

    # Final evaluation on test set
    y_test_proba = model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_proba >= final_threshold).astype(int)

    roc_auc_test = roc_auc_score(y_test, y_test_proba)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    logger.info("Test ROC-AUC: %.4f", roc_auc_test)
    net_value = None
    net_per_flagged = None
    if retained_value is not None and contact_cost is not None:
        net_value = int(tp) * retained_value - int(tp + fp) * contact_cost
        net_per_flagged = net_value / (int(tp + fp) or 1)

    logger.info("Test Precision: %.4f", precision)
    logger.info("Test Recall: %.4f", recall)
    logger.info("Test F1: %.4f", f1)
    if net_value is not None:
        logger.info("Test Net Value: %.2f", net_value)

    final_results = {
        "model_type": cfg["model"]["name"],
        "final_threshold": final_threshold,
        "test_set_size": int(len(y_test)),
        "roc_auc": roc_auc_test,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "net_value": net_value,
        "net_value_per_flagged": net_per_flagged,
        "evaluation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df_threshold.to_csv(models_dir / threshold_file, index=False)
    pd.DataFrame([final_results]).to_csv(models_dir / results_file, index=False)

    logger.info("Wrote %s", models_dir / threshold_file)
    logger.info("Wrote %s", models_dir / results_file)

    # Experiment tracking
    tracking_cfg = cfg.get("tracking", {})
    if tracking_cfg.get("enabled", False):
        raw_path = resolve_path(root, cfg["paths"]["data_raw"])
        payload = {
            "stage": "evaluate",
            "model": cfg["model"]["name"],
            "metrics": final_results,
            "selection_rule": reason,
            "artifacts": {
                "thresholds": str(models_dir / threshold_file),
                "results": str(models_dir / results_file),
            },
            "data_hash": file_sha256(raw_path) if raw_path.exists() else None,
            "config_path": str(args.config),
        }
        log_run(resolve_path(root, tracking_cfg["file"]), payload)
        logger.info("Logged evaluation run to %s", tracking_cfg["file"])

    # Quality gates
    quality = cfg.get("quality", {})
    if quality:
        failures = []
        if roc_auc_test < quality["min_roc_auc"]:
            failures.append("roc_auc")
        if recall < quality["min_recall"]:
            failures.append("recall")
        if precision < quality["min_precision"]:
            failures.append("precision")
        if failures:
            raise SystemExit(f"Quality gates failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
