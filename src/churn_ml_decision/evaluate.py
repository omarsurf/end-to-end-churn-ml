from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import load_typed_config, log_loaded_config, project_root, resolve_path
from .io import load_test_arrays, load_val_arrays
from .logging_config import setup_logging
from .mlflow_utils import log_artifact, log_metrics, log_params, set_tag, start_run
from .model_registry import ModelNotFoundError, ModelRegistry
from .track import file_sha256, log_run

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def load_model_with_retry(model_path: Path):
    return joblib.load(model_path)


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
                "Precision": float(precision),
                "Recall": float(recall),
                "F1_Score": float(f1),
                "True_Positives": int(tp),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "True_Negatives": int(tn),
                "Churners_Captured_%": float(churners_captured_pct),
                "False_Positive_Rate_%": float(false_positive_rate),
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


def check_quality_gates(
    roc_auc: float, recall: float, precision: float, quality: dict
) -> list[str]:
    """Return list of failed gate names. Empty list means all gates passed."""
    failures = []
    if roc_auc < quality["min_roc_auc"]:
        failures.append("roc_auc")
    if recall < quality["min_recall"]:
        failures.append("recall")
    if precision < quality["min_precision"]:
        failures.append("precision")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select threshold on validation and evaluate on test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--min-recall", type=float, default=None, help="Override min recall.")
    parser.add_argument("--threshold-min", type=float, default=None, help="Override threshold min.")
    parser.add_argument("--threshold-max", type=float, default=None, help="Override threshold max.")
    parser.add_argument(
        "--threshold-step", type=float, default=None, help="Override threshold step."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reserved strict mode flag for future compatibility.",
    )
    return parser.parse_args()


def _resolve_model_from_registry(root: Path, registry: ModelRegistry) -> tuple[Path, str | None]:
    model = None
    try:
        model = registry.get_production_model()
    except ModelNotFoundError:
        model = registry.get_latest_model()
    model_path = Path(model.model_path)
    if not model_path.is_absolute():
        model_path = resolve_path(root, model_path)
    return model_path, model.model_id


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

    model_id: str | None = None
    registry: ModelRegistry | None = None

    try:
        data_dir = resolve_path(root, cfg.paths.data_processed)
        models_dir = resolve_path(root, cfg.paths.models)
        threshold_file = cfg.artifacts.threshold_analysis_file
        results_file = cfg.artifacts.final_results_file

        min_recall = args.min_recall if args.min_recall is not None else cfg.evaluation.min_recall
        t_min = (
            args.threshold_min if args.threshold_min is not None else cfg.evaluation.threshold_min
        )
        t_max = (
            args.threshold_max if args.threshold_max is not None else cfg.evaluation.threshold_max
        )
        t_step = (
            args.threshold_step
            if args.threshold_step is not None
            else cfg.evaluation.threshold_step
        )

        x_val, y_val = load_val_arrays(data_dir)
        x_test, y_test = load_test_arrays(data_dir)

        model_path = models_dir / cfg.artifacts.model_file
        if cfg.registry.enabled and cfg.registry.use_current:
            registry = ModelRegistry(resolve_path(root, cfg.registry.file))
            try:
                model_path, model_id = _resolve_model_from_registry(root, registry)
            except ModelNotFoundError:
                logger.warning("No registry model found, falling back to canonical model artifact.")

        model = load_model_with_retry(model_path)
        logger.info(
            "Evaluate stage started",
            extra={
                "model_path": str(model_path),
                "model_id": model_id,
                "val_rows": int(len(y_val)),
            },
        )

        actual_model_type = cfg.model.name
        summary_path = models_dir / cfg.artifacts.train_summary_file
        if summary_path.exists():
            train_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            actual_model_type = train_summary.get("model_type", actual_model_type)

        y_val_proba = model.predict_proba(x_val)[:, 1]
        roc_auc_val = float(roc_auc_score(y_val, y_val_proba))
        logger.info("Validation baseline computed", extra={"roc_auc_val": roc_auc_val})

        thresholds = np.arange(t_min, t_max + 1e-9, t_step)
        thresholds = np.round(thresholds, 2)

        retained_value = cfg.business.retained_value
        contact_cost = cfg.business.contact_cost
        df_threshold = threshold_analysis(
            y_val,
            y_val_proba,
            thresholds,
            retained_value=retained_value,
            contact_cost=contact_cost,
        )
        selected_row, reason = select_threshold(df_threshold, min_recall)
        final_threshold = float(selected_row["Threshold"])
        logger.info(
            "Threshold selected",
            extra={"final_threshold": final_threshold, "selection_reason": reason},
        )

        y_test_proba = model.predict_proba(x_test)[:, 1]
        y_test_pred = (y_test_proba >= final_threshold).astype(int)

        roc_auc_test = float(roc_auc_score(y_test, y_test_proba))
        precision = float(precision_score(y_test, y_test_pred, zero_division=0))
        recall = float(recall_score(y_test, y_test_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_test_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

        net_value = None
        net_per_flagged = None
        if retained_value is not None and contact_cost is not None:
            net_value = int(tp) * retained_value - int(tp + fp) * contact_cost
            net_per_flagged = net_value / (int(tp + fp) or 1)

        final_results = {
            "model_type": actual_model_type,
            "model_id": model_id,
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
            "selection_reason": reason,
        }

        (models_dir / threshold_file).parent.mkdir(parents=True, exist_ok=True)
        df_threshold.to_csv(models_dir / threshold_file, index=False)
        pd.DataFrame([final_results]).to_csv(models_dir / results_file, index=False)

        with start_run(
            cfg.model_dump(mode="python"), run_name=f"evaluate-{actual_model_type}"
        ) as run:
            if run is not None:
                set_tag("stage", "evaluate")
                set_tag("model_type", actual_model_type)
                if model_id:
                    set_tag("model_id", model_id)
                log_params(
                    {
                        "final_threshold": final_threshold,
                        "selection_rule": reason,
                        "min_recall": min_recall,
                    }
                )
                test_metrics = {
                    "test_roc_auc": roc_auc_test,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1": f1,
                }
                if net_value is not None:
                    test_metrics["test_net_value"] = float(net_value)
                    test_metrics["test_net_per_flagged"] = float(net_per_flagged)
                log_metrics(test_metrics)
                log_artifact(str(models_dir / threshold_file))
                log_artifact(str(models_dir / results_file))
                logger.info("Logged evaluation run to MLflow", extra={"run_id": run.info.run_id})

        if cfg.tracking.enabled:
            raw_path = resolve_path(root, cfg.paths.data_raw)
            payload = {
                "stage": "evaluate",
                "model": actual_model_type,
                "model_id": model_id,
                "metrics": final_results,
                "selection_rule": reason,
                "artifacts": {
                    "thresholds": str(models_dir / threshold_file),
                    "results": str(models_dir / results_file),
                },
                "data_hash": file_sha256(raw_path) if raw_path.exists() else None,
                "config_path": str(args.config),
            }
            log_run(resolve_path(root, cfg.tracking.file), payload)

        failures = check_quality_gates(
            roc_auc_test,
            recall,
            precision,
            cfg.quality.model_dump(mode="python"),
        )
        if registry is not None and model_id is not None:
            status = "validation" if not failures else "deprecated"
            registry.update_status(
                model_id,
                status=status,
                metrics={
                    "test_roc_auc": roc_auc_test,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1": f1,
                },
            )

        if failures:
            logger.error("Quality gates failed", extra={"failed_gates": failures})
            raise SystemExit(f"Quality gates failed: {', '.join(failures)}")

        logger.info(
            "Evaluate stage finished",
            extra={
                "model_id": model_id,
                "roc_auc": roc_auc_test,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )
    except Exception:
        logger.exception("Evaluate stage failed")
        raise


if __name__ == "__main__":
    main()
