from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import load_config, project_root, resolve_path
from .io import load_train_val_arrays
from .mlflow_utils import log_artifact, log_metrics, log_model, log_params, set_tag, start_run
from .registry import update_registry
from .track import file_sha256, log_run

logger = logging.getLogger(__name__)


def build_model(candidate: dict):
    """Build a model instance from a candidate config dict."""
    model_type = candidate["type"]
    params = candidate.get("params", {})
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install with pip install -e '.[ml]'"
            ) from exc
        return XGBClassifier(**params)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "lightgbm is not installed. Install with pip install -e '.[ml]'"
            ) from exc
        return LGBMClassifier(**params)
    raise ValueError(f"Unsupported model type: {model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model on processed arrays.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Override model output filename (joblib).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    data_dir = resolve_path(root, cfg["paths"]["data_processed"])
    models_dir = resolve_path(root, cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_file = args.output_model or cfg["artifacts"]["model_file"]

    x_train, y_train, x_val, y_val = load_train_val_arrays(data_dir)

    model_cfg = cfg["model"]
    candidates = model_cfg.get("candidates") or []

    results = []
    for candidate in candidates:
        if not candidate.get("enabled", True):
            continue
        name = candidate["name"]
        logger.info("Training candidate: %s", name)
        model = build_model(candidate)
        model.fit(x_train, y_train)

        y_val_proba = model.predict_proba(x_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val, y_val_proba)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        results.append(
            {
                "name": name,
                "type": candidate["type"],
                "params": candidate.get("params", {}),
                "model": model,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        logger.info(
            "%s - ROC-AUC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
            name,
            roc_auc,
            precision,
            recall,
            f1,
        )

    if not results:
        raise SystemExit("No enabled model candidates found.")

    selection_metric = model_cfg.get("selection_metric", "roc_auc")
    best = max(results, key=lambda r: r[selection_metric])

    registry_cfg = cfg.get("registry", {})
    if registry_cfg.get("enabled", False):
        template = registry_cfg["template"]
        version = model_cfg.get("version", 1)
        model_file = template.format(name=best["name"], version=version)

    joblib.dump(best["model"], models_dir / model_file)
    logger.info("Saved model: %s", models_dir / model_file)

    # Always save canonical name for DVC pipeline compatibility
    canonical = cfg["artifacts"]["model_file"]
    if model_file != canonical:
        joblib.dump(best["model"], models_dir / canonical)
        logger.info("Saved canonical model: %s", models_dir / canonical)

    summary = {
        "model_type": best["name"],
        "selection_metric": selection_metric,
        "params": best["params"],
        "validation_roc_auc": best["roc_auc"],
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_f1": best["f1"],
    }
    summary_path = models_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)

    # MLflow tracking
    with start_run(cfg, run_name=f"train-{best['name']}") as run:
        if run is not None:
            set_tag("stage", "train")
            set_tag("model_type", best["name"])
            log_params(best["params"])
            log_params({"selection_metric": selection_metric})
            log_metrics({
                "val_roc_auc": best["roc_auc"],
                "val_precision": best["precision"],
                "val_recall": best["recall"],
                "val_f1": best["f1"],
            })
            log_artifact(str(summary_path))
            log_model(best["model"], artifact_path="model", cfg=cfg)
            logger.info("Logged training run to MLflow (run_id=%s)", run.info.run_id)

    # JSONL experiment tracking
    tracking_cfg = cfg.get("tracking", {})
    if tracking_cfg.get("enabled", False):
        raw_path = resolve_path(root, cfg["paths"]["data_raw"])
        payload = {
            "stage": "train",
            "model": best["name"],
            "params": best["params"],
            "metrics": {
                "validation_roc_auc": best["roc_auc"],
                "validation_precision": best["precision"],
                "validation_recall": best["recall"],
                "validation_f1": best["f1"],
            },
            "artifacts": {"model": str(models_dir / model_file)},
            "data_hash": file_sha256(raw_path) if raw_path.exists() else None,
            "config_path": str(args.config),
        }
        log_run(resolve_path(root, tracking_cfg["file"]), payload)
        logger.info("Logged train run to %s", tracking_cfg["file"])

    if registry_cfg.get("enabled", False):
        registry_path = resolve_path(root, registry_cfg["file"])
        update_registry(
            registry_path,
            {
                "model_name": best["name"],
                "model_path": str(models_dir / model_file),
                "metrics": {
                    "roc_auc": best["roc_auc"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                },
                "params": best["params"],
            },
        )
        logger.info("Updated registry at %s", registry_cfg["file"])


if __name__ == "__main__":
    main()
