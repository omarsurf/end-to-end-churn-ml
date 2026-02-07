from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import (
    config_hash_from_file,
    load_typed_config,
    log_loaded_config,
    project_root,
    resolve_path,
)
from .io import load_train_val_arrays
from .logging_config import setup_logging
from .mlflow_utils import log_artifact, log_metrics, log_model, log_params, set_tag, start_run
from .model_registry import ModelMetadata, ModelRegistry
from .track import file_sha256, log_run

logger = logging.getLogger(__name__)


def _parse_major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    values: list[int] = []
    for part in parts[:2]:
        digits = "".join(ch for ch in part if ch.isdigit())
        values.append(int(digits) if digits else 0)
    while len(values) < 2:
        values.append(0)
    return values[0], values[1]


def _sklearn_penalty_is_deprecated() -> bool:
    return _parse_major_minor(sklearn.__version__) >= (1, 8)


def _normalize_logistic_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize LogisticRegression parameters across sklearn versions."""
    normalized = dict(params)
    has_penalty = "penalty" in normalized
    penalty = normalized.get("penalty")
    if isinstance(penalty, str):
        penalty = penalty.lower()

    if penalty in {"l1", "l2"} and "l1_ratio" not in normalized:
        normalized["l1_ratio"] = 1.0 if penalty == "l1" else 0.0

    if _sklearn_penalty_is_deprecated():
        if has_penalty and penalty in {None, "none"}:
            normalized["C"] = float("inf")
        normalized.pop("penalty", None)

    return normalized


def build_model(candidate: dict[str, Any]):
    """Build a model instance from a candidate config dict."""
    model_type = candidate["type"]
    params = candidate.get("params", {})
    if model_type == "logistic_regression":
        return LogisticRegression(**_normalize_logistic_params(params))
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is not installed. Install with pip install -e '.[ml]'"
            ) from exc
        return XGBClassifier(**params)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any enabled candidate cannot train/evaluate.",
    )
    return parser.parse_args()


def _load_feature_names(models_dir: Path) -> list[str]:
    feature_file = models_dir / "final_feature_names.csv"
    if not feature_file.exists():
        return []
    df = pd.read_csv(feature_file)
    if "feature_name" not in df.columns:
        return []
    return df["feature_name"].astype(str).tolist()


def _extract_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "coef_"):
        coefs = model.coef_
        if getattr(coefs, "ndim", 1) > 1:
            coefs = coefs[0]
        values = [abs(float(v)) for v in coefs.tolist()]
    elif hasattr(model, "feature_importances_"):
        values = [float(v) for v in model.feature_importances_.tolist()]
    else:
        return {}

    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(len(values))]
    if len(feature_names) != len(values):
        size = min(len(feature_names), len(values))
        feature_names = feature_names[:size]
        values = values[:size]
    return dict(zip(feature_names, values))


def _build_registry_model_file(
    template: str,
    *,
    model_name: str,
    version: int,
    timestamp: str,
) -> str:
    has_timestamp_placeholder = "{timestamp}" in template
    rendered = template.format(name=model_name, version=version, timestamp=timestamp)
    if has_timestamp_placeholder:
        return rendered

    rendered_path = Path(rendered)
    if rendered_path.suffix:
        unique_name = f"{rendered_path.stem}_{timestamp}{rendered_path.suffix}"
    else:
        unique_name = f"{rendered_path.name}_{timestamp}"

    if str(rendered_path.parent) in {"", "."}:
        return unique_name
    return str(rendered_path.parent / unique_name)


def _register_model(
    *,
    registry: ModelRegistry,
    model_path: Path,
    model_id: str,
    config_path: Path,
    metrics: dict[str, float],
    input_features: list[str],
    feature_importance: dict[str, float],
    auto_promote_first_model: bool,
) -> str:
    metadata = ModelMetadata(
        model_id=model_id,
        model_path=str(model_path),
        config_hash=config_hash_from_file(config_path),
        metrics=metrics,
        status="training",
        input_features=input_features,
        feature_importance=feature_importance,
    )
    registry.register(model_path, metadata)

    if auto_promote_first_model and len(registry.list_models(status="production")) == 0:
        registry.promote(model_id)
        logger.info("Auto-promoted first registered model", extra={"model_id": model_id})

    return model_id


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
        data_dir = resolve_path(root, cfg.paths.data_processed)
        models_dir = resolve_path(root, cfg.paths.models)
        models_dir.mkdir(parents=True, exist_ok=True)

        model_file = args.output_model or cfg.artifacts.model_file
        x_train, y_train, x_val, y_val = load_train_val_arrays(data_dir)
        logger.info(
            "Train stage started",
            extra={
                "train_rows": int(x_train.shape[0]),
                "val_rows": int(x_val.shape[0]),
                "features": int(x_train.shape[1]),
            },
        )

        candidates = [c.model_dump(mode="python") for c in cfg.model.candidates]
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            if not candidate.get("enabled", True):
                continue
            name = candidate["name"]
            logger.info("Training candidate", extra={"candidate": name, "type": candidate["type"]})
            try:
                model = build_model(candidate)
                model.fit(x_train, y_train)

                y_val_proba = model.predict_proba(x_val)[:, 1]
                y_val_pred = (y_val_proba >= 0.5).astype(int)

                roc_auc = float(roc_auc_score(y_val, y_val_proba))
                precision = float(precision_score(y_val, y_val_pred, zero_division=0))
                recall = float(recall_score(y_val, y_val_pred, zero_division=0))
                f1 = float(f1_score(y_val, y_val_pred, zero_division=0))

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
                    "Candidate metrics",
                    extra={
                        "candidate": name,
                        "roc_auc": roc_auc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    },
                )
            except Exception:
                if args.strict:
                    raise
                logger.exception(
                    "Candidate failed and was skipped (strict disabled).",
                    extra={"candidate": name, "type": candidate["type"]},
                )

        if not results:
            raise SystemExit("No enabled model candidates found.")

        selection_metric = cfg.model.selection_metric
        best = max(results, key=lambda r: r[selection_metric])
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        generated_model_id = f"{best['name']}-v{run_timestamp}"

        if cfg.registry.enabled:
            model_template = args.output_model or cfg.registry.template
            model_file = _build_registry_model_file(
                model_template,
                model_name=best["name"],
                version=cfg.model.version,
                timestamp=run_timestamp,
            )
        else:
            # Enforce timestamp even without registry to prevent overwrite
            model_file = args.output_model or f"{best['name']}_v{cfg.model.version}_{run_timestamp}.joblib"

        model_path = models_dir / model_file
        joblib.dump(best["model"], model_path)
        logger.info("Saved selected model", extra={"model_path": str(model_path)})

        # Canonical file for backward compatibility with DVC/tests.
        canonical_path = models_dir / cfg.artifacts.model_file
        if canonical_path != model_path:
            joblib.dump(best["model"], canonical_path)
            logger.info(
                "Saved canonical model alias", extra={"canonical_path": str(canonical_path)}
            )

        summary = {
            "model_type": best["name"],
            "selection_metric": selection_metric,
            "params": best["params"],
            "validation_roc_auc": best["roc_auc"],
            "validation_precision": best["precision"],
            "validation_recall": best["recall"],
            "validation_f1": best["f1"],
        }
        summary_path = models_dir / cfg.artifacts.train_summary_file
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        registered_model_id = None
        if cfg.registry.enabled:
            registry = ModelRegistry(resolve_path(root, cfg.registry.file))
            input_features = _load_feature_names(models_dir)
            feature_importance = _extract_feature_importance(best["model"], input_features)
            # Registry points to timestamped file (immutable) for true rollback support.
            # best_model.joblib is kept as local alias for notebooks/tests only.
            registered_model_id = _register_model(
                registry=registry,
                model_path=model_path,
                model_id=generated_model_id,
                config_path=args.config,
                metrics={
                    "roc_auc": best["roc_auc"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                },
                input_features=input_features,
                feature_importance=feature_importance,
                auto_promote_first_model=cfg.registry.auto_promote_first_model,
            )
            logger.info("Model registered", extra={"model_id": registered_model_id})

        with start_run(cfg.model_dump(mode="python"), run_name=f"train-{best['name']}") as run:
            if run is not None:
                set_tag("stage", "train")
                set_tag("model_type", best["name"])
                if registered_model_id:
                    set_tag("model_id", registered_model_id)
                log_params(best["params"])
                log_params({"selection_metric": selection_metric})
                log_metrics(
                    {
                        "val_roc_auc": best["roc_auc"],
                        "val_precision": best["precision"],
                        "val_recall": best["recall"],
                        "val_f1": best["f1"],
                    }
                )
                log_artifact(str(summary_path))
                log_model(best["model"], artifact_path="model", cfg=cfg.model_dump(mode="python"))
                logger.info("Logged training run to MLflow", extra={"run_id": run.info.run_id})

        if cfg.tracking.enabled:
            raw_path = resolve_path(root, cfg.paths.data_raw)
            payload = {
                "stage": "train",
                "model": best["name"],
                "model_id": registered_model_id,
                "params": best["params"],
                "metrics": {
                    "validation_roc_auc": best["roc_auc"],
                    "validation_precision": best["precision"],
                    "validation_recall": best["recall"],
                    "validation_f1": best["f1"],
                },
                "artifacts": {"model": str(model_path)},
                "data_hash": file_sha256(raw_path) if raw_path.exists() else None,
                "config_path": str(args.config),
            }
            log_run(resolve_path(root, cfg.tracking.file), payload)

        logger.info(
            "Train stage finished",
            extra={
                "selected_model": best["name"],
                "selection_metric": selection_metric,
                "registered_model_id": registered_model_id,
            },
        )
    except Exception:
        logger.exception("Train stage failed")
        raise


if __name__ == "__main__":
    main()
