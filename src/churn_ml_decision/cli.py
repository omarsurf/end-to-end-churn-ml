from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_typed_config, project_root, resolve_path
from .model_registry import ModelNotFoundError, ModelRegistry
from .monitoring import DataDriftDetector


def validate_config_main() -> None:
    parser = argparse.ArgumentParser(description="Validate churn pipeline config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    cfg = load_typed_config(args.config)
    payload = {
        "status": "ok",
        "config_path": str(args.config),
        "model_name": cfg.model.name,
        "selection_metric": cfg.model.selection_metric,
        "registry_enabled": cfg.registry.enabled,
        "monitoring_enabled": cfg.monitoring.enabled,
    }
    print(json.dumps(payload, indent=2))


def model_info_main() -> None:
    parser = argparse.ArgumentParser(description="Show active production model from registry.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)
    registry = ModelRegistry(resolve_path(root, cfg.registry.file))

    try:
        production = registry.get_production_model()
        payload = {
            "status": "ok",
            "production_model": production.model_dump(mode="json"),
            "total_models": len(registry.list_models()),
        }
    except ModelNotFoundError:
        payload = {"status": "no_production_model", "total_models": len(registry.list_models())}
    print(json.dumps(payload, indent=2))


def model_promote_main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model to production.")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID to promote.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)
    registry = ModelRegistry(resolve_path(root, cfg.registry.file))
    model = registry.promote(args.model_id)
    print(
        json.dumps(
            {
                "status": "ok",
                "action": "promote",
                "model_id": model.model_id,
                "model_path": model.model_path,
            },
            indent=2,
        )
    )


def model_rollback_main() -> None:
    parser = argparse.ArgumentParser(description="Rollback production model.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional explicit model ID to rollback to.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)
    registry = ModelRegistry(resolve_path(root, cfg.registry.file))
    model = registry.rollback(args.model_id)
    print(
        json.dumps(
            {
                "status": "ok",
                "action": "rollback",
                "model_id": model.model_id,
                "model_path": model.model_path,
            },
            indent=2,
        )
    )


def check_drift_main() -> None:
    parser = argparse.ArgumentParser(description="Check data drift against reference distribution.")
    parser.add_argument("--input", type=Path, required=True, help="Path to new batch CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)

    detector_path = resolve_path(root, cfg.monitoring.reference_file)
    if not detector_path.exists():
        fallback = resolve_path(root, cfg.paths.models) / cfg.artifacts.drift_reference_file
        detector_path = fallback
    if not detector_path.exists():
        raise SystemExit("Drift reference not found. Run churn-prepare first.")

    detector = DataDriftDetector.load(detector_path)
    new_data = pd.read_csv(args.input)
    report = detector.detect_drift(new_data)

    columns = report.get("columns", {})
    drifted = [
        col
        for col, details in columns.items()
        if isinstance(details, dict) and details.get("status") == "DRIFT_DETECTED"
    ]
    report["drift_score"] = float(len(drifted) / max(len(columns), 1))

    report_path = resolve_path(root, cfg.monitoring.drift_report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def health_check_main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline production health check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)

    models_dir = resolve_path(root, cfg.paths.models)
    preprocessor_path = models_dir / cfg.artifacts.preprocessor_file
    canonical_model = models_dir / cfg.artifacts.model_file
    metrics_path = resolve_path(root, cfg.monitoring.metrics_file)
    registry_path = resolve_path(root, cfg.registry.file)

    checks = {
        "config_valid": True,
        "preprocessor_exists": preprocessor_path.exists(),
        "canonical_model_exists": canonical_model.exists(),
        "registry_exists": registry_path.exists(),
        "metrics_file_exists": metrics_path.exists(),
    }

    production_model = None
    if registry_path.exists():
        registry = ModelRegistry(registry_path)
        try:
            production_model = registry.get_production_model().model_dump(mode="json")
            checks["production_model_set"] = True
        except ModelNotFoundError:
            checks["production_model_set"] = False
    else:
        checks["production_model_set"] = False

    status = "healthy" if all(checks.values()) else "degraded"
    payload = {"status": status, "checks": checks, "production_model": production_model}
    print(json.dumps(payload, indent=2))
    if status != "healthy":
        raise SystemExit(1)
