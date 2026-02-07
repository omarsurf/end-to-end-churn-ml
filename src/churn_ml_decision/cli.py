from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_typed_config, project_root, resolve_path
from .model_registry import ModelNotFoundError, ModelRegistry
from .monitoring import DataDriftDetector, ProductionMetricsTracker
from .prepare import clean_total_charges, engineer_features


def _load_train_medians(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _prepare_drift_input(
    new_data: pd.DataFrame,
    *,
    cfg,
    models_dir: Path,
    reference_columns: list[str],
) -> pd.DataFrame:
    prepared = new_data.copy()
    if "Churn" in prepared.columns:
        prepared = prepared.drop(columns=["Churn"])
    if "customerID" in prepared.columns:
        prepared = prepared.drop(columns=["customerID"])

    if set(reference_columns).issubset(prepared.columns):
        return prepared

    if "TotalCharges" in prepared.columns:
        prepared["TotalCharges"] = clean_total_charges(prepared)

    if not cfg.engineering.enabled:
        return prepared

    medians_path = models_dir / cfg.artifacts.train_medians_file
    train_medians = _load_train_medians(medians_path)
    engineered, _ = engineer_features(
        prepared,
        train_medians=train_medians,
        fit=False,
        cfg=cfg,
    )
    return engineered


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
    models_dir = resolve_path(root, cfg.paths.models)

    detector_path = resolve_path(root, cfg.monitoring.reference_file)
    if not detector_path.exists():
        fallback = models_dir / cfg.artifacts.drift_reference_file
        detector_path = fallback
    if not detector_path.exists():
        raise SystemExit("Drift reference not found. Run churn-prepare first.")

    detector = DataDriftDetector.load(detector_path)
    new_data = pd.read_csv(args.input)
    reference_columns = list(detector.reference_samples.keys())
    drift_input = _prepare_drift_input(
        new_data,
        cfg=cfg,
        models_dir=models_dir,
        reference_columns=reference_columns,
    )
    report = detector.detect_drift(drift_input)

    columns = report.get("columns", {})
    problematic_statuses = {"DRIFT_DETECTED", "MISSING_IN_NEW_DATA", "NO_VALID_VALUES"}
    problematic = [
        col
        for col, details in columns.items()
        if isinstance(details, dict) and details.get("status") in problematic_statuses
    ]
    report["drift_score"] = float(len(problematic) / max(len(columns), 1))
    report["drift_columns"] = problematic

    metrics_tracker = ProductionMetricsTracker(resolve_path(root, cfg.monitoring.metrics_file))
    metrics_tracker.update_drift_metrics(drift_score=report["drift_score"])

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

    if cfg.monitoring.enabled:
        configured_drift_ref = resolve_path(root, cfg.monitoring.reference_file)
        fallback_drift_ref = models_dir / cfg.artifacts.drift_reference_file
        checks["drift_reference_exists"] = configured_drift_ref.exists() or fallback_drift_ref.exists()
    else:
        checks["drift_reference_exists"] = True

    production_model = None
    if registry_path.exists():
        registry = ModelRegistry(registry_path)
        try:
            production = registry.get_production_model()
            production_model = production.model_dump(mode="json")
            checks["production_model_set"] = True
            checks["production_model_status_valid"] = production.status == "production"
            production_model_path = Path(production.model_path)
            if not production_model_path.is_absolute():
                production_model_path = resolve_path(root, production_model_path)
                checks["production_model_path_portable"] = True
            else:
                try:
                    production_model_path.resolve().relative_to(root.resolve())
                    checks["production_model_path_portable"] = True
                except ValueError:
                    checks["production_model_path_portable"] = False
            checks["production_model_exists"] = production_model_path.exists()
        except ModelNotFoundError:
            checks["production_model_set"] = False
            checks["production_model_exists"] = False
            checks["production_model_status_valid"] = False
            checks["production_model_path_portable"] = False
    else:
        checks["production_model_set"] = False
        checks["production_model_exists"] = False
        checks["production_model_status_valid"] = False
        checks["production_model_path_portable"] = False

    status = "healthy" if all(checks.values()) else "degraded"
    payload = {"status": status, "checks": checks, "production_model": production_model}
    print(json.dumps(payload, indent=2))
    if status != "healthy":
        raise SystemExit(1)
