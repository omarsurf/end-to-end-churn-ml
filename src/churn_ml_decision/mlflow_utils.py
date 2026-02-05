"""MLflow integration utilities with lazy imports and graceful degradation."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

_mlflow = None  # lazy-loaded


def _get_mlflow():
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow as _m

            _mlflow = _m
        except ImportError:
            _mlflow = False
    return _mlflow if _mlflow is not False else None


def is_available() -> bool:
    return _get_mlflow() is not None


@contextmanager
def start_run(
    cfg: dict[str, Any],
    run_name: str | None = None,
) -> Generator[Any, None, None]:
    """Start an MLflow run if enabled in *cfg*. Yields run or ``None``."""
    mlflow_cfg = cfg.get("mlflow", {})
    if not mlflow_cfg.get("enabled", False):
        yield None
        return

    mlflow = _get_mlflow()
    if mlflow is None:
        logger.warning("mlflow.enabled is true but mlflow is not installed; skipping.")
        yield None
        return

    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "churn-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_params(_flatten_dict(params))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=step)


def log_artifact(local_path: str | Path) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(local_path))


def log_model(model: Any, artifact_path: str, cfg: dict[str, Any]) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return

    mlflow.sklearn.log_model(model, artifact_path=artifact_path)

    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("register_model", False):
        model_name = mlflow_cfg.get("model_name", "churn-classifier")
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mlflow.register_model(model_uri, model_name)
        logger.info("Registered model '%s' from run %s", model_name, run_id)


def set_tag(key: str, value: str) -> None:
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.set_tag(key, value)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
