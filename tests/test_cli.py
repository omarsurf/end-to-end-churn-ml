"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from churn_ml_decision.cli import (
    check_drift_main,
    health_check_main,
    model_info_main,
    model_promote_main,
    model_rollback_main,
    validate_config_main,
)
from churn_ml_decision.model_registry import ModelMetadata, ModelRegistry
from churn_ml_decision.monitoring import DataDriftDetector


def _minimal_config(tmp_path: Path, registry_enabled: bool = True) -> dict:
    """Return minimal config dict for CLI tests."""
    return {
        "paths": {
            "data_raw": str(tmp_path / "data" / "raw"),
            "data_processed": str(tmp_path / "data" / "processed"),
            "models": str(tmp_path / "models"),
        },
        "model": {
            "name": "logistic_regression",
            "version": 1,
            "selection_metric": "roc_auc",
            "candidates": [
                {
                    "name": "lr",
                    "type": "logistic_regression",
                    "enabled": True,
                    "params": {"C": 1.0},
                }
            ],
        },
        "artifacts": {
            "preprocessor_file": "preprocessor.joblib",
            "model_file": "best_model.joblib",
            "final_results_file": "final_test_results.csv",
            "drift_reference_file": "drift_reference.json",
        },
        "registry": {
            "enabled": registry_enabled,
            "file": str(tmp_path / "models" / "registry.json"),
            "auto_promote_first_model": True,
        },
        "monitoring": {
            "enabled": False,
            "reference_file": str(tmp_path / "models" / "drift_reference.json"),
            "metrics_file": str(tmp_path / "metrics" / "production_metrics.json"),
            "drift_report_file": str(tmp_path / "metrics" / "drift_report.json"),
        },
        "tracking": {"enabled": False},
    }


@pytest.fixture
def cli_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Setup config, directories and mock project_root for CLI tests."""
    # Create directories
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "metrics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Write config
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_minimal_config(tmp_path)))

    # Mock project_root
    monkeypatch.setattr("churn_ml_decision.cli.project_root", lambda: tmp_path)

    return tmp_path, config_path


# =============================================================================
# validate_config_main tests
# =============================================================================


def test_validate_config_main_success(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    with patch("sys.argv", ["validate-config", "--config", str(config_path)]):
        validate_config_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "ok"
    assert output["model_name"] == "logistic_regression"
    assert output["registry_enabled"] is True


def test_validate_config_main_registry_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_minimal_config(tmp_path, registry_enabled=False)))
    monkeypatch.setattr("churn_ml_decision.cli.project_root", lambda: tmp_path)

    with patch("sys.argv", ["validate-config", "--config", str(config_path)]):
        validate_config_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["registry_enabled"] is False


# =============================================================================
# model_info_main tests
# =============================================================================


def test_model_info_main_with_production(
    cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture
):
    tmp_path, config_path = cli_setup

    # Setup registry with production model
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(
        "models/v1.joblib",
        ModelMetadata(
            model_id="v1",
            model_path="models/v1.joblib",
            config_hash="abc",
            metrics={"roc_auc": 0.85},
        ),
    )
    registry.promote("v1")

    with patch("sys.argv", ["model-info", "--config", str(config_path)]):
        model_info_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "ok"
    assert output["production_model"]["model_id"] == "v1"
    assert output["total_models"] == 1


def test_model_info_main_no_production(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    # Empty registry
    ModelRegistry(tmp_path / "models" / "registry.json")

    with patch("sys.argv", ["model-info", "--config", str(config_path)]):
        model_info_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "no_production_model"
    assert output["total_models"] == 0


# =============================================================================
# model_promote_main tests
# =============================================================================


def test_model_promote_main_success(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(
        "models/v1.joblib",
        ModelMetadata(
            model_id="v1",
            model_path="models/v1.joblib",
            config_hash="abc",
            status="training",
        ),
    )

    with patch("sys.argv", ["model-promote", "--model-id", "v1", "--config", str(config_path)]):
        model_promote_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "ok"
    assert output["action"] == "promote"
    assert output["model_id"] == "v1"


# =============================================================================
# model_rollback_main tests
# =============================================================================


def test_model_rollback_main_auto(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(
        "models/v1.joblib",
        ModelMetadata(model_id="v1", model_path="models/v1.joblib", config_hash="abc"),
    )
    registry.register(
        "models/v2.joblib",
        ModelMetadata(model_id="v2", model_path="models/v2.joblib", config_hash="abc"),
    )
    registry.promote("v1")
    registry.promote("v2")

    with patch("sys.argv", ["model-rollback", "--config", str(config_path)]):
        model_rollback_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "ok"
    assert output["action"] == "rollback"
    assert output["model_id"] == "v1"


def test_model_rollback_main_explicit_id(
    cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture
):
    tmp_path, config_path = cli_setup

    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(
        "models/v1.joblib",
        ModelMetadata(model_id="v1", model_path="models/v1.joblib", config_hash="abc"),
    )
    registry.register(
        "models/v2.joblib",
        ModelMetadata(model_id="v2", model_path="models/v2.joblib", config_hash="abc"),
    )
    registry.register(
        "models/v3.joblib",
        ModelMetadata(model_id="v3", model_path="models/v3.joblib", config_hash="abc"),
    )
    registry.promote("v3")

    with patch("sys.argv", ["model-rollback", "--model-id", "v1", "--config", str(config_path)]):
        model_rollback_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["model_id"] == "v1"


# =============================================================================
# check_drift_main tests
# =============================================================================


def test_check_drift_main_success(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    # Create reference data and fit detector
    ref_data = pd.DataFrame(
        {"MonthlyCharges": [20.0, 50.0, 80.0, 30.0, 60.0], "tenure": [1, 12, 24, 6, 18]}
    )
    detector = DataDriftDetector()
    detector.fit(ref_data)
    detector.save(tmp_path / "models" / "drift_reference.json")

    # Create input data
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"MonthlyCharges": [25.0, 55.0], "tenure": [2, 15]}).to_csv(
        input_path, index=False
    )

    with patch(
        "sys.argv",
        ["check-drift", "--input", str(input_path), "--config", str(config_path)],
    ):
        check_drift_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert "columns" in output
    assert "drift_score" in output


def test_check_drift_main_no_reference(cli_setup: tuple[Path, Path]):
    tmp_path, config_path = cli_setup

    input_path = tmp_path / "input.csv"
    pd.DataFrame({"MonthlyCharges": [25.0]}).to_csv(input_path, index=False)

    with patch(
        "sys.argv",
        ["check-drift", "--input", str(input_path), "--config", str(config_path)],
    ):
        with pytest.raises(SystemExit) as exc_info:
            check_drift_main()
        assert "Drift reference not found" in str(exc_info.value)


# =============================================================================
# health_check_main tests
# =============================================================================


def test_health_check_main_healthy(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    # Create all required artifacts
    (tmp_path / "models" / "preprocessor.joblib").write_text("fake")
    (tmp_path / "models" / "best_model.joblib").write_text("fake")
    (tmp_path / "metrics" / "production_metrics.json").write_text("{}")

    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register(
        "models/v1.joblib",
        ModelMetadata(model_id="v1", model_path="models/v1.joblib", config_hash="abc"),
    )
    registry.promote("v1")

    with patch("sys.argv", ["health-check", "--config", str(config_path)]):
        health_check_main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "healthy"
    assert output["checks"]["preprocessor_exists"] is True
    assert output["checks"]["production_model_set"] is True


def test_health_check_main_degraded(cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture):
    tmp_path, config_path = cli_setup

    # Missing preprocessor, model - degraded state

    with patch("sys.argv", ["health-check", "--config", str(config_path)]):
        with pytest.raises(SystemExit) as exc_info:
            health_check_main()
        assert exc_info.value.code == 1

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "degraded"
    assert output["checks"]["preprocessor_exists"] is False


def test_health_check_main_no_production_model(
    cli_setup: tuple[Path, Path], capsys: pytest.CaptureFixture
):
    tmp_path, config_path = cli_setup

    # Create artifacts but no production model
    (tmp_path / "models" / "preprocessor.joblib").write_text("fake")
    (tmp_path / "models" / "best_model.joblib").write_text("fake")
    (tmp_path / "metrics" / "production_metrics.json").write_text("{}")

    # Empty registry (no production model)
    ModelRegistry(tmp_path / "models" / "registry.json")

    with patch("sys.argv", ["health-check", "--config", str(config_path)]):
        with pytest.raises(SystemExit) as exc_info:
            health_check_main()
        assert exc_info.value.code == 1

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["checks"]["production_model_set"] is False
