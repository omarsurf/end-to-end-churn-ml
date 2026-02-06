"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import yaml

from churn_ml_decision.config import load_typed_config
from churn_ml_decision.pipeline import PipelineOrchestrator, StageResult


def _minimal_config(tmp_path: Path) -> Path:
    """Create minimal config for pipeline tests."""
    config = {
        "paths": {
            "data_raw": str(tmp_path / "data" / "raw"),
            "data_processed": str(tmp_path / "data" / "processed"),
            "models": str(tmp_path / "models"),
        },
        "model": {
            "name": "logistic_regression",
            "version": 1,
            "selection_metric": "roc_auc",
            "candidates": [{"name": "lr", "type": "logistic_regression", "enabled": True}],
        },
        "artifacts": {
            "preprocessor_file": "preprocessor.joblib",
            "model_file": "best_model.joblib",
            "final_results_file": "final_test_results.csv",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def test_health_snapshot_all_present(tmp_path: Path):
    """Test health_snapshot returns all True when artifacts exist."""
    config_path = _minimal_config(tmp_path)
    cfg = load_typed_config(config_path)

    # Create all artifacts
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "preprocessor.joblib").write_text("fake")
    (models_dir / "best_model.joblib").write_text("fake")
    (models_dir / "final_test_results.csv").write_text("col\n1")

    orchestrator = PipelineOrchestrator(cfg)
    snapshot = orchestrator.health_snapshot(tmp_path)

    assert snapshot["preprocessor_exists"] is True
    assert snapshot["model_exists"] is True
    assert snapshot["results_exists"] is True


def test_health_snapshot_partial(tmp_path: Path):
    """Test health_snapshot returns mixed values with partial artifacts."""
    config_path = _minimal_config(tmp_path)
    cfg = load_typed_config(config_path)

    # Create only preprocessor
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "preprocessor.joblib").write_text("fake")

    orchestrator = PipelineOrchestrator(cfg)
    snapshot = orchestrator.health_snapshot(tmp_path)

    assert snapshot["preprocessor_exists"] is True
    assert snapshot["model_exists"] is False
    assert snapshot["results_exists"] is False


def test_run_stage_success():
    """Test run_stage returns success for successful callable."""

    # Create a minimal config mock
    class MockConfig:
        class Paths:
            models = "models"

        paths = Paths()

        class Artifacts:
            preprocessor_file = "p.joblib"
            model_file = "m.joblib"
            final_results_file = "r.csv"

        artifacts = Artifacts()

    orchestrator = PipelineOrchestrator(MockConfig())

    def success_fn():
        pass

    result = orchestrator.run_stage("test-stage", success_fn)

    assert isinstance(result, StageResult)
    assert result.stage == "test-stage"
    assert result.success is True
    assert result.detail == "ok"


def test_run_stage_exception():
    """Test run_stage returns failure for callable that raises."""

    class MockConfig:
        class Paths:
            models = "models"

        paths = Paths()

        class Artifacts:
            preprocessor_file = "p.joblib"
            model_file = "m.joblib"
            final_results_file = "r.csv"

        artifacts = Artifacts()

    orchestrator = PipelineOrchestrator(MockConfig())

    def fail_fn():
        raise ValueError("Something went wrong")

    result = orchestrator.run_stage("failing-stage", fail_fn)

    assert result.stage == "failing-stage"
    assert result.success is False
    assert "Something went wrong" in result.detail
