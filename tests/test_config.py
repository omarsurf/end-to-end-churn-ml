from pathlib import Path

from churn_ml_decision.config import load_config, project_root


def test_default_config_loads():
    cfg_path = project_root() / "config" / "default.yaml"
    cfg = load_config(cfg_path)

    assert "paths" in cfg
    assert "model" in cfg
    assert "evaluation" in cfg
    assert "artifacts" in cfg
    assert "split" in cfg
    assert "features" in cfg
    assert "tracking" in cfg
    assert "quality" in cfg
    assert "registry" in cfg


def test_paths_are_relative():
    cfg_path = project_root() / "config" / "default.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg["paths"]["data_processed"], str)
    assert isinstance(cfg["paths"]["models"], str)
