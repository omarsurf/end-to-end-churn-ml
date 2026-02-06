from pathlib import Path

import pytest
import yaml

from churn_ml_decision.config import ChurnConfig, load_config, load_typed_config, project_root
from churn_ml_decision.exceptions import ConfigValidationError


def test_default_config_loads_as_typed_model():
    cfg_path = project_root() / "config" / "default.yaml"
    cfg = load_typed_config(cfg_path)

    assert isinstance(cfg, ChurnConfig)
    assert cfg.business.clv > 0
    assert 0 <= cfg.business.success_rate <= 1
    assert cfg.split.random_state >= 0


def test_default_config_loads_as_dict_for_backward_compatibility():
    cfg_path = project_root() / "config" / "default.yaml"
    cfg = load_config(cfg_path)

    assert "paths" in cfg
    assert "model" in cfg
    assert "evaluation" in cfg
    assert "validation" in cfg
    assert isinstance(cfg["paths"]["data_processed"], str)


def test_config_validation_fails_on_invalid_rate(tmp_path: Path):
    cfg = {
        "business": {
            "clv": 2000,
            "success_rate": 1.5,
            "contact_cost": 50,
        }
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ConfigValidationError):
        load_typed_config(path)


def test_config_env_override(monkeypatch: pytest.MonkeyPatch):
    cfg_path = project_root() / "config" / "default.yaml"
    monkeypatch.setenv("CHURN__BUSINESS__SUCCESS_RATE", "0.45")
    cfg = load_typed_config(cfg_path)
    assert cfg.business.success_rate == pytest.approx(0.45)


def test_paths_are_relative():
    cfg_path = project_root() / "config" / "default.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg["paths"]["data_processed"], str)
    assert isinstance(cfg["paths"]["models"], str)
