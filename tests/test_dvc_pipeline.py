"""Validate dvc.yaml pipeline definition is consistent with project structure."""

import pytest
import yaml

from churn_ml_decision.config import project_root


@pytest.fixture
def dvc_config():
    dvc_path = project_root() / "dvc.yaml"
    if not dvc_path.exists():
        pytest.skip("dvc.yaml not found")
    return yaml.safe_load(dvc_path.read_text())


def test_dvc_yaml_has_three_stages(dvc_config):
    stages = dvc_config["stages"]
    assert "prepare" in stages
    assert "train" in stages
    assert "evaluate" in stages


def test_dvc_stages_use_cli_commands(dvc_config):
    stages = dvc_config["stages"]
    assert "churn-prepare" in stages["prepare"]["cmd"]
    assert "churn-train" in stages["train"]["cmd"]
    assert "churn-evaluate" in stages["evaluate"]["cmd"]


def test_dvc_train_depends_on_prepare_outputs(dvc_config):
    prepare_outs = set(dvc_config["stages"]["prepare"]["outs"])
    train_deps = set(dvc_config["stages"]["train"]["deps"])
    data_deps = {d for d in train_deps if d.startswith("data/processed/")}
    data_outs = {o for o in prepare_outs if o.startswith("data/processed/")}
    assert data_deps.issubset(data_outs)
