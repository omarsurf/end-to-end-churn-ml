from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from churn_ml_decision.mlflow_utils import _flatten_dict, is_available, start_run


def test_flatten_dict():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    assert _flatten_dict(nested) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_flatten_dict_empty():
    assert _flatten_dict({}) == {}


def test_is_available():
    assert is_available() is True


def test_start_run_disabled():
    cfg = {"mlflow": {"enabled": False}}
    with start_run(cfg, run_name="test") as run:
        assert run is None


def test_start_run_missing_config():
    with start_run({}, run_name="test") as run:
        assert run is None


def test_start_run_creates_run(tmp_path: Path):
    cfg = {
        "mlflow": {
            "enabled": True,
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "unit-test",
        }
    }
    with start_run(cfg, run_name="test-run") as run:
        assert run is not None
    assert (tmp_path / "mlruns").exists()
