import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from churn_ml_decision.train import (
    _extract_feature_importance,
    _load_feature_names,
    build_model,
    main as train_main,
)


def test_build_model_logistic_regression():
    candidate = {"type": "logistic_regression", "params": {"C": 1.0, "solver": "liblinear"}}
    model = build_model(candidate)
    assert isinstance(model, LogisticRegression)
    assert model.C == 1.0
    assert model.solver == "liblinear"


def test_build_model_default_params():
    candidate = {"type": "logistic_regression"}
    model = build_model(candidate)
    assert isinstance(model, LogisticRegression)


def test_build_model_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_model({"type": "random_forest", "params": {}})


def _make_synthetic_data(tmp_path: Path, n_train: int = 200, n_val: int = 50, n_features: int = 5):
    """Create synthetic .npy arrays and return (data_dir, models_dir)."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(n_train, n_features)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = rng.randn(n_val, n_features)
    y_val = (X_val[:, 0] > 0).astype(int)

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    np.save(data_dir / "X_train_processed.npy", X_train)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "X_val_processed.npy", X_val)
    np.save(data_dir / "y_val.npy", y_val)

    return data_dir, models_dir


def _write_train_config(tmp_path, data_dir, models_dir, candidates_yaml: str) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_raw: {tmp_path / 'data' / 'raw'}",
                f"  data_processed: {data_dir}",
                f"  models: {models_dir}",
                "model:",
                "  name: logistic_regression",
                "  version: 1",
                "  selection_metric: roc_auc",
                "  candidates:",
                candidates_yaml,
                "artifacts:",
                "  model_file: best_model.joblib",
                "tracking:",
                "  enabled: false",
                "registry:",
                "  enabled: false",
            ]
        )
    )
    return cfg_path


def test_train_selects_best_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, models_dir = _make_synthetic_data(tmp_path)
    candidates = "\n".join(
        [
            "    - name: lr_weak",
            "      type: logistic_regression",
            "      enabled: true",
            "      params:",
            "        C: 0.001",
            "        solver: liblinear",
            "    - name: lr_strong",
            "      type: logistic_regression",
            "      enabled: true",
            "      params:",
            "        C: 10.0",
            "        solver: liblinear",
        ]
    )
    cfg_path = _write_train_config(tmp_path, data_dir, models_dir, candidates)

    monkeypatch.setattr("sys.argv", ["churn-train", "--config", str(cfg_path)])
    train_main()

    assert (models_dir / "best_model.joblib").exists()
    summary = json.loads((models_dir / "train_summary.json").read_text())
    assert summary["selection_metric"] == "roc_auc"
    assert summary["validation_roc_auc"] > 0


def test_train_skips_disabled_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, models_dir = _make_synthetic_data(tmp_path)
    candidates = "\n".join(
        [
            "    - name: lr_enabled",
            "      type: logistic_regression",
            "      enabled: true",
            "      params:",
            "        C: 1.0",
            "        solver: liblinear",
            "    - name: lr_disabled",
            "      type: logistic_regression",
            "      enabled: false",
            "      params:",
            "        C: 10.0",
            "        solver: liblinear",
        ]
    )
    cfg_path = _write_train_config(tmp_path, data_dir, models_dir, candidates)

    monkeypatch.setattr("sys.argv", ["churn-train", "--config", str(cfg_path)])
    train_main()

    summary = json.loads((models_dir / "train_summary.json").read_text())
    assert summary["model_type"] == "lr_enabled"


def test_train_no_enabled_candidates_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, models_dir = _make_synthetic_data(tmp_path)
    candidates = "\n".join(
        [
            "    - name: lr_disabled",
            "      type: logistic_regression",
            "      enabled: false",
        ]
    )
    cfg_path = _write_train_config(tmp_path, data_dir, models_dir, candidates)

    monkeypatch.setattr("sys.argv", ["churn-train", "--config", str(cfg_path)])
    with pytest.raises(SystemExit, match="No enabled model candidates"):
        train_main()


def test_train_logs_to_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, models_dir = _make_synthetic_data(tmp_path)
    mlruns_dir = tmp_path / "mlruns"
    candidates = "\n".join(
        [
            "    - name: lr_test",
            "      type: logistic_regression",
            "      enabled: true",
            "      params:",
            "        C: 1.0",
            "        solver: liblinear",
        ]
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_raw: {tmp_path / 'data' / 'raw'}",
                f"  data_processed: {data_dir}",
                f"  models: {models_dir}",
                "model:",
                "  name: logistic_regression",
                "  version: 1",
                "  selection_metric: roc_auc",
                "  candidates:",
                candidates,
                "artifacts:",
                "  model_file: best_model.joblib",
                "tracking:",
                "  enabled: false",
                "registry:",
                "  enabled: false",
                "mlflow:",
                "  enabled: true",
                f"  tracking_uri: {mlruns_dir}",
                "  experiment_name: test-experiment",
                "  register_model: false",
            ]
        )
    )

    monkeypatch.setattr("sys.argv", ["churn-train", "--config", str(cfg_path)])
    train_main()

    assert mlruns_dir.exists()
    # Verify at least one experiment directory was created
    experiment_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name != ".trash"]
    assert len(experiment_dirs) >= 1


# =============================================================================
# Additional tests for helper functions
# =============================================================================


def test_load_feature_names_success(tmp_path: Path):
    """Test loading feature names from CSV."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    feature_file = models_dir / "final_feature_names.csv"
    feature_file.write_text("feature_name\ntenure\nMonthlyCharges\nTotalCharges\n")

    features = _load_feature_names(models_dir)

    assert features == ["tenure", "MonthlyCharges", "TotalCharges"]


def test_load_feature_names_missing_file(tmp_path: Path):
    """Test loading feature names returns empty list when file missing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    features = _load_feature_names(models_dir)

    assert features == []


def test_load_feature_names_wrong_column(tmp_path: Path):
    """Test loading feature names returns empty when column missing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    feature_file = models_dir / "final_feature_names.csv"
    feature_file.write_text("wrong_column\nval1\nval2\n")

    features = _load_feature_names(models_dir)

    assert features == []


def test_extract_feature_importance_coef():
    """Test extracting feature importance from logistic regression coef_."""
    model = LogisticRegression()
    # Simulate fitted model
    model.coef_ = np.array([[0.5, -0.3, 0.8]])
    model.classes_ = np.array([0, 1])

    features = ["f1", "f2", "f3"]
    importance = _extract_feature_importance(model, features)

    assert importance == {"f1": 0.5, "f2": 0.3, "f3": 0.8}


def test_extract_feature_importance_tree():
    """Test extracting feature importance from tree-based model."""

    class MockTreeModel:
        feature_importances_ = np.array([0.1, 0.5, 0.4])

    model = MockTreeModel()
    features = ["a", "b", "c"]
    importance = _extract_feature_importance(model, features)

    assert importance == {"a": 0.1, "b": 0.5, "c": 0.4}


def test_extract_feature_importance_no_features():
    """Test feature importance with no feature names generates defaults."""
    model = LogisticRegression()
    model.coef_ = np.array([[0.1, 0.2]])
    model.classes_ = np.array([0, 1])

    importance = _extract_feature_importance(model, [])

    assert importance == {"feature_0": 0.1, "feature_1": 0.2}


def test_extract_feature_importance_mismatched_size():
    """Test feature importance with mismatched feature name count."""
    model = LogisticRegression()
    model.coef_ = np.array([[0.1, 0.2, 0.3]])
    model.classes_ = np.array([0, 1])

    # Only 2 feature names but 3 coefs
    importance = _extract_feature_importance(model, ["f1", "f2"])

    assert len(importance) == 2
    assert list(importance.keys()) == ["f1", "f2"]


def test_extract_feature_importance_no_attr():
    """Test feature importance returns empty for model without coef_ or feature_importances_."""

    class MockModel:
        pass

    model = MockModel()
    importance = _extract_feature_importance(model, ["f1"])

    assert importance == {}
