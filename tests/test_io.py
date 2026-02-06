from pathlib import Path

import numpy as np
import pytest

from churn_ml_decision.io import load_test_arrays, load_train_val_arrays, load_val_arrays


def _save_arrays(data_dir: Path, prefix: str, X: np.ndarray, y: np.ndarray):
    np.save(data_dir / f"X_{prefix}_processed.npy", X)
    np.save(data_dir / f"y_{prefix}.npy", y)


# ---------- load_train_val_arrays ----------


def test_load_train_val_arrays_ok(tmp_path: Path):
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    _save_arrays(tmp_path, "train", X, y)
    _save_arrays(tmp_path, "val", X, y)

    x_train, y_train, x_val, y_val = load_train_val_arrays(tmp_path)
    assert x_train.shape == (2, 2)
    assert y_val.shape == (2,)


def test_load_train_val_arrays_train_shape_mismatch(tmp_path: Path):
    _save_arrays(tmp_path, "train", np.ones((3, 2)), np.ones(2))
    _save_arrays(tmp_path, "val", np.ones((2, 2)), np.ones(2))

    with pytest.raises(ValueError, match="Train sample mismatch"):
        load_train_val_arrays(tmp_path)


def test_load_train_val_arrays_val_shape_mismatch(tmp_path: Path):
    _save_arrays(tmp_path, "train", np.ones((2, 2)), np.ones(2))
    _save_arrays(tmp_path, "val", np.ones((3, 2)), np.ones(2))

    with pytest.raises(ValueError, match="Validation sample mismatch"):
        load_train_val_arrays(tmp_path)


def test_load_train_val_arrays_nan_in_train(tmp_path: Path):
    X_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
    _save_arrays(tmp_path, "train", X_nan, np.array([0, 1]))
    _save_arrays(tmp_path, "val", np.ones((2, 2)), np.ones(2))

    with pytest.raises(ValueError, match="NaN"):
        load_train_val_arrays(tmp_path)


def test_load_train_val_arrays_nan_in_val(tmp_path: Path):
    _save_arrays(tmp_path, "train", np.ones((2, 2)), np.ones(2))
    X_nan = np.array([[np.nan, 2.0], [3.0, 4.0]])
    _save_arrays(tmp_path, "val", X_nan, np.array([0, 1]))

    with pytest.raises(ValueError, match="NaN"):
        load_train_val_arrays(tmp_path)


# ---------- load_val_arrays ----------


def test_load_val_arrays_ok(tmp_path: Path):
    _save_arrays(tmp_path, "val", np.ones((4, 3)), np.ones(4))
    x_val, y_val = load_val_arrays(tmp_path)
    assert x_val.shape == (4, 3)


def test_load_val_arrays_shape_mismatch(tmp_path: Path):
    _save_arrays(tmp_path, "val", np.ones((4, 3)), np.ones(3))
    with pytest.raises(ValueError, match="Validation sample mismatch"):
        load_val_arrays(tmp_path)


def test_load_val_arrays_nan(tmp_path: Path):
    _save_arrays(tmp_path, "val", np.array([[np.nan]]), np.array([1]))
    with pytest.raises(ValueError, match="NaN"):
        load_val_arrays(tmp_path)


# ---------- load_test_arrays ----------


def test_load_test_arrays_ok(tmp_path: Path):
    _save_arrays(tmp_path, "test", np.ones((5, 2)), np.ones(5))
    x_test, y_test = load_test_arrays(tmp_path)
    assert x_test.shape == (5, 2)


def test_load_test_arrays_shape_mismatch(tmp_path: Path):
    _save_arrays(tmp_path, "test", np.ones((5, 2)), np.ones(3))
    with pytest.raises(ValueError, match="Test sample mismatch"):
        load_test_arrays(tmp_path)


def test_load_test_arrays_nan(tmp_path: Path):
    _save_arrays(tmp_path, "test", np.array([[np.nan, 1.0]]), np.array([0]))
    with pytest.raises(ValueError, match="NaN"):
        load_test_arrays(tmp_path)
