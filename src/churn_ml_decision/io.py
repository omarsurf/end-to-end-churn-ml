from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_train_val_arrays(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(data_dir / "X_train_processed.npy")
    y_train = np.load(data_dir / "y_train.npy")
    x_val = np.load(data_dir / "X_val_processed.npy")
    y_val = np.load(data_dir / "y_val.npy")

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Train sample mismatch.")
    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError("Validation sample mismatch.")
    if np.isnan(x_train).sum() > 0 or np.isnan(x_val).sum() > 0:
        raise ValueError("Found NaN values in processed train/val arrays.")

    return x_train, y_train, x_val, y_val


def load_val_arrays(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    x_val = np.load(data_dir / "X_val_processed.npy")
    y_val = np.load(data_dir / "y_val.npy")

    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError("Validation sample mismatch.")
    if np.isnan(x_val).sum() > 0:
        raise ValueError("Found NaN values in processed validation array.")

    return x_val, y_val


def load_test_arrays(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    x_test = np.load(data_dir / "X_test_processed.npy")
    y_test = np.load(data_dir / "y_test.npy")

    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test sample mismatch.")
    if np.isnan(x_test).sum() > 0:
        raise ValueError("Found NaN values in processed test array.")

    return x_test, y_test
