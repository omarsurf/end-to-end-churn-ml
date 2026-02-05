import numpy as np

from churn_ml_decision.evaluate import select_threshold, threshold_analysis


def test_select_threshold_prefers_precision_at_min_recall():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    thresholds = np.array([0.3, 0.5, 0.7])

    df = threshold_analysis(y_true, y_proba, thresholds)
    row, reason = select_threshold(df, min_recall=0.66)

    assert row["Recall"] >= 0.66
    assert "Recall >=" in reason


def test_select_threshold_fallback_to_f1():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.4, 0.3, 0.2, 0.1])
    thresholds = np.array([0.5, 0.6])

    df = threshold_analysis(y_true, y_proba, thresholds)
    row, reason = select_threshold(df, min_recall=0.9)

    assert reason == "Best F1 (fallback)"
    assert row["Threshold"] in thresholds
