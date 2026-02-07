import numpy as np

from churn_ml_decision.evaluate import select_threshold, threshold_analysis


def test_select_threshold_prefers_precision_at_min_recall():
    """When optimize_for=precision, pick best precision among high-recall rows."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    thresholds = np.array([0.3, 0.5, 0.7])

    df = threshold_analysis(y_true, y_proba, thresholds)
    row, reason = select_threshold(df, min_recall=0.66, optimize_for="precision")

    assert row["Recall"] >= 0.66
    assert "Precision" in reason


def test_select_threshold_fallback_to_f1():
    """When no row meets min_recall, fallback to best F1."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.4, 0.3, 0.2, 0.1])
    thresholds = np.array([0.5, 0.6])

    df = threshold_analysis(y_true, y_proba, thresholds)
    row, reason = select_threshold(df, min_recall=0.9)

    assert "Best F1" in reason
    assert row["Threshold"] in thresholds


def test_select_threshold_net_value_optimization():
    """Default optimize_for=net_value selects most profitable threshold."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    thresholds = np.array([0.3, 0.5, 0.7])

    df = threshold_analysis(y_true, y_proba, thresholds, retained_value=600, contact_cost=50)
    row, reason = select_threshold(df, min_recall=0.66, optimize_for="net_value")

    assert row["Recall"] >= 0.66
    assert "Net_Value" in reason
    # Should pick threshold with highest Net_Value among those meeting recall
    high_recall = df[df["Recall"] >= 0.66]
    expected_threshold = high_recall.loc[high_recall["Net_Value"].idxmax(), "Threshold"]
    assert row["Threshold"] == expected_threshold


def test_select_threshold_with_min_precision_constraint():
    """min_precision filters out low-precision thresholds before optimization."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "Threshold": [0.2, 0.35, 0.45, 0.6],
            "Recall": [0.97, 0.92, 0.86, 0.74],
            "Precision": [0.39, 0.44, 0.50, 0.58],
            "F1_Score": [0.55, 0.60, 0.64, 0.65],
            "Net_Value": [170900, 168200, 161250, 141300],
        }
    )
    # Without min_precision: selects 0.2 (highest Net_Value)
    row_no_constraint, _ = select_threshold(
        df, min_recall=0.70, optimize_for="net_value", min_precision=0.0
    )
    assert row_no_constraint["Threshold"] == 0.2

    # With min_precision=0.40: filters out 0.2, selects 0.35 (next highest Net_Value)
    row_with_constraint, reason = select_threshold(
        df, min_recall=0.70, optimize_for="net_value", min_precision=0.40
    )
    assert row_with_constraint["Threshold"] == 0.35
    assert "Precision >= 0.40" in reason
    assert "Net_Value" in reason
