import numpy as np

from churn_ml_decision.evaluate import check_quality_gates, threshold_analysis


def test_threshold_analysis_computes_business_metrics():
    y_true = np.array([1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.7, 0.4, 0.2, 0.1])
    thresholds = np.array([0.5])

    df = threshold_analysis(y_true, y_proba, thresholds, retained_value=600.0, contact_cost=50.0)

    assert "Net_Value" in df.columns
    assert "Net_per_Flagged" in df.columns

    row = df.iloc[0]
    # At threshold 0.5: predictions are [1, 1, 0, 0, 0]
    assert row["True_Positives"] == 2
    assert row["False_Positives"] == 0
    # Net = TP * retained - flagged * cost = 2*600 - 2*50 = 1100
    assert row["Net_Value"] == 2 * 600.0 - 2 * 50.0


def test_threshold_analysis_without_business_metrics():
    y_true = np.array([1, 0])
    y_proba = np.array([0.8, 0.2])
    thresholds = np.array([0.5])

    df = threshold_analysis(y_true, y_proba, thresholds)

    assert "Net_Value" not in df.columns
    assert "Precision" in df.columns
    assert "Recall" in df.columns


def test_threshold_analysis_multiple_thresholds():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.4, 0.1])
    thresholds = np.array([0.3, 0.5, 0.7])

    df = threshold_analysis(y_true, y_proba, thresholds)

    assert len(df) == 3
    # Lower threshold = more flagged = higher recall
    assert df.iloc[0]["Recall"] >= df.iloc[2]["Recall"]


def test_quality_gates_all_pass():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    failures = check_quality_gates(roc_auc=0.88, recall=0.75, precision=0.55, quality=quality)
    assert failures == []


def test_quality_gates_all_fail():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    failures = check_quality_gates(roc_auc=0.70, recall=0.50, precision=0.30, quality=quality)
    assert set(failures) == {"roc_auc", "recall", "precision"}


def test_quality_gates_partial_fail():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    failures = check_quality_gates(roc_auc=0.90, recall=0.60, precision=0.55, quality=quality)
    assert failures == ["recall"]


def test_quality_gates_boundary_values():
    quality = {"min_roc_auc": 0.83, "min_recall": 0.70, "min_precision": 0.50}
    # Exactly at threshold should pass (not strictly less than)
    failures = check_quality_gates(roc_auc=0.83, recall=0.70, precision=0.50, quality=quality)
    assert failures == []
