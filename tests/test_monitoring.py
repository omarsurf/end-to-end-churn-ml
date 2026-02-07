from pathlib import Path

import pandas as pd

from churn_ml_decision.monitoring import DataDriftDetector, ProductionMetricsTracker


def test_drift_detector_detects_no_drift_for_similar_data():
    reference = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 11, 12, 13, 14]})
    current = pd.DataFrame({"x": [1.1, 2.2, 3.0, 4.1, 5.2], "y": [10, 11, 12, 13, 14]})

    detector = DataDriftDetector(p_value_threshold=0.01)
    detector.fit(reference)
    report = detector.detect_drift(current)

    assert report["drift_detected"] is False
    assert "x" in report["columns"]


def test_metrics_tracker_updates_file(tmp_path: Path):
    tracker = ProductionMetricsTracker(tmp_path / "metrics.json")
    payload = tracker.update_prediction_metrics(batch_size=10, failed_rows=2, latency_ms=120.0)

    assert payload["prediction_batches"] == 1
    assert payload["predictions_total"] == 10
    assert payload["prediction_failures"] == 2
    assert (tmp_path / "metrics.json").exists()


def test_metrics_tracker_updates_drift_score_without_prediction_counters(tmp_path: Path):
    tracker = ProductionMetricsTracker(tmp_path / "metrics.json")
    tracker.update_prediction_metrics(batch_size=10, failed_rows=1, latency_ms=100.0)
    payload = tracker.update_drift_metrics(drift_score=0.42)

    assert payload["last_drift_score"] == 0.42
    assert payload["prediction_batches"] == 1
    assert payload["predictions_total"] == 10
    assert payload["prediction_failures"] == 1
