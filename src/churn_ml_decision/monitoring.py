from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp


class DataDriftDetector:
    """Detect drift using two-sample Kolmogorov-Smirnov tests."""

    def __init__(self, *, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold
        self.reference_samples: dict[str, list[float]] = {}

    def fit(self, reference_data: pd.DataFrame) -> None:
        numeric_df = reference_data.select_dtypes(include=["number"]).copy()
        self.reference_samples = {
            col: numeric_df[col].dropna().astype(float).tolist() for col in numeric_df.columns
        }

    def detect_drift(self, new_data: pd.DataFrame) -> dict[str, Any]:
        if not self.reference_samples:
            raise ValueError("Drift detector is not fitted.")

        report: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "p_value_threshold": self.p_value_threshold,
            "drift_detected": False,
            "columns": {},
        }

        for col, reference_values in self.reference_samples.items():
            if col not in new_data.columns:
                report["columns"][col] = {"status": "MISSING_IN_NEW_DATA"}
                report["drift_detected"] = True
                continue

            current = pd.to_numeric(new_data[col], errors="coerce").dropna().astype(float)
            if current.empty:
                report["columns"][col] = {"status": "NO_VALID_VALUES"}
                report["drift_detected"] = True
                continue

            stat, p_value = ks_2samp(reference_values, current.tolist())
            drift = bool(p_value < self.p_value_threshold)
            report["columns"][col] = {
                "status": "DRIFT_DETECTED" if drift else "OK",
                "ks_statistic": float(stat),
                "p_value": float(p_value),
            }
            if drift:
                report["drift_detected"] = True

        return report

    def save(self, path: str | Path) -> None:
        payload = {
            "p_value_threshold": self.p_value_threshold,
            "reference_samples": self.reference_samples,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DataDriftDetector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        detector = cls(p_value_threshold=float(payload["p_value_threshold"]))
        detector.reference_samples = {
            col: [float(v) for v in values] for col, values in payload["reference_samples"].items()
        }
        return detector


class ProductionMetricsTracker:
    """Track operational metrics in a JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _default(self) -> dict[str, Any]:
        return {
            "updated_at": None,
            "prediction_batches": 0,
            "predictions_total": 0,
            "prediction_failures": 0,
            "failure_rate": 0.0,
            "avg_latency_ms": 0.0,
            "last_drift_score": None,
            "history": [],
        }

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update_prediction_metrics(
        self,
        *,
        batch_size: int,
        failed_rows: int,
        latency_ms: float,
        drift_score: float | None = None,
    ) -> dict[str, Any]:
        metrics = self.load()

        prev_batches = int(metrics["prediction_batches"])
        prev_avg = float(metrics["avg_latency_ms"])

        metrics["prediction_batches"] = prev_batches + 1
        metrics["predictions_total"] = int(metrics["predictions_total"]) + int(batch_size)
        metrics["prediction_failures"] = int(metrics["prediction_failures"]) + int(failed_rows)
        total = max(int(metrics["predictions_total"]), 1)
        metrics["failure_rate"] = float(metrics["prediction_failures"] / total)

        new_batches = int(metrics["prediction_batches"])
        metrics["avg_latency_ms"] = ((prev_avg * prev_batches) + float(latency_ms)) / new_batches
        metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
        if drift_score is not None:
            metrics["last_drift_score"] = float(drift_score)

        metrics["history"].append(
            {
                "timestamp": metrics["updated_at"],
                "batch_size": int(batch_size),
                "failed_rows": int(failed_rows),
                "latency_ms": float(latency_ms),
                "drift_score": drift_score,
            }
        )
        metrics["history"] = metrics["history"][-200:]
        self.save(metrics)
        return metrics

    def update_drift_metrics(self, *, drift_score: float) -> dict[str, Any]:
        """Persist latest drift score without mutating prediction counters."""
        metrics = self.load()
        metrics["last_drift_score"] = float(drift_score)
        metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save(metrics)
        return metrics
