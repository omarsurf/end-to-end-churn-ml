# Production Guide

## Architecture

```text
Raw Data (CSV)
   |
   v
churn-prepare
  - Config validation (Pydantic)
  - Data quality checks
  - Feature engineering + preprocessing
  - Drift reference export
   |
   v
Processed Arrays + Preprocessor + Reports
   |
   v
churn-train
  - Candidate training
  - Best-model selection
  - Registry metadata registration
   |
   v
Model Artifacts + Registry
   |
   v
churn-evaluate
  - Threshold optimization
  - Test evaluation + quality gates
  - Registry status update
   |
   v
churn-predict
  - Input validation
  - Registry-based model resolution
  - Output validation
  - Production metrics update
```

## Pipeline Flow (Inputs -> Outputs)

- `churn-prepare`:
  - Input: `paths.data_raw`
  - Output: `data/processed/*.npy`, `models/preprocessor.joblib`, `models/data_quality_report.json`
- `churn-train`:
  - Input: processed arrays
  - Output: model artifact, `models/train_summary.json`, `models/registry.json`
- `churn-evaluate`:
  - Input: validation/test arrays + model
  - Output: `models/threshold_analysis_val.csv`, `models/final_test_results.csv`
- `churn-predict`:
  - Input: customer batch CSV
  - Output: predictions CSV + `metrics/production_metrics.json`

## Configuration Tuning Guide

- `business.*`: impacts expected value and targeting economics.
- `quality.*`: controls release gates (`roc_auc`, `recall`, `precision`).
- `validation.*`: controls ingestion strictness (missing/duplicates/ranges).
- `evaluation.*`: controls threshold search grid.
- `monitoring.*`: controls drift sensitivity and metric export paths.

Environment overrides use prefix `CHURN__`, e.g.:

```bash
export CHURN__BUSINESS__SUCCESS_RATE=0.4
export CHURN__QUALITY__MIN_RECALL=0.75
```

## Troubleshooting

- `ConfigValidationError`:
  - Run `churn-validate-config --config config/default.yaml`
  - Fix invalid ranges/types in YAML or environment overrides.
- `DataValidationError` in `churn-prepare`:
  - Inspect `models/data_quality_report.json`.
  - Fix source data (missing columns, invalid ranges, class distribution).
- `ModelNotFoundError` in predict/evaluate:
  - Check `models/registry.json`.
  - Promote a model: `churn-model-promote --model-id <id>`.
- Predictions all near `0.5`:
  - Check logs for prediction fallback due to transform/model inference errors.

## Emergency Procedures

- Show production model:
  - `churn-model-info --config config/default.yaml`
- Promote tested model:
  - `churn-model-promote --model-id <model_id> --config config/default.yaml`
- Rollback immediately:
  - `churn-model-rollback --config config/default.yaml`
- Validate platform health:
  - `churn-health-check --config config/default.yaml`
