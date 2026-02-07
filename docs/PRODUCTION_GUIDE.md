# Production Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Configuration                             │
│                     config/default.yaml                          │
│  (paths, model, features, business, quality, validation, etc.)  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      churn-prepare                               │
│                                                                  │
│  • Load raw CSV from paths.data_raw                             │
│  • Validate schema (required columns, types, ranges)            │
│  • Quality checks (missing, duplicates, target distribution)    │
│  • Feature engineering (14+ derived features)                   │
│  • Train/val/test split                                         │
│  • Fit preprocessing pipeline (StandardScaler + OneHotEncoder) │
│  • Export drift reference statistics                            │
│                                                                  │
│  Outputs:                                                        │
│  ├── data/processed/X_train_processed.npy, y_train.npy          │
│  ├── data/processed/X_val_processed.npy, y_val.npy              │
│  ├── data/processed/X_test_processed.npy, y_test.npy            │
│  ├── models/preprocessor.joblib                                 │
│  ├── models/train_medians.json                                  │
│  ├── models/data_quality_report.json                            │
│  └── models/drift_reference.json                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       churn-train                                │
│                                                                  │
│  • Load processed arrays                                        │
│  • Train each enabled candidate (LR, XGB, LGBM)                 │
│  • Evaluate on validation set                                   │
│  • Select best by model.selection_metric                        │
│  • Log to MLflow (if enabled)                                   │
│  • Register in model registry                                   │
│                                                                  │
│  Outputs:                                                        │
│  ├── models/best_model.joblib                                   │
│  ├── models/train_summary.json                                  │
│  ├── models/registry.json                                       │
│  ├── models/experiments.jsonl                                   │
│  └── mlruns/                                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      churn-evaluate                              │
│                                                                  │
│  • Load model target (`--target latest|production|local`)       │
│    - default: `latest` (most recent registered candidate)       │
│  • Threshold sweep on validation set                            │
│  • Select threshold (maximize Net_Value under constraints)       │
│  • Final evaluation on test set                                 │
│  • Compute business value metrics                               │
│  • Enforce quality gates                                        │
│  • Update registry status                                       │
│                                                                  │
│  Outputs:                                                        │
│  ├── models/threshold_analysis_val.csv                          │
│  └── models/final_test_results.csv                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      churn-predict                               │
│                                                                  │
│  • Validate input CSV                                           │
│  • Load production model from registry                          │
│  • Apply preprocessing                                          │
│  • Generate predictions                                         │
│  • Update production metrics                                    │
│                                                                  │
│  Outputs:                                                        │
│  ├── predictions.csv                                            │
│  └── metrics/production_metrics.json                            │
└─────────────────────────────────────────────────────────────────┘
```

## Module Reference

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.py` | 239 | Configuration loading, Pydantic models, env overrides |
| `prepare.py` | 162 | Data prep, feature engineering, preprocessing |
| `train.py` | 145 | Model training, candidate selection, registry |
| `evaluate.py` | 168 | Threshold optimization, test evaluation, business metrics |
| `predict.py` | 142 | Batch inference, input validation |
| `cli.py` | 97 | CLI entry points for all commands |
| `model_registry.py` | 155 | Version control, promote/rollback |
| `validators.py` | 124 | Data validation rules |
| `schemas.py` | 110 | Pydantic schemas for I/O |
| `monitoring.py` | 73 | Drift detection (KS-test) |
| `mlflow_utils.py` | 76 | MLflow integration |
| `logging_config.py` | 35 | Structured logging setup |
| `io.py` | 32 | File I/O utilities |
| `track.py` | 18 | Experiment tracking (JSONL) |
| `pipeline.py` | 20 | Pipeline orchestration |
| `registry.py` | 17 | Registry utilities |
| `exceptions.py` | 6 | Custom exceptions |

## Configuration Reference

### Paths
```yaml
paths:
  data_raw: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
  data_processed: data/processed
  models: models
```

### Model
```yaml
model:
  name: logistic_regression
  version: 1
  selection_metric: roc_auc
  candidates:
    - name: logistic_regression
      enabled: true
      params: { C: 1.0, penalty: l2, ... }
    - name: xgboost
      enabled: false
    - name: lightgbm
      enabled: false
```

### Business
```yaml
business:
  clv: 2000              # Customer lifetime value ($)
  success_rate: 0.30     # Retention campaign success rate
  contact_cost: 50       # Cost per contact ($)
```

### Quality Gates
```yaml
quality:
  min_roc_auc: 0.83
  min_recall: 0.70
  min_precision: 0.45
```

### Validation
```yaml
validation:
  required_columns: [tenure, MonthlyCharges, ...]
  max_missing_ratio: 0.05
  max_duplicate_ratio: 0.02
  min_target_rate: 0.05
  max_target_rate: 0.95
```

### Environment Overrides
```bash
export CHURN__BUSINESS__CLV=3000
export CHURN__QUALITY__MIN_RECALL=0.75
export CHURN__VALIDATION__MAX_MISSING_RATIO=0.10
```

## Troubleshooting

### ConfigValidationError
```bash
churn-validate-config --config config/default.yaml
```
- Check YAML syntax
- Verify numeric ranges are valid
- Check environment override format

### DataValidationError in prepare
```bash
# Inspect quality report
cat models/data_quality_report.json | python -m json.tool
```
- Missing required columns → check source data
- Invalid numeric ranges → check tenure, charges are non-negative
- Class imbalance → target rate outside [0.05, 0.95]

### ModelNotFoundError in predict/evaluate
```bash
# Check registry
churn-model-info --config config/default.yaml
```
- No model registered → run `churn-train` first
- Model file missing → check `models/` directory
- Promote a model: `churn-model-promote --model-id <id>`

### Quality Gate Failures
```bash
# Check final results
cat models/final_test_results.csv
```
- Adjust thresholds in `config/default.yaml:quality`
- Review model selection metric
- Check for data drift

### Predictions Near 0.5
- Check logs for preprocessing errors
- Verify feature names match training
- Check for missing values in input

## Emergency Procedures

### View Current Production Model
```bash
churn-model-info --config config/default.yaml
```

### Promote New Model
```bash
churn-model-promote --model-id <model_id> --config config/default.yaml
```

### Evaluate Target Selection
```bash
# Default: evaluate latest registered candidate (pre-promotion validation)
churn-evaluate --config config/default.yaml --target latest

# Evaluate active production model (monitoring/comparison)
churn-evaluate --config config/default.yaml --target production

# Evaluate local canonical artifact only (bypass registry routing)
churn-evaluate --config config/default.yaml --target local
```

### Immediate Rollback
```bash
churn-model-rollback --config config/default.yaml
```

### Platform Health Check
```bash
churn-health-check --config config/default.yaml
```

### Check for Data Drift
```bash
churn-check-drift --config config/default.yaml --input data/new_batch.csv
```

## Logging

Logs are written to `logs/pipeline.log`:
```bash
tail -f logs/pipeline.log
```

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

Configure in `config/default.yaml:logging.level`

## Monitoring Outputs

| File | Description |
|------|-------------|
| `metrics/production_metrics.json` | Prediction statistics |
| `metrics/data_drift_report.json` | Drift test results |
| `models/data_quality_report.json` | Data quality summary |
| `models/registry.json` | Model version history |
| `models/experiments.jsonl` | Experiment logs |
