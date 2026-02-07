# Production Guide

## Pipeline Flow

```
config/default.yaml
        │
        ▼
┌─────────────────┐
│  churn-prepare  │  → preprocessor.joblib, train_medians.json, X/y splits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  churn-train    │  → model registered in registry.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  churn-evaluate │  → threshold_analysis_val.csv, final_test_results.csv
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  churn-predict  │  → predictions.csv (uses production model from registry)
└─────────────────┘
```

## Key Modules
| Module | Purpose |
|--------|---------|
| `prepare.py` | Data validation, feature engineering, preprocessing |
| `train.py` | Model training, registry registration |
| `evaluate.py` | Threshold optimization, quality gates |
| `predict.py` | Batch inference (production model only, no silent fallback) |
| `model_registry.py` | Model versioning, promote/rollback |
| `monitoring.py` | Drift detection |

## Configuration

### Business
```yaml
business:
  clv: 2000
  success_rate: 0.30
  contact_cost: 50
```

### Quality Gates
```yaml
quality:
  min_roc_auc: 0.83
  min_recall: 0.70
  min_precision: 0.45
```

### Environment Overrides
```bash
export CHURN__BUSINESS__CLV=3000
export CHURN__QUALITY__MIN_RECALL=0.75
```

## Prediction Behavior
- **Production**: requires `registry.enabled=true` + promoted model
- **Dev/test**: use `--allow-unregistered` flag (logs warning)
- **No silent fallback**: fails explicitly if no production model

## Troubleshooting

### No production model
```bash
churn-model-info  # check registry
churn-model-promote --model-id <id>  # promote
```

### Quality gate failure
```bash
cat models/final_test_results.csv  # check metrics
```

### Drift detected
```bash
churn-check-drift --input data/new.csv
```

### MLflow file backend deprecation (February 2026)
- Current config uses `mlflow.tracking_uri: mlruns` (filesystem backend).
- This backend is deprecated by MLflow and already emits `FutureWarning` during tests/runs.
- Recommended action: migrate tracking to a DB backend (for example `sqlite:///mlflow.db`) before February 2026.
- Update `config/default.yaml` (or `CHURN__MLFLOW__TRACKING_URI`) to avoid future breakage.

## Model Versioning

### Artifact Strategy
| Artifact | Purpose | Tracking |
|----------|---------|----------|
| `*_vN_TIMESTAMP.joblib` | Immutable production model | Local only (gitignored) |
| `best_model.joblib` | DVC-tracked alias | DVC (for CI/notebooks) |
| `registry.json` | Source of truth | Git (points to timestamped files) |

**Note:** Timestamped files are NOT tracked by DVC. They must be preserved locally or archived manually for rollback.

### Release Workflow
```bash
# 1. Train and validate
churn-prepare --strict && churn-train --strict && churn-evaluate --target latest --strict

# 2. Archive timestamped model (before it gets overwritten)
cp models/*_TIMESTAMP.joblib /path/to/model-archive/

# 3. Tag release
git add models/registry.json
git commit -m "Release model vX.Y.Z"
git tag -a vX.Y.Z -m "Model release vX.Y.Z"
git push origin vX.Y.Z

# 4. Push DVC artifacts
dvc push
```

### Rollback

**Quick rollback** (timestamped artifact still present locally):
```bash
churn-model-rollback --model-id <stable_id>
churn-health-check
```

**Full rollback** (artifact missing, need to restore from archive):
```bash
# 1. Restore registry state
git checkout <release_tag> -- models/registry.json

# 2. Restore timestamped model from archive
cp /path/to/model-archive/<model_file>.joblib models/

# 3. Verify
churn-model-info
churn-health-check
```

## Emergency
```bash
churn-model-rollback  # rollback to previous
churn-health-check    # verify status
```
