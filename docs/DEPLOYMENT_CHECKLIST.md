# Deployment Checklist

## Pre-Deployment

### Tests
- [ ] `pytest` passes
- [ ] `ruff check src tests` passes

### Pipeline
- [ ] `churn-prepare --strict` succeeds
- [ ] `churn-train --strict` succeeds
- [ ] `churn-evaluate --target latest --strict` succeeds

### Quality Gates
Check `models/final_test_results.csv`:
- [ ] ROC-AUC >= 0.83
- [ ] Recall >= 0.70
- [ ] Precision >= 0.45 (relaxed from 0.50 validation constraint to allow val→test variance)

## Deployment

### 1. Check Current State
```bash
churn-model-info
```

### 2. Promote Model
```bash
churn-model-promote --model-id <id>
```

### 3. Smoke Test
```bash
churn-predict --input data/test.csv --output /tmp/pred.csv
```

### 4. Drift Check
```bash
churn-check-drift --input data/new_batch.csv
```

## Rollback

### Quick Rollback (artifact present)
```bash
# 1. Identify stable model
churn-model-info

# 2. Rollback to previous or specific model
churn-model-rollback
# or: churn-model-rollback --model-id <stable_id>

# 3. Verify
churn-health-check
churn-predict --input data/test.csv --output /tmp/smoke.csv
```

### Full Rollback (artifact missing)
If the timestamped model file is missing locally:
```bash
# 1. Restore registry state from release tag
git checkout <release_tag> -- models/registry.json

# 2. Restore timestamped model from archive
cp /path/to/model-archive/<model_file>.joblib models/

# 3. Verify
churn-model-info
churn-health-check
```

**Note:** Timestamped models are not tracked by DVC. Maintain a model archive for disaster recovery.

## Quick Reference
| Action | Command |
|--------|---------|
| Model info | `churn-model-info` |
| Promote | `churn-model-promote --model-id <id>` |
| Rollback | `churn-model-rollback` |
| Health | `churn-health-check` |
| Drift | `churn-check-drift --input <file>` |
