# Deployment Checklist

## Pre-Deployment Validation

### Code Quality
- [ ] `python -m ruff check src tests` passes (zero warnings)
- [ ] `python -m pytest` passes (88% coverage target)
- [ ] `churn-validate-config --config config/default.yaml` succeeds

### Pipeline Execution
- [ ] `churn-prepare --config config/default.yaml --strict` succeeds
- [ ] `churn-train --config config/default.yaml --strict` succeeds
- [ ] `churn-evaluate --config config/default.yaml --strict` succeeds

### Quality Gates
Verify `models/final_test_results.csv` meets thresholds:
- [ ] ROC-AUC ≥ 0.83
- [ ] Recall ≥ 0.70
- [ ] Precision ≥ 0.45

### Registry Status
- [ ] Model registered in `models/registry.json`
- [ ] Model file exists at specified path
- [ ] Train summary available in `models/train_summary.json`

## Deployment Steps

### 1. Confirm Current State
```bash
churn-model-info --config config/default.yaml
```
- [ ] Note current production model ID
- [ ] Record current model metrics for comparison

### 2. Promote New Model
```bash
churn-model-promote --model-id <release_model_id> --config config/default.yaml
```
- [ ] Promotion succeeds
- [ ] Registry updated with new current model

### 3. Smoke Test
```bash
churn-predict --config config/default.yaml \
  --input data/new_customers.csv \
  --output data/predictions.csv
```
- [ ] Prediction completes without errors
- [ ] Output file generated with valid probabilities

### 4. Drift Check
```bash
churn-check-drift --config config/default.yaml --input data/new_customers.csv
```
- [ ] Input follows raw scoring schema (the command applies training-consistent feature engineering before drift comparison)
- [ ] No significant drift detected (p-value > 0.05 for all features)
- [ ] Or: drift documented and accepted

## Post-Deployment Validation

### Health Check
```bash
churn-health-check --config config/default.yaml
```
- [ ] Status: healthy

### Logs Review
```bash
tail -100 logs/pipeline.log | grep -E "(ERROR|CRITICAL)"
```
- [ ] No unexpected errors

### Metrics Verification
- [ ] `metrics/production_metrics.json` updated after predictions
- [ ] Prediction distribution within expected range

### Output Validation
- [ ] Predictions in valid range [0, 1]
- [ ] No null/NaN values in output
- [ ] Correct number of rows (matches input)

## Rollback Procedure

### 1. Identify Previous Model
```bash
churn-model-info --config config/default.yaml
```
Note the previous stable model ID from history.

### 2. Execute Rollback
```bash
# Quick rollback to previous
churn-model-rollback --config config/default.yaml

# Or explicit target
churn-model-rollback --model-id <stable_id> --config config/default.yaml
```
- [ ] Rollback succeeds

### 3. Verify Rollback
```bash
churn-model-info --config config/default.yaml
```
- [ ] Current model is the previous stable version

### 4. Smoke Test After Rollback
```bash
churn-predict --config config/default.yaml \
  --input data/new_customers.csv \
  --output data/predictions_rollback.csv
```
- [ ] Predictions complete successfully

### 5. Health Check After Rollback
```bash
churn-health-check --config config/default.yaml
```
- [ ] Status: healthy

### 6. Document Incident
- [ ] Record timestamp of rollback
- [ ] Document root cause
- [ ] Create follow-up ticket for investigation

## Quick Reference

| Action | Command |
|--------|---------|
| Validate config | `churn-validate-config --config config/default.yaml` |
| Run tests | `make test` |
| Full pipeline | `make pipeline` |
| Model info | `churn-model-info --config config/default.yaml` |
| Promote | `churn-model-promote --model-id <id> --config config/default.yaml` |
| Rollback | `churn-model-rollback --config config/default.yaml` |
| Health check | `churn-health-check --config config/default.yaml` |
| Drift check | `churn-check-drift --config config/default.yaml --input <file>` |
| View logs | `tail -f logs/pipeline.log` |

## Monitoring Checklist (Ongoing)

Daily:
- [ ] Review `logs/pipeline.log` for errors
- [ ] Check prediction volumes in `metrics/production_metrics.json`

Weekly:
- [ ] Run drift check on recent batch
- [ ] Compare prediction distribution to baseline

Monthly:
- [ ] Review model performance metrics
- [ ] Evaluate need for retraining
- [ ] Audit registry history
