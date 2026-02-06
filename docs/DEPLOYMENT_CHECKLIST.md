# Deployment Checklist

## Pre-Deployment

- [ ] `churn-validate-config --config config/default.yaml` passes.
- [ ] `python -m pytest -q` passes.
- [ ] `python -m ruff check src tests` passes.
- [ ] `churn-prepare --config config/default.yaml --strict` succeeds.
- [ ] `churn-train --config config/default.yaml --strict` succeeds.
- [ ] `churn-evaluate --config config/default.yaml --strict` succeeds.
- [ ] Quality gates in `models/final_test_results.csv` meet `config/default.yaml:quality`.
- [ ] Registry contains candidate model for promotion.

## Deployment Steps

- [ ] Confirm current prod model: `churn-model-info --config config/default.yaml`.
- [ ] Promote release model:
  - `churn-model-promote --model-id <release_model_id> --config config/default.yaml`
- [ ] Run smoke batch:
  - `churn-predict --config config/default.yaml --input data/new_customers.csv --output data/predictions.csv`
- [ ] Validate drift on latest batch:
  - `churn-check-drift --config config/default.yaml --input data/new_customers.csv`

## Post-Deployment Validation

- [ ] `churn-health-check --config config/default.yaml` reports healthy.
- [ ] `metrics/production_metrics.json` is updated after predictions.
- [ ] No unexpected `ERROR/CRITICAL` entries in `logs/pipeline.log`.
- [ ] Prediction outputs include valid probabilities in `[0, 1]`.

## Rollback Procedure

- [ ] Identify previous stable model ID from `churn-model-info`.
- [ ] Roll back:
  - `churn-model-rollback --config config/default.yaml`
  - or explicit target: `churn-model-rollback --model-id <stable_id> --config config/default.yaml`
- [ ] Re-run smoke prediction and health check.
- [ ] Record incident summary + root cause in ops notes.
