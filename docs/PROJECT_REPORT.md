# Project Report

## Overview
Production-ready churn prediction pipeline from raw CSV to retention decisions.

## Pipeline
1. `churn-prepare`: validation, feature engineering, preprocessing artifacts.
2. `churn-train`: train candidates, select best model by ROC-AUC.
3. `churn-evaluate`: threshold search on validation, final metrics on test (`--target latest` by default).
4. `churn-predict`: batch scoring with validated inputs/outputs.

## Model and Threshold
- Default model: Logistic Regression
- Selection metric: ROC-AUC (validation)
- Threshold objective: maximize `Net_Value` on validation
- Threshold constraints: Recall `>= 0.70`, Precision `>= 0.50`

## Quality Gates (Test)
| Metric | Threshold |
|---|---|
| ROC-AUC | `>= 0.83` |
| Recall | `>= 0.70` |
| Precision | `>= 0.45` |

## Typical Results
- ROC-AUC: ~0.84
- Recall: ~0.83
- Precision: ~0.49–0.50
- Selected threshold: ~0.45
- Net Value: ~$154k (test set)

## Business Parameters (Default)
- `clv = 2000`
- `success_rate = 0.30`
- `contact_cost = 50`
- `retained_value = clv * success_rate = 600`

## Main Commands
```bash
churn-prepare --config config/default.yaml
churn-train --config config/default.yaml
churn-evaluate --config config/default.yaml
churn-predict --config config/default.yaml --input data/new.csv --output pred.csv
```

## Key Artifacts
- Model: `models/best_model.joblib`
- Threshold analysis: `models/threshold_analysis_val.csv`
- Final metrics: `models/final_test_results.csv`

## Related Docs
- `docs/THRESHOLD_ANALYSIS_NOTE.md`
- `config/default.yaml`
