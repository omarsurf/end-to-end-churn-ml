# Project Report

## Overview
Production-ready churn prediction pipeline from raw CSV to retention decisions.

## Pipeline
1. `churn-prepare`: validation, feature engineering, preprocessing artifacts
2. `churn-train`: train candidates, select best model by ROC-AUC, register in registry
3. `churn-evaluate`: threshold optimization on validation, final metrics on test
4. `churn-predict`: batch scoring using production model from registry

## Model
- Type: Logistic Regression (L1/saga)
- Selection metric: ROC-AUC
- Threshold objective: maximize `Net_Value` under constraints (Recall >= 0.70, Precision >= 0.50)

## Quality Gates (Test)
| Metric | Threshold | Note |
|--------|-----------|------|
| ROC-AUC | >= 0.83 | |
| Recall | >= 0.70 | |
| Precision | >= 0.45 | Relaxed from 0.50 selection constraint |

See [THRESHOLD_ANALYSIS_NOTE.md](THRESHOLD_ANALYSIS_NOTE.md) for policy rationale.

## Business Parameters
| Parameter | Value |
|-----------|-------|
| CLV | $2,000 |
| Success rate | 30% |
| Contact cost | $50 |
| Retained value | $600 |

## Commands
```bash
churn-prepare --config config/default.yaml --strict
churn-train --config config/default.yaml --strict
churn-evaluate --config config/default.yaml --target latest --strict
churn-predict --config config/default.yaml --input data/new.csv --output pred.csv
```

## Key Artifacts
| Artifact | Path |
|----------|------|
| Production model | `models/registry.json` → model path |
| Preprocessor | `models/preprocessor.joblib` |
| Threshold analysis | `models/threshold_analysis_val.csv` |
| Test results | `models/final_test_results.csv` |

## Related Docs
- [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)
- [THRESHOLD_ANALYSIS_NOTE.md](THRESHOLD_ANALYSIS_NOTE.md)
