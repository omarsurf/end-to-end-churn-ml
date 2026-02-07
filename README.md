# Customer Churn Prediction (ML + MLOps)

Production-ready churn prediction pipeline with business decision framework.

## Features
- Config-driven CLI pipeline
- Model registry with promote/rollback
- Threshold optimization for business value
- Quality gates and drift detection

## Quick Start
```bash
pip install -e ".[dev]"

churn-prepare --config config/default.yaml --strict
churn-train --config config/default.yaml --strict
churn-evaluate --config config/default.yaml --target latest --strict
churn-predict --config config/default.yaml --input data/new.csv --output pred.csv
```

## Pipeline
| Stage | Command | Output |
|-------|---------|--------|
| Prepare | `churn-prepare` | preprocessor, train/val/test splits |
| Train | `churn-train` | model in registry |
| Evaluate | `churn-evaluate` | threshold, test metrics |
| Predict | `churn-predict` | predictions CSV |

## Business Parameters
| Parameter | Default |
|-----------|---------|
| CLV | $2,000 |
| Success rate | 30% |
| Contact cost | $50 |

## Model Management
```bash
churn-model-info                    # view current model
churn-model-promote --model-id <id> # promote to production
churn-model-rollback                # rollback
```

## MLOps Stack
- **DVC**: pipeline orchestration
- **MLflow**: experiment tracking
- **GitHub Actions**: CI (lint + tests)

## Project Structure
```
├── config/           # YAML configuration
├── data/raw/         # Source dataset
├── models/           # Trained models + registry
├── notebooks/        # Exploratory analysis
├── src/              # Production package
├── tests/            # Unit tests
└── docs/             # Documentation
```

## Docs
- [Production Guide](docs/PRODUCTION_GUIDE.md)
- [Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md)
- [Project Report](docs/PROJECT_REPORT.md)

## Author
**Omar Piro** - Machine Learning Engineer
