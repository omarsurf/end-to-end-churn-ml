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

---

## Production Status

### Current State: Batch-Ready

| Capability | Status |
|------------|--------|
| Batch inference | Ready |
| Model versioning | Ready |
| Quality gates | Ready |
| Drift detection | Ready |
| CI/CD pipeline | Ready |
| Test coverage | 89% |

### Roadmap: Real-Time Deployment

The following items are planned to extend this project for real-time inference:

| Item | Priority | Description |
|------|----------|-------------|
| **FastAPI wrapper** | High | REST API endpoints for single/batch predictions |
| **Dockerfile** | High | Container image for consistent deployment |
| **Health endpoint** | Medium | `/health` and `/ready` for orchestrators |
| **Async processing** | Medium | Queue-based inference for high throughput |
| **Cloud storage** | Medium | S3/GCS for model artifact backup |
| **Drift alerting** | Low |  Webhook/email when drift detected |

#### Proposed API Design

```
POST /predict          → Single customer prediction
POST /predict/batch    → Batch predictions (JSON array)
GET  /model/info       → Current production model metadata
GET  /health           → Service health status
GET  /metrics          → Prometheus metrics
```

#### Proposed Directory Additions

```
├── api/
│   ├── main.py        # FastAPI application
│   ├── routes.py      # Endpoint definitions
│   └── schemas.py     # Request/response models
├── Dockerfile
├── docker-compose.yml
└── k8s/               # Kubernetes manifests (optional)
```

> **Note**: The core ML logic (`predict.py`, `model_registry.py`, `monitoring.py`) is already designed for easy integration with an API layer.

---

## Author
**Omar Piro** - Machine Learning Engineer
