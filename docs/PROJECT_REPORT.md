# Project Report

## Project Overview

Production-ready **churn prediction pipeline** that transforms customer data into actionable retention targeting decisions. Features end-to-end ML workflow with full MLOps capabilities.

## Key Highlights

| Metric | Value |
|--------|-------|
| **Test Coverage** | 89% |
| **Production Modules** | 17 |
| **Test Files** | 20 |
| **CLI Commands** | 10 |
| **Notebooks** | 11 |

## Dataset

- **Source**: IBM Telco Customer Churn (Kaggle)
- **Size**: 7,043 rows, 21 columns
- **Target**: Binary churn prediction (Yes/No)
- **Class Distribution**: ~27% churn rate

## Architecture

```
Raw Data (CSV)
     │
     ▼
churn-prepare ─────────────────────────────────────────┐
  • Config validation (Pydantic)                       │
  • Data quality checks                                │
  • Feature engineering (14+ features)                 │
  • Preprocessing pipeline                             │
  • Drift reference export                             │
     │                                                 │
     ▼                                                 │
Processed Arrays + Preprocessor                        │
     │                                                 │
     ▼                                                 │
churn-train ───────────────────────────────────────────┤
  • Candidate training (LR, XGB, LGBM)                 │
  • Best-model selection (ROC-AUC)                     │
  • MLflow logging                                     │
  • Registry registration                              │
     │                                                 │
     ▼                                                 │
Model Artifacts + Registry                             │
     │                                                 │
     ▼                                                 │
churn-evaluate ────────────────────────────────────────┤
  • Threshold optimization (recall-constrained)        │
  • Test set evaluation                                │
  • Quality gates enforcement                          │
  • Business value metrics                             │
     │                                                 │
     ▼                                                 │
churn-predict ─────────────────────────────────────────┘
  • Input validation
  • Registry-based model resolution
  • Batch inference
  • Production metrics
```

## MLOps Stack

| Tool | Purpose | Status |
|------|---------|--------|
| **DVC** | Pipeline orchestration & reproducibility | ✅ Active |
| **MLflow** | Experiment tracking & model logging | ✅ Active |
| **Pydantic** | Config & data validation | ✅ Active |
| **Model Registry** | JSON-based version control | ✅ Active |
| **GitHub Actions** | CI/CD pipeline | ✅ Active |
| **Quality Gates** | Configurable metric thresholds | ✅ Active |
| **Drift Detection** | KS-test based monitoring | ✅ Active |

## Production Modules

| Module | Coverage | Purpose |
|--------|----------|---------|
| `prepare.py` | 93% | Data prep & feature engineering |
| `train.py` | 81% | Model training & selection |
| `evaluate.py` | 86% | Threshold optimization & metrics |
| `predict.py` | 82% | Batch inference |
| `config.py` | 92% | Configuration management |
| `cli.py` | 98% | Command-line interface |
| `model_registry.py` | 97% | Model versioning |
| `monitoring.py` | 90% | Drift detection |
| `validators.py` | 85% | Data validation |
| `schemas.py` | 83% | Pydantic schemas |
| `io.py` | 100% | File I/O utilities |
| `track.py` | 100% | Experiment logging |
| `registry.py` | 100% | Registry utilities |
| `pipeline.py` | 100% | Pipeline orchestration |
| `exceptions.py` | 100% | Custom exceptions |
| `logging_config.py` | 100% | Logging setup |
| `mlflow_utils.py` | 80% | MLflow integration |

## Modeling

### Features
- **14+ engineered features**: tenure groups, charge ratios, interaction terms
- **7 categorical features**: Contract, InternetService, PaymentMethod, etc.
- **Preprocessing**: StandardScaler + OneHotEncoder pipeline

### Model Candidates
| Model | Status | Description |
|-------|--------|-------------|
| Logistic Regression | ✅ Default | L2 regularization, balanced weights |
| XGBoost | Optional | Gradient boosting (400 estimators) |
| LightGBM | Optional | Light gradient boosting |

### Selection Metric
- **ROC-AUC** on validation set
- Threshold optimized for **recall ≥ 0.70** with max precision

## Business Decision Framework

Expected value calculation using configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clv` | $2,000 | Customer Lifetime Value |
| `success_rate` | 30% | Retention campaign success rate |
| `contact_cost` | $50 | Cost per retention contact |

**Output metrics**:
- `Net_Value`: Total expected profit from campaign
- `Net_per_Flagged`: ROI per customer contacted
- Threshold sweep: 0.20 to 0.85 (step 0.05)

## Quality Gates

| Metric | Threshold | Enforcement |
|--------|-----------|-------------|
| ROC-AUC | ≥ 0.83 | Pipeline fails if not met |
| Recall | ≥ 0.70 | Pipeline fails if not met |
| Precision | ≥ 0.50 | Pipeline fails if not met |

## Typical Results

With default configuration:
- **ROC-AUC**: ~0.84
- **Recall**: ~0.78 (at optimized threshold)
- **Precision**: ~0.52
- **Optimal Threshold**: ~0.35

## CLI Commands

```bash
# Pipeline
churn-prepare --config config/default.yaml
churn-train --config config/default.yaml
churn-evaluate --config config/default.yaml
churn-predict --config config/default.yaml --input data/new.csv --output pred.csv

# Registry
churn-model-info --config config/default.yaml
churn-model-promote --config config/default.yaml --model-id <id>
churn-model-rollback --config config/default.yaml

# Monitoring
churn-check-drift --config config/default.yaml --input data/batch.csv
churn-health-check --config config/default.yaml
churn-validate-config --config config/default.yaml
```

## Testing

```bash
# Run all tests
make test

# With coverage report
python -m pytest --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

## Project Links

- **Documentation**: `docs/`
- **Configuration**: `config/default.yaml`
- **Experiments**: `mlflow ui --backend-store-uri mlruns`
- **Reports**: `models/final_test_results.csv`

## Limitations

- Single-node training (no distributed computing)
- Local MLflow backend (no remote server)
- JSON-based registry (not production database)

## Future Improvements

1. Remote MLflow tracking server
2. Docker containerization
3. REST API for real-time predictions
4. A/B testing framework
5. Enhanced drift monitoring with alerting
