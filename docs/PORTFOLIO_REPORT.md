# Project Report

## Overview

This project builds a **churn prediction model** and turns it into a **business decision** for retention targeting. It includes an end-to-end notebook narrative and a production-ready CLI pipeline with full MLOps capabilities.

## Dataset

- **Source**: IBM Telco Customer Churn (Kaggle)
- **Size**: 7,043 rows, 21 columns
- **Target**: Binary churn prediction (Yes/No)

## Pipeline (Production-Ready)

The CLI pipeline is config-driven (`config/default.yaml`) and fully reproducible:

| Stage | Command | Description |
|-------|---------|-------------|
| **Prepare** | `churn-prepare` | Clean data, split train/val/test, engineer features, build preprocessing pipeline |
| **Train** | `churn-train` | Train model candidates, select best by validation metric, log to MLflow |
| **Evaluate** | `churn-evaluate` | Select threshold on validation, evaluate on test, compute business value |
| **Predict** | `churn-predict` | Batch inference on new data |

## MLOps Stack

| Tool | Purpose | Status |
|------|---------|--------|
| **DVC** | Pipeline orchestration & parameter tracking | Enabled |
| **MLflow** | Experiment tracking, model logging, artifact storage | Enabled |
| **JSONL Tracking** | Lightweight local experiment logs | Enabled |
| **Model Registry** | Simple JSON-based model versioning | Enabled |
| **GitHub Actions** | CI/CD with lint, pipeline, and tests | Active |
| **Quality Gates** | Configurable min ROC-AUC, recall, precision | Enforced |

### DVC Pipeline

```bash
dvc repro        # Run full pipeline
dvc dag          # Visualize pipeline DAG
dvc params diff  # Compare parameter changes
```

### MLflow Tracking

```bash
mlflow ui --backend-store-uri mlruns
```

Tracks: model parameters, validation/test metrics, artifacts, and model files.

## Modeling Highlights

- **Baseline**: Interpretable Logistic Regression with L2 regularization
- **Candidates**: XGBoost, LightGBM (configurable, disabled by default)
- **Feature Engineering**: 14+ engineered features for non-linear effects and interactions
- **Threshold Selection**: Recall-constrained precision optimization on validation set
- **Model Selection Metric**: ROC-AUC (configurable)

## Business Decision Framework

The evaluation computes expected value (EV) using configurable business assumptions:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clv` | $2,000 | Customer Lifetime Value |
| `success_rate` | 30% | Retention campaign success rate |
| `contact_cost` | $50 | Cost per retention contact |

**Output metrics**:
- `Net_Value`: Total expected profit from retention campaign
- `Net_per_Flagged`: ROI per customer contacted

## Quality & Testing

| Metric | Value |
|--------|-------|
| **Test Coverage** | 90% |
| **Linting** | ruff (zero warnings) |
| **CI Pipeline** | lint + full pipeline + tests |

Key modules coverage:
- `prepare.py`: 99%
- `evaluate.py`: 94%
- `io.py`, `track.py`, `registry.py`: 100%

## Latest Results

See `models/final_test_results.csv` for the most recent evaluation.

Typical results with default config:
- ROC-AUC: ~0.84
- Recall: ~0.78 (at selected threshold)
- Precision: ~0.52

## Quickstart

```bash
# Install
pip install -e ".[dev,ops]"

# Run pipeline
make pipeline    # or: dvc repro

# Run tests
make test

# View experiments
mlflow ui --backend-store-uri mlruns
```

## Project Structure

```
churn_ml_decision/
├── config/default.yaml     # Central configuration
├── data/
│   ├── raw/                # Source dataset
│   └── processed/          # Train/val/test arrays
├── models/                 # Trained models & artifacts
├── notebooks/              # 11 exploratory notebooks
├── src/churn_ml_decision/  # Production package
├── tests/                  # Unit & integration tests
├── dvc.yaml                # DVC pipeline definition
└── .github/workflows/      # CI configuration
```

## Limitations

- Feature engineering kept minimal for interpretability
- Single-node training (no distributed)
- Local MLflow backend (no remote server)

## Future Improvements

1. Add remote MLflow tracking server
2. Implement model A/B testing framework
3. Add data drift monitoring
4. Containerize with Docker for deployment
5. Add API endpoint for real-time predictions
