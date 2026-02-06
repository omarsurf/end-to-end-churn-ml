# End-to-End Customer Churn Prediction (ML + MLOps)

## Overview
This project builds an **end-to-end machine learning system for customer churn prediction** and translates model outputs into **business retention decisions**.

It combines:
- an exploratory, notebook-driven analysis
- a **production-ready, config-driven CLI pipeline**
- lightweight **MLOps practices** (CI, tracking, reproducibility, quality gates)

The goal is not only predictive performance, but **reliable and interpretable decision-making**.

---

## Dataset
- **Source**: IBM Telco Customer Churn (Kaggle)
- **Size**: 7,043 rows × 21 features
- **Target**: Binary churn label (`Yes` / `No`)

---

## Project Structure
```
end-to-end-churn-ml/
├── .github/workflows/        # CI pipeline (lint, tests)
├── config/                   # YAML configs (training, evaluation)
├── data/
│   ├── raw/                  # Source dataset
│   └── processed/            # Generated artifacts (not tracked)
├── models/                   # Trained models & evaluation outputs
├── notebooks/                # End-to-end analysis (recommended flow)
├── src/churn_ml_decision/    # Production package & CLI
├── tests/                    # Unit tests
├── docs/                     # Portfolio report & documentation
├── dvc.yaml                  # DVC pipeline definition
├── pyproject.toml
└── README.md
```

---

## Notebook Flow (Recommended Order)
1. `01_business_context.ipynb`
2. `02_data_audit.ipynb`
3. `03_data_preparation.ipynb`
4. `04_EDA.ipynb`
5. `05_preprocessing.ipynb`
6. `06_baseline_model.ipynb`
7. `07_feature_importance_and_interpretation.ipynb`
8. `08_feature_engineering.ipynb`
9. `09_hyperparameter_tuning.ipynb`
10. `10_final_evaluation_and_business_decision.ipynb`
11. `11_model_improvement.ipynb` *(optional)*

---

## Production Pipeline (CLI)
The pipeline is fully **config-driven** via `config/default.yaml`.

| Stage     | Command            | Description |
|----------|--------------------|-------------|
| Prepare  | `churn-prepare`    | Clean data, split train/val/test, build features |
| Train    | `churn-train`      | Train models and log metrics |
| Evaluate | `churn-evaluate`   | Threshold selection & business evaluation |
| Predict  | `churn-predict`    | Batch inference on new customers |

Example:
```bash
churn-prepare  --config config/default.yaml
churn-train    --config config/default.yaml
churn-evaluate --config config/default.yaml
churn-predict  --config config/default.yaml \
               --input data/new_customers.csv \
               --output data/predictions.csv
```

---

## MLOps Stack
| Tool            | Purpose                                  |
|-----------------|------------------------------------------|
| **DVC**         | Pipeline orchestration & parameter tracking |
| **MLflow**      | Experiment tracking & artifact logging   |
| **GitHub Actions** | CI (lint + tests)                     |
| **Ruff**        | Fast Python linting                      |
| **Config Gates**| Min ROC-AUC / recall constraints         |

---

## Modeling Highlights
- **Baseline**: Logistic Regression (interpretable, class_weight=balanced)
- **Optional candidates**: XGBoost, LightGBM (disabled by default)
- **Feature engineering**: Non-linear transformations & interactions
- **Threshold selection**: Recall-constrained optimization on validation
- **Final evaluation**: Single pass on test set (no leakage)

---

## Business Decision Framework
Expected value (EV) is computed using configurable assumptions:

| Parameter        | Default | Description |
|------------------|---------|-------------|
| `clv`            | $2,000  | Customer lifetime value |
| `success_rate`   | 30%     | Retention success rate |
| `contact_cost`   | $50     | Cost per contact |

Outputs:
- `Net_Value`
- `Net_per_Flagged`
- Optimal decision threshold

Results are saved to:
- `models/threshold_analysis_val.csv`
- `models/final_test_results.csv`

---

## Quality & Testing
- **Linting**: Ruff (zero warnings enforced)
- **CI**: Automated on every commit
- **Unit tests**: Core pipeline components
- **Reproducibility**: Config + deterministic splits

---

## Quickstart
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e ".[dev,ops]"

# Run full pipeline
make pipeline   # or: dvc repro

# Run tests
make test

# Track experiments
mlflow ui --backend-store-uri mlruns
```

---

## Portfolio Note
A full project write-up is available in:
```
docs/PROJECT_REPORT.md
```

This repository is designed to demonstrate **production-oriented ML engineering**, not just notebook experimentation.

---

## Author
**Omar Piro**  
Machine Learning Engineer
