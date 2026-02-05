**Portfolio Report**

**Overview**
This project builds a churn prediction model and turns it into a business decision for retention targeting. It includes an end-to-end notebook narrative and a production-ready CLI pipeline.

**Dataset**
- Telco Customer Churn (public dataset)
- 7,043 rows, 21 columns

**Pipeline (Production-Ready)**
1. `churn-prepare`: clean data, split train/val/test, engineer features, and build a preprocessing pipeline.
2. `churn-train`: train a Logistic Regression model and log validation metrics.
3. `churn-evaluate`: select a threshold on validation, evaluate once on test, and compute business value.

**Modeling Highlights**
- Interpretable Logistic Regression baseline
- Feature engineering for non-linear effects and interactions
- Threshold selection based on recall and precision trade-off

**Business Decision**
The evaluation computes expected value (EV) using configurable business assumptions:
- Retained value per churner
- Cost per retention contact

This produces `Net_Value` and `Net_per_Flagged` to guide threshold choice.

**Latest Scripted Results**
See `models/final_test_results.csv` for the most recent run.

**Reproducibility**
```bash
make prepare
make train
make evaluate
```

**Quality Signals**
- CLI pipeline with YAML config
- Tests and CI
- Clear separation of notebooks vs production code

**Limitations**
- Feature engineering is kept minimal and interpretable
- No MLOps tracking system beyond local JSONL logs

**Next Steps**
1. Add MLflow or DVC for experiment tracking
2. Add performance gate tests on CI (min metrics)
3. Extend model comparison beyond LR in production pipeline
