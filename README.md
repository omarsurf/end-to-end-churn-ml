# Churn ML Decision

This project builds a churn prediction model and turns it into a business decision for retention targeting. The main workflow lives in notebooks, with artifacts saved to `data/processed` and `models`.

**Project Structure**
- `data/raw`: source dataset (Telco Customer Churn)
- `data/processed`: prepared splits and numpy arrays
- `models`: trained models, preprocessors, and evaluation outputs
- `notebooks`: end-to-end analysis and modeling
- `src`: production-ready package and CLI tools
- `config`: YAML configuration for training/evaluation

**Notebook Flow (Recommended Order)**
1. `01_business_context.ipynb`
2. `02_data_audit.ipynb`
3. `03_data_preparation.ipynb`
4. `04_EDA.ipynb`
5. `05_preprocessing.ipynb`
6. `06_baseline_model.ipynb`
7. `07_feature_importance._and_interpretation.ipynb`
8. `08_feature_engineering.ipynb`
9. `09_hyperparameter_tuning.ipynb`
10. `10_final_evaluation_and_business_decision.ipynb`
11. `11_model_improvement.ipynb` (optional, benchmark/experimentation)

**Quickstart**
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```
2. Run the notebooks in the order above.

Optional advanced dependencies:
```bash
pip install -e ".[ml]"
```

**Production CLI (Config-Driven)**
The default config is `config/default.yaml`.
```bash
churn-prepare --config config/default.yaml
churn-train --config config/default.yaml
churn-evaluate --config config/default.yaml
churn-predict --config config/default.yaml --input data/new_customers.csv --output data/predictions.csv
```

**Make Targets**
```bash
make install
make prepare
make train
make evaluate
make pipeline
make test
make lint
```

**Repeatable Evaluation (No Notebook Required)**
After artifacts exist in `data/processed` and `models`, you can also run:
```bash
python src/evaluate.py --min-recall 0.70 --threshold-min 0.20 --threshold-max 0.85 --threshold-step 0.05
```

This script will:
- choose a decision threshold on the validation set
- evaluate once on the test set
- write results to `models/threshold_analysis_val.csv`
- write results to `models/final_test_results.csv`

**Notes**
- Advanced notebooks require `xgboost`, `lightgbm`, and `shap`.
- The CLI pipeline uses a stable Logistic Regression configuration by default (`l2`, `class_weight=balanced`). If you prefer tuned params from notebook 09 (e.g. `l1`/`saga`), update `config/default.yaml`.
- The final decision threshold should be selected on validation, then tested once on the test set.
- `data/processed` and intermediate artifacts are not tracked; regenerate them via notebooks or the pipeline.
- Business value metrics (EV/ROI) are computed during evaluation using `business` config values.
- Each train/evaluate run logs a record to `models/experiments.jsonl` when tracking is enabled.
- The current model is recorded in `models/registry.json` when registry is enabled.
- Portfolio summary: `docs/PORTFOLIO_REPORT.md`
