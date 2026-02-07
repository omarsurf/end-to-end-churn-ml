# Threshold Selection Policy

## Goal
Define how the churn decision threshold is selected and validated.

## Selection Rule (Validation Set)
- Optimize: `Net_Value`
- Constraints:
  - `Recall >= 0.70`
  - `Precision >= 0.50`
- Search grid:
  - `threshold_min = 0.20`
  - `threshold_max = 0.85`
  - `threshold_step = 0.05`

## Notebook 10 vs Script
- Notebook 10: exploration and reporting workflow.
- Script (`churn-evaluate`): production evaluation workflow.
- Both use the same selection logic (maximize `Net_Value` with recall/precision constraints).
- If thresholds differ, it is usually because they were run on different artifacts/timestamps (model/data/results files).
- Production reference is the latest scripted output in `models/final_test_results.csv`.
- Notebook exports are saved separately (`models/final_test_results_notebook.csv`, `models/threshold_analysis_val_notebook.csv`) to avoid overwriting production artifacts.

## Quality Gates (Test Set)
- `ROC-AUC >= 0.83`
- `Recall >= 0.70`
- `Precision >= 0.45`

## Current Reference Result
- Selected threshold: `0.45`
- Selection reason: `Best Net_Value on validation (Recall >= 0.70 & Precision >= 0.50)`

## Artifact Files
- Production/scripted: `models/threshold_analysis_val.csv`, `models/final_test_results.csv`
- Notebook exports: `models/threshold_analysis_val_notebook.csv`, `models/final_test_results_notebook.csv`
