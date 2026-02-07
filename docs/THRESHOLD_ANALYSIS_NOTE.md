# Threshold Selection Policy

## Selection Rule (Validation Set)
- **Objective**: maximize `Net_Value`
- **Constraints**: Recall >= 0.70, Precision >= 0.50
- **Grid**: 0.20 to 0.85, step 0.05

## Quality Gates (Test Set)
- ROC-AUC >= 0.83
- Recall >= 0.70
- Precision >= 0.45

## Why Precision 0.50 (selection) vs 0.45 (gate)?

| Stage | Precision | Purpose |
|-------|-----------|---------|
| Validation selection | >= 0.50 | Strict constraint during threshold optimization |
| Test quality gate | >= 0.45 | Relaxed by 10% to absorb val→test variance |

This split prevents false deployment failures from normal statistical variance between validation and test sets, while maintaining a high bar during model development.

## Artifacts
- `models/threshold_analysis_val.csv`: validation sweep results
- `models/final_test_results.csv`: test set metrics + selected threshold
