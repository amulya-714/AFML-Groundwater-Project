# Groundwater ML Baseline Handoff

## Status
The baseline model is complete. The metrics below were verified from
`results/baseline_metrics.csv`.

## Dataset and split
- Training: observations before 2018-01-01
- Testing: observations from 2018-01-01 onward
- Test rows: 14,863
- Features: latitude, longitude, 30-day rainfall, 90-day rainfall,
  previous groundwater level, month sine, and month cosine

## Results

| Model | Test rows | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|---:|
| Persistence baseline | 14,863 | 2.601288 | 4.438926 | 0.628011 |
| Random Forest | 14,863 | 1.990118 | 3.497733 | 0.769034 |

Random Forest reduced MAE by 23.5% and RMSE by 21.2% relative to persistence.

## Main files
- `src/baseline_model.py`
- `results/baseline_metrics.csv`
- `results/baseline_predictions.csv`
- `src/support_analysis.py`
- `src/full_support_analysis.py`
- `src/same_well_analysis.py`
- `src/paired_gwl_support.py`
- `src/paired_bootstrap.py`
- `src/bootstrap_gwl_support.py`

## Caveat
Support-analysis scripts have reported inconsistent training percentile
cutoffs. Standardize date parsing and the training split before treating
those analyses as final. The baseline metrics above are verified.

## Next step
Develop and evaluate a reliability-aware groundwater predictor with
prediction intervals and an unfamiliar-input indicator. Use the same
chronological test split. Evaluate interval coverage and width alongside
MAE and RMSE, and do not fit calibration on the final test set.
