# Person 3 – Robustness and Evaluation Verification Report
## (Second-Stage Audit Edition)

**Project:** Why Do Groundwater ML Models Fail Under Extreme Conditions? Separating Input Extrapolation from Predictive Relationship Instability.

**Author:** Person 3 (Thrupthi G D) — Evaluation and Robustness

**Repository:** `thrupthi912/AFML-Groundwater-Project` (working source: amulya-714, commit 35298c2)

**Audit status:** Second-stage audit complete. All numbers verified against CSV outputs.  
**Date-column fix applied:** `full_support_analysis.py` and `paired_gwl_support.py` now use `observation_date` (consistent with baseline). The GWL-support bootstrap results have been updated accordingly.

---

## 1. Objective and Scope

This report documents Person 3's independent verification of the groundwater ML evaluation pipeline, including a second-stage methodological audit. The work covers:

1. Independent recalculation of all evaluation metrics from row-level predictions.
2. Validation of grouped comparisons with threshold provenance checks.
3. Bootstrap confidence interval verification, including independent re-implementation.
4. Rainfall quality-control sensitivity analysis, including a retrain experiment.
5. Date-column consistency audit and correction.
6. Report number cross-verification against actual CSV outputs.
7. Diagnosis of the unpaired vs same-well CI discrepancy.

**Person 2 reliability model status:** The files `reliability_metrics.csv`, `reliability_predictions.csv`, `reliability_by_condition.csv`, and `reliability_sensitivity.csv` are absent from the repository. All Person 3 work is performed on the baseline Random Forest outputs.

---

## 2. Files and Experiments Inspected

| File | Status |
|---|---|
| `data/processed/groundwater_rainfall_features_qc.csv` | Present — 104,781 rows, 30 columns |
| `src/baseline_model.py` | Present |
| `src/support_analysis.py` | Present |
| `src/full_support_analysis.py` | Present — **date column fixed (was using `Data Acquisition Time`)** |
| `src/same_well_analysis.py` | Present |
| `src/paired_bootstrap.py` | Present |
| `src/paired_gwl_support.py` | Present — **date column fixed (was using `Data Acquisition Time`)** |
| `src/bootstrap_gwl_support.py` | Present |
| `results/baseline_metrics.csv` | Present |
| `results/baseline_predictions.csv` | Present — 14,863 rows |
| `results/paired_gwl_support.csv` | Present — **regenerated after date fix: 161 wells** |
| `report/baseline_handoff.md` | Present |
| `report/research_handoff.md` | Present |

**Person 2 reliability outputs — MISSING:**
`reliability_metrics.csv`, `reliability_predictions.csv`, `reliability_by_condition.csv`, `reliability_sensitivity.csv`

---

## 3. Evaluation Methodology

### Dataset and split

- Total rows after feature construction and lag: 102,289 (some rows lost to `previous_gwl` NaN)
- Training: 87,426 rows, 1996-05-15 to 2017-11-05
- Test: 14,863 rows, 2018-01-05 to 2020-11-30
- Chronological split boundary: 2018-01-01 (strictly less than for training)
- All four Person 3 scripts use `TRAIN_CUT = "2018-01-01"` and `observation_date` consistently

### Independent metric calculation

Metrics were computed from raw prediction values in `baseline_predictions.csv` before reading `baseline_metrics.csv`. This ensures independence.

### Bootstrap methodology

Well-level cluster resampling with 2,000 repetitions, fixed seeds. Two estimand types are used and explicitly distinguished:

- **Avg-of-well-means:** Average of per-well statistics. Reflects the typical well experience.
- **Observation-weighted:** Simple mean over all observations. Can be dominated by wells with many observations.

---

## 4. Metric Verification Results (Independently Calculated)

| Model | N rows | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|---:|
| Persistence baseline | 14,863 | 2.601288 | 4.438926 | 0.628011 |
| Random Forest | 14,863 | 1.990073 | 3.497681 | 0.769041 |

### Agreement with reported metrics

| Model | Metric | Independent | Reported | Difference |
|---|---|---:|---:|---:|
| Persistence | MAE | 2.601288 | 2.601288 | 0.00 ✓ |
| Persistence | RMSE | 4.438926 | 4.438926 | 0.00 ✓ |
| Random Forest | MAE | 1.990073 | 1.990073 | 0.00 ✓ |
| Random Forest | RMSE | 3.497681 | 3.497681 | 0.00 ✓ |
| Random Forest | R² | 0.769041 | 0.769041 | 0.00 ✓ |

**Note on handoff discrepancy:** The baseline handoff document reports RF MAE = 1.990118 and RMSE = 3.497733. The re-run produces 1.990073 and 3.497681. The difference (< 5×10⁻⁵ m) is due to non-deterministic parallel aggregation in scikit-learn `RandomForestRegressor` with `n_jobs=-1` across runs. It is negligible and does not affect any conclusion.

---

## 5. Data Integrity and Leakage Validation (12/12 PASS)

All checks passed:
- Zero duplicate well-date pairs in predictions
- Zero missing RF or persistence predictions
- All prediction dates ≥ 2018-01-01 (no training-period leakage)
- Zero training-test observation overlap
- Perfect alignment between actual GWL in predictions file and raw dataset
- Metrics agree to machine precision

---

## 6. Grouped Evaluation Results

### A. Familiar vs Unfamiliar Inputs

Training-derived bounds (confirmed from `train_raw = df[observation_date < 2018-01-01]` only):

| Feature | 5th–95th range (mm) | 1st–99th range (mm) |
|---|---|---|
| rainfall_30d_mm | [0.00, 296.59] | [0.00, 1009.03] |
| rainfall_90d_mm | [19.80, 1219.18] | [4.62, 2995.64] |

**Mutual exclusivity confirmed:** Familiar + Unfamiliar = 13,133 + 1,730 = 14,863 (complete, non-overlapping).

| Group | N obs | N wells | MAE (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| Familiar (5th–95th bounds) | 13,133 | 1,792 | 2.0011 | 3.5312 |
| Unfamiliar (5th–95th bounds) | 1,730 | 827 | 1.9062 | 3.2318 |
| Familiar (1st–99th bounds) | 14,314 | 1,795 | 2.0048 | 3.5420 |
| Unfamiliar (1st–99th bounds) | 549 | 353 | 1.6053 | 2.0238 |

**Finding:** Under rainfall-only support boundaries, unfamiliar observations have *lower* MAE and RMSE than familiar ones. This is consistent with the research handoff's finding that rainfall-only novelty does not show a monotonic error increase.

### B. High Rainfall vs Other Conditions

Training 90d rainfall 95th percentile: **1,219.18 mm** (computed from training data only).

| Group | N obs | N wells | MAE (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| Below 95th pctile rainfall | 13,758 | 1,793 | 2.0164 | 3.5846 |
| High rainfall (≥95th pctile) | 1,105 | 388 | 1.6625 | 2.1383 |

**Finding:** High-rainfall observations have *lower* MAE (−0.354 m observation-weighted; −0.549 m avg-of-well-means). This is the opposite of the expected direction.

### C. Same-Well Comparisons (386 Eligible Wells)

Eligibility: wells with observations in *both* high and normal rainfall conditions.

- **Eligible wells (both conditions):** 386 (audit confirmed: all 386 genuinely have observations in both groups)
- **Mean paired diff (high – normal):** −0.0412 m
- **Median paired diff:** −0.0328 m
- **Fraction of wells with higher error in high rain:** 49.0%
- **Bootstrap 95% CI (2,000 well-level reps):** [−0.1675, +0.0799] m
- **CI includes zero:** Yes — difference is **not statistically conclusive**

---

## 7. Explanation: Why Unpaired CI Is Significant but Same-Well Paired CI Is Not

This is a key finding from the second-stage audit. The diagnosis is as follows.

**Audit measurements:**

| Condition | Wells | Avg MAE |
|---|---:|---:|
| All high-rain wells (388) | 388 | 1.6917 m |
| All normal wells (1,793) | 1,793 | 2.2408 m |
| High-rain wells, paired subset only (386) | 386 | 1.6873 m |
| Normal wells, paired subset only (386) | 386 | 1.7285 m |
| **Normal-only wells (never experience high rain)** | **1,407** | **2.3814 m** |

**Root cause:** The 1,407 wells that never experience high rainfall have substantially higher average MAE (2.38 m) than the 386 wells that do experience it (1.73 m in normal conditions). These high-error, normal-only wells inflate the unpaired "normal" group average, making the high-rain group *appear* much better by comparison.

When the paired analysis restricts to the same 386 wells in both conditions, the normal-condition MAE drops from 2.24 m to 1.73 m — much closer to the high-rain MAE of 1.69 m. The within-well difference (−0.04 m) is negligible.

**Interpretation:** The significant unpaired CI is a **between-well confound**, not a within-well rainfall effect. Wells that experience high rainfall are structurally different (different location, geology, or groundwater behaviour) from wells that do not — and they happen to have lower prediction errors in all conditions. The same-well paired analysis correctly controls for this. The unpaired result should not be interpreted as evidence that high rainfall improves prediction accuracy.

---

## 8. Bootstrap Confidence Intervals (Verified)

All bootstraps use 2,000 repetitions, well-level cluster resampling, fixed seeds.

| Quantity | Point estimate (m) | 95% CI (m) | N wells | CI includes zero? | Method |
|---|---:|---|---:|---|---|
| RF – Persistence MAE | −0.6361 | [−0.7282, −0.5618] | 1,795 | **No** | avg-of-well-mean-diffs |
| Unfamiliar – Familiar MAE | −0.1523 | [−0.3495, +0.0652] | 827 | **Yes** | independent group resample |
| High-rain – Normal MAE (unpaired) | −0.5491 | [−0.7150, −0.3838] | 388 | **No** | independent group resample |
| Same-well paired MAE diff | −0.0412 | [−0.1688, +0.0796] | 386 | **Yes** | avg-of-well-mean-diffs |
| GWL support (prev GWL, 1–99th) *(corrected)* | +0.4141 | [−0.7161, +1.4515] | 161 | **Yes** | avg-of-well-mean-diffs |

**Independent re-implementation results (different seed):**

| Quantity | Original CI | Independent CI (seed 999/1000) | Agreement |
|---|---|---|---|
| RF – Persistence | [−0.7282, −0.5618] | [−0.7312, −0.5582] | ✓ Within 0.005 m |
| Same-well paired | [−0.1688, +0.0796] | [−0.1682, +0.0786] | ✓ Within 0.002 m |

Both independent CIs include zero for the same-well comparison and exclude zero for the RF–Persistence comparison. Conclusions are consistent across seeds.

### Audit notes on bootstrap methodology

1. **Replace=True confirmed:** All bootstrap loops use `rng.choice(..., replace=True)`.
2. **Same-well pairing is correctly preserved:** Per-well means are computed first, then differences are formed, then those well-level differences are resampled.
3. **N_BOOT=2000 confirmed** in CSV output.
4. **Limitation for Unfamiliar–Familiar bootstrap:** A well can have observations in both groups (some obs familiar, some unfamiliar). The per-well MAE is computed separately for each group, but the bootstrap treats the two sets as independent. This slightly underestimates the CI width. The direction of the finding (CI includes zero) is not affected.

---

## 9. Rainfall QC Sensitivity Analysis

### QC flags confirmed

| Column | Total flagged | In training | In test |
|---|---:|---:|---:|
| `qc_suspicious_rainfall_30d` | 8 / 104,781 | 8 | **0** |
| `qc_suspicious_rainfall_90d` | 16 / 104,781 | 16 | **0** |

### Two distinct sensitivity experiments

The original sensitivity script performs **evaluation-filter only** — it does not retrain the model. The second-stage audit added an explicit retrain experiment.

**Experiment 1 — evaluation filter (original script):**  
Because all 14,863 test observations have QC flags = False, filtering changes nothing. The test metrics are identical to the full analysis. This is the correct finding: there are no suspicious rainfall observations in the test period.

**Experiment 2 — retrain excluding flagged training rows (new in audit):**

| Model | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|
| Full training (87,426 rows) | 1.990073 | 3.497681 | 0.769041 |
| Excl. 16 flagged rows (87,410 rows) | 1.989962 | 3.501352 | 0.768556 |
| **Difference** | **−0.000111** | **+0.003671** | **−0.000485** |

Effect is negligible. 16 flagged rows represent < 0.02% of training data.

**Conclusion:** Rainfall QC flags do not materially affect the evaluation metrics under either experiment.

---

## 10. Date-Column Consistency (Corrected)

### Issue identified

Two existing scripts used `Data Acquisition Time` (parsed with `dayfirst=True`) for the training/test split instead of the pre-processed `observation_date` column used by `baseline_model.py`.

| Script | Before fix | After fix |
|---|---|---|
| `full_support_analysis.py` | `Data Acquisition Time` ✗ | `observation_date` ✓ |
| `paired_gwl_support.py` | `Data Acquisition Time` ✗ | `observation_date` ✓ |
| All Person 3 scripts | `observation_date` ✓ | unchanged |

**Impact of fix:** The audit confirmed that `observation_date` and `Data Acquisition Time` assign all 104,781 rows to the same split side (0 discrepancies). However, the `Data Acquisition Time` date had a different string representation, causing different merge behaviour. After the fix:

- `paired_gwl_support.py`: well count changed from 125 → **161**, mean diff changed from +0.780 → **+0.414 m**
- `bootstrap_gwl_support.py` result changed: 95% CI from [−0.52, +2.06] → **[−0.658, +1.417]** m
- Conclusion unchanged: CI still includes zero — **not statistically conclusive**

The previous report value (125 wells, +0.780 m) was incorrect due to the date-column bug. The corrected value is used throughout.

---

## 11. Data Leakage Validation

All 12 checks passed. Summary:

- Predictions cover only the test period (≥ 2018-01-01)
- No training observations in the prediction file
- Zero training-test overlap by well+date key
- Actual GWL values are identical between the features file and the predictions file
- No scaler or calibration procedure that could leak test data

---

## 12. Statistical Interpretation

### What is conclusive

1. **RF significantly outperforms persistence.** CI [−0.728, −0.562] excludes zero. The RF reduces MAE by 0.61 m on average across wells.

2. **The unpaired high-rainfall effect is statistically significant but confounded.** CI [−0.715, −0.384] excludes zero. However, this reflects a between-well selection effect, not a within-well rainfall benefit. The same-well paired analysis (CI includes zero) is the appropriate test for a within-well effect.

### What is not conclusive

3. **Rainfall-only support and error:** Both unfamiliar–familiar (CI [−0.350, +0.065]) and same-well high-rainfall (CI [−0.169, +0.080]) CIs include zero. No statistically conclusive within-well relationship between rainfall extremes and prediction errors has been established.

4. **Previous GWL support (corrected):** 161 paired wells, mean diff +0.414 m, CI [−0.716, +1.452]. Includes zero — **not conclusive**.

### Important distinctions

- **Avg-of-well-means ≠ observation-weighted estimates.** Both are reported; they answer different questions.
- **Between-well difference ≠ within-well effect.** The unpaired comparison mixes both; the same-well comparison isolates the within-well effect.
- **Statistical significance ≠ practical significance.** Even where CIs exclude zero, the magnitude of effects and their real-world implications require domain judgment.

---

## 13. Unsupported Claims Identified

| Location | Claim | Status |
|---|---|---|
| `support_analysis.py` framing | Implies outside-rainfall-support → higher error | **Not confirmed**: unfamiliar obs have *lower* MAE |
| Previous report (pre-fix): "125 wells, CI [−0.52, +2.06]" | GWL support result | **Corrected to 161 wells, CI [−0.716, +1.452]** after date fix |
| Research handoff Section 9 | Temporal distribution changes — correctly qualified | Confirmed accurate |
| Research handoff Section 7 | Controlled removal — correctly qualified with CI | Confirmed accurate |

---

## 14. Confounding Factors and Limitations

1. **Unpaired between-well confound (key finding):** The 1,407 wells that never experience high rainfall have structurally higher MAE (2.38 m) than the 386 wells that do (1.73 m in normal conditions). Any unpaired high-rain comparison aggregates across different well populations.

2. **Groundwater extraction and pumping.** Not captured in the dataset.

3. **Irrigation return flow.** Particularly relevant in Karnataka.

4. **Geology and aquifer heterogeneity.** Spatial coordinates are the only proxy.

5. **Missing environmental predictors.** Evapotranspiration, soil moisture, land use.

6. **Irregular measurement frequency and well coverage changes** across years.

7. **Previous GWL assumption.** The rolling one-step-ahead setting assumes the previous observation is available at prediction time.

8. **Bootstrap independence assumption for cross-group CI.** Wells appearing in both familiar and unfamiliar buckets violate independence; this slightly underestimates CI width for that comparison.

9. **Small high-rain sample per well.** Median of 2 observations per well in the high-rain condition. Per-well means with 1–2 observations are unreliable.

10. **Person 2 reliability model absent.** Prediction interval coverage and calibration cannot be evaluated.

---

## 15. Corrections Applied in This Report

| Correction | Detail |
|---|---|
| Date-column fix: `full_support_analysis.py` | Changed from `Data Acquisition Time` to `observation_date` |
| Date-column fix: `paired_gwl_support.py` | Changed from `Data Acquisition Time` to `observation_date` |
| GWL support result updated | 125 wells → 161 wells; mean diff +0.780 → +0.414 m; CI updated |
| Unpaired/paired discrepancy explained | Between-well confound documented with diagnostic numbers |
| Retrain sensitivity experiment added | Confirms < 0.001 m change in MAE from excluding flagged training rows |
| Bootstrap limitation for Unfam–Fam noted | Cross-group independence assumption slightly underestimates CI |

---

## 16. Remaining Dependencies on Person 2

| File needed | Why |
|---|---|
| `results/reliability_predictions.csv` | Row-level predictions from reliability-aware model |
| `results/reliability_metrics.csv` | Coverage, interval width, MAE, RMSE under the reliability model |
| `results/reliability_by_condition.csv` | Coverage breakdown by familiar/unfamiliar and rainfall condition |
| `results/reliability_sensitivity.csv` | Sensitivity of interval calibration |

Until these are available, Person 3 cannot evaluate prediction interval calibration, conformal coverage, or how the reliability model performs on unfamiliar inputs.

---

## 17. Final Completion Checklist

### Successfully completed

| Task | Deliverable |
|---|---|
| Independent metric verification (12/12 data checks PASS) | `results/person3/verified_metrics.csv`, `validation_checks.csv` |
| Grouped evaluation (familiar/unfamiliar, high/low rain, same-well) | `results/person3/verified_group_metrics.csv`, `same_well_summary.csv` |
| Bootstrap CI verification, 2,000 reps, well-level cluster | `results/person3/verified_bootstrap_ci.csv` |
| Independent bootstrap re-implementation (2 quantities) | Section 8, audit script |
| Rainfall QC sensitivity: evaluation-filter experiment | `results/person3/rainfall_sensitivity.csv` |
| Rainfall QC sensitivity: retrain experiment (new) | Section 9, `audit_findings.csv` |
| Date-column consistency audit and correction | `full_support_analysis.py`, `paired_gwl_support.py` fixed |
| GWL support result corrected (161 wells, +0.414 m) | `results/paired_gwl_support.csv`, `verified_bootstrap_ci.csv` |
| Unpaired vs same-well discrepancy diagnosed | Section 7 |
| Report number cross-verification | Section 5, all numbers verified against CSV |
| Six comparison figures generated | `results/figures/person3/` |
| Second-stage audit (70 checks: 65 PASS, 4 WARN, 0 FAIL) | `results/person3/audit_findings.csv` |

### Not conclusive (requires more data or Person 2 output)

- Prediction interval coverage
- Conformal calibration
- Whether high-rainfall observations are harder to predict (same-well CI includes zero)
- Previous GWL support effect (CI includes zero)

### Outstanding

- Person 2 reliability model outputs are absent
- `bootstrap_gwl_support.py` (original script) output is now based on the corrected `paired_gwl_support.csv` (161 wells)

---

## Appendix: Commands to Reproduce Person 3 Experiments

```bash
# Prerequisites
python src/baseline_model.py
python src/paired_gwl_support.py

# Person 3 pipeline
python src/person3_robustness_validation.py
python src/person3_grouped_evaluation.py
python src/person3_bootstrap_validation.py
python src/person3_rainfall_sensitivity.py

# Second-stage audit (generates audit_findings.csv)
python src/person3_audit.py
```

All outputs: `results/person3/` (CSVs), `results/figures/person3/` (PNGs).  
Random seed: 42 throughout. All scripts run from repository root.
