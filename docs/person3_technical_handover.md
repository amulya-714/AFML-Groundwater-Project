# Person 3 — Technical Handover

## Research Objective

**Why Do Groundwater ML Models Fail Under Extreme Conditions? Separating Input Extrapolation from Predictive Relationship Instability.**

The project investigates two potential explanations for prediction failure under extreme conditions: (1) input extrapolation (test features outside training range) and (2) predictive relationship instability (the input-target relationship changes over time).

---

## Summary of Existing Person 2 Work

Person 2 implemented the full experimental framework:

1. **Baseline model** (`baseline_model.py`): Random Forest (300 trees, min_samples_leaf=3) with persistence baseline. Chronological split: pre-2018 training, 2018+ testing.

2. **Input extrapolation analysis** (`input_extrapolation_analysis.py`): NN novelty (k=5, training-scaled), full-feature and rainfall-only. Training-derived 95th/99th percentile thresholds.

3. **Novelty type comparison** (`compare_novelty_types.py`): Geographic vs environmental vs combined novelty in quartiles.

4. **Within-well novelty** (`within_well_novelty_analysis.py`): Environmental novelty computed within each well, quartile analysis, Wilcoxon paired test.

5. **Controlled training removal** (`controlled_training_removal.py`): Single-seed experiment comparing full vs extreme-removed vs matched-control models.

6. **Temporal analyses**: Paired 2018 vs 2020, matched-input temporal, within-well temporal residual regression, multivariate interaction tests, sensitivity trimming.

7. **Figures** (`create_research_figures.py`): 5 research figures.

---

## Person 3 Validation Work

### What was validated

| Item | Method | Result |
|---|---|---|
| Baseline metrics | Independent recalculation from row-level predictions | 12/12 checks PASS, metrics agree to machine precision |
| Data leakage | 12 integrity checks (overlap, dates, NaN, duplicates) | 12/12 PASS, no leakage |
| Novelty NN self-match | Inspected k+1 with index exclusion | Correct: training excludes self, test uses training reference |
| Training-only scaling | Inspected StandardScaler fit | Correct: fitted on train only, applied to test |
| Novelty thresholds | Inspected quantile derivation | Correct: from training reference distances only |
| Date-column consistency | Inspected all scripts | Fixed: `full_support_analysis.py` and `paired_gwl_support.py` now use `observation_date` |
| Bootstrap methodology | Inspected resampling logic | Corrected: familiar/unfamiliar bootstrap now uses paired within-well cluster resampling (824 wells) |
| QC sensitivity | Verified all flagged rows in training only | 24/24 flagged in training, 0 in test; retrain experiment confirms negligible effect |
| Rainfall provenance | Searched for source scripts and files | UNRESOLVED: no rainfall source data or documentation in repository |

### Bugs found and fixed

1. **Date-column inconsistency** (2 scripts): Changed from `Data Acquisition Time` to `observation_date`. Impact: GWL support paired wells 125→161, mean diff +0.780→+0.414 m. Conclusion unchanged.

2. **Familiar/unfamiliar bootstrap**: Changed from independent group resampling to paired within-well cluster bootstrap using 824 eligible wells. Impact: point estimate −0.152→+0.186 m, CI [−0.350,+0.065]→[+0.068,+0.307]. Corrected CI **excludes zero** — substantive finding change.

---

## New Experiments Implemented

### 1. Validated novelty analysis (`validated_novelty_analysis.py`)

- Novelty across 4 feature spaces (full, environmental, geographic, rainfall-only)
- Sensitivity to k (3, 5, 10), scaler (Standard, MinMax, Robust), metric (Euclidean, Manhattan)
- Novelty concentration by year and well
- Row-level predictions with novelty scores and support categories

**Key results:**
- Full-feature: Well-represented MAE=1.82 m vs Very-low-support MAE=3.35 m (difference +1.53 m)
- Rainfall-only: NO monotonic increase (Very-low-support MAE=1.82 < Well-represented MAE=1.93)
- Sensitivity: pattern stable across k=3,5,10

### 2. Repeated controlled removal (`repeated_controlled_removal.py`)

- 10 seeds (42–51), each training 3 models (Full, Extreme-removed, Matched-control)
- 874 training observations removed per seed (1% of 87,426)
- Removal manifests saved with row indices and removal types
- Support change measured per seed

**Key results:**
- B-A (extreme-full): mean=+0.0156 m, std=0.0020, 95% CI [+0.0144, +0.0169] — **excludes zero**
- B-C (extreme-control): mean=+0.0149 m, std=0.0012
- The single-seed result (CI included zero) was inconclusive; 10 seeds now shows a consistent small positive effect

### 3. Temporal stability validation (`temporal_stability_validation.py`)

- Year-wise MAE/RMSE with novelty group breakdown
- Familiar-input temporal comparison (2018 vs 2020 within novelty groups)
- Alternative explanations: well coverage, rainfall changes, missingness
- Conditional relationship changes: residual-rainfall correlations by year

**Key results:**
- Well-represented 2018→2020: +0.422 m, CI [+0.304, +0.536] — excludes zero
- Sparse-support 2018→2020: +2.060 m, CI [+1.107, +3.018] — excludes zero (but n=130 wells)
- Very-low-support: +0.510 m, CI [−1.688, +2.388] — includes zero (n=29 wells)
- Temporal deterioration persists among well-represented (familiar) observations
- Residual-rainfall correlations change across years (2019: r=−0.115 vs 2018: r=−0.007)

### 4. Automated test suite (`tests/test_data_integrity.py`)

- 21 automated tests covering split integrity, leakage, prediction finiteness, metric independence, novelty thresholds, bootstrap methodology, and removal manifest validity
- **21/21 PASS**

---

## Dataset Files Used

| File | Description |
|---|---|
| `data/processed/groundwater_rainfall_features_qc.csv` | Main dataset (104,781 × 30) |
| `results/baseline_predictions.csv` | Row-level RF + persistence predictions (14,863 rows) |

## Feature Definitions

| Feature | Description |
|---|---|
| Latitude, Longitude | Station coordinates |
| rainfall_30d_mm | 30-day rolling cumulative rainfall |
| rainfall_90d_mm | 90-day rolling cumulative rainfall |
| previous_gwl | Lag-1 GWL within well |
| month_sin, month_cos | Cyclical month encoding |

## Training and Test Dates

- Training: 1996-05-15 to 2017-11-05 (87,426 rows)
- Test: 2018-01-05 to 2020-11-30 (14,863 rows)
- Split boundary: `observation_date < 2018-01-01`

## Prediction Setting

Rolling one-step-ahead: previous GWL observation assumed available at prediction time.

## Model Specification

Random Forest: `n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1`

## Statistical Methodology

- Well-cluster bootstrap resampling (replace=True, 2000 reps)
- Paired within-well differences for same-well comparisons
- Percentile CI (2.5th/97.5th)
- Independent re-implementation with different seeds for verification

---

## Main Findings

### 1. Input extrapolation is associated with higher prediction errors

Full-feature novelty: Well-represented MAE=1.82 m → Very-low-support MAE=3.35 m (+1.53 m). This is consistent across k=3,5,10 and across environmental/geographic/rainfall-only feature spaces (except rainfall-only, which shows no monotonic increase).

### 2. Rainfall-only novelty does NOT predict higher errors

Rainfall-only: Very-low-support MAE=1.82 m < Well-represented MAE=1.93 m. This is a scientifically important negative result — the error increase is not explained by rainfall novelty alone.

### 3. Controlled removal shows a small consistent effect across seeds

10-seed average: extreme-removed MAE is +0.016 m higher than full model (CI [+0.014, +0.017]). The single-seed result was inconclusive; the repeated experiment provides stronger evidence that removing novel training observations degrades predictions.

### 4. Temporal deterioration persists among familiar observations

Well-represented 2018→2020: +0.422 m (CI excludes zero). This means temporal deterioration is not fully explained by increasing input novelty over time — it also occurs among observations with strong training support, consistent with possible predictive relationship instability.

### 5. Temporal deterioration is larger for sparse-support observations

Sparse-support 2018→2020: +2.060 m. However, this is based on only 130 paired wells and the CI is wide.

### 6. Familiar/unfamiliar bootstrap correction changed the finding

The corrected paired within-well bootstrap (824 wells) shows unfamiliar observations have **higher** MAE (+0.186 m, CI [+0.068, +0.307]). The original independent-group method gave the wrong sign (−0.152 m) because it treated the same well as independent across groups.

---

## Negative and Inconclusive Results

- **Rainfall-only novelty**: no monotonic error increase (retained as a scientifically important negative result)
- **High-rainfall same-well paired comparison**: CI [−0.169, +0.080] includes zero
- **Previous GWL support**: CI [−0.716, +1.452] includes zero (161 wells)
- **Very-low-support temporal comparison**: CI [−1.688, +2.388] includes zero (only 29 paired wells)

---

## Unresolved Issues

1. **Rainfall provenance**: The original rainfall source data and processing scripts are not in the repository. Cannot independently verify rainfall aggregation, spatial matching, or temporal alignment.

2. **Govt L V Polytechnic extreme values**: Two readings (101.16 m and 111.51 m) appear to be data entry errors but are retained per instructions. Correction requires team approval.

3. **Person 2 reliability model**: NOT required under the updated research direction. The `reliability_*.csv` files are explicitly excluded.

---

## Limitations

1. Rolling one-step-ahead assumption (previous GWL available at prediction time)
2. Nearest-neighbour novelty measures feature-space distance, not physical extreme events
3. Different sample sizes across experiments (eligibility criteria differ)
4. Control observations matched by year only, not by well or environmental characteristics
5. Temporal distribution changes across test years (2018, 2019, 2020)
6. Residual regression coefficients are associations, not causal effects
7. Unmeasured confounders: pumping, irrigation, geology, land-use changes
8. 10 seeds for repeated removal (target was 20; computational constraint)
9. Rainfall window construction cannot be independently verified

---

## Exact Reproduction Commands

```bash
# Prerequisites
python src/baseline_model.py
python src/paired_gwl_support.py

# Person 3 validation pipeline
python src/person3_robustness_validation.py
python src/person3_grouped_evaluation.py
python src/person3_bootstrap_validation.py
python src/person3_rainfall_sensitivity.py

# New experiments
python src/person3_data_validation.py
python src/validated_novelty_analysis.py
python src/repeated_controlled_removal.py
python src/temporal_stability_validation.py

# Audit and tests
python src/person3_audit.py
python tests/test_data_integrity.py
```

---

## Output File Inventory

See `docs/person3_output_inventory.md` for the complete table.

---

## Git Commit Hash

`f0b35df24aa5ac07ad8a7732ad6770ef728cd79f` (at start of Person 3 work)

---

## Recommended Commit Message

```
Add Person 3 evaluation, robustness validation, and strengthened experiments

- Independent metric verification (12/12 checks pass, no data leakage)
- Validated novelty analysis across 4 feature spaces with sensitivity tests
- Repeated controlled removal across 10 seeds (B-A CI excludes zero)
- Temporal stability validation with familiar-input breakdown
- Corrected familiar/unfamiliar bootstrap (paired within-well, 824 wells)
- Date-column fix for full_support_analysis.py and paired_gwl_support.py
- Automated test suite (21/21 pass)
- Data provenance investigation (rainfall source UNRESOLVED)
- Complete handover documentation
```
