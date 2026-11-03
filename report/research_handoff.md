# Groundwater ML Research Handoff

## Project Title

**Why Do Groundwater ML Models Fail Under Extreme Conditions? Separating Input Extrapolation from Predictive Relationship Instability.**

**Document purpose:** Research progress and technical handoff to the project team.

**Status:** Experimental implementation and analysis substantially completed. Results are being consolidated and validated. This is an interim research handoff, not the final project report.

---

## 1. Project Overview

This project investigates why machine learning models used for groundwater-level prediction may experience increased prediction errors under unfamiliar or changing environmental conditions.

The research focuses on separating two possible explanations:

1. **Input extrapolation:** Test observations contain combinations of input features that are poorly represented in the training data.
2. **Predictive relationship instability:** Prediction errors increase over time even when observations have similar or familiar input characteristics, suggesting that the learned predictive relationship may not remain stable.

The central objective is to understand whether prediction failures can be attributed primarily to unfamiliar inputs, temporal changes in predictive relationships, or a combination of both.

The project uses groundwater-level observations and rainfall-related features, with a chronological training and testing strategy.

## 2. Dataset and Preprocessing

### Dataset

Main processed dataset:

`data/processed/groundwater_rainfall_features_qc.csv`

Previously inspected dataset dimensions:

- 104,781 observations
- 30 columns

The dataset contains groundwater-level measurements, station and geographical information, rainfall features, rainfall grid information, observation dates and quality-control flags.

The groundwater observations include Central Ground Water Board (CGWB) records.

The processed dataset includes:
- Station and agency information
- State, district and geographical coordinates
- Groundwater level measurements
- Observation dates
- 30-day accumulated rainfall
- 90-day accumulated rainfall
- Rainfall grid identifiers and coordinates
- Rainfall quality-control flags

The rainfall source and complete rainfall preprocessing methodology should be confirmed from the original data-processing documentation before the final report is submitted.

### Target variable

`Groundwater Level Quarterly Manual (meter)`

### Model input features

The Random Forest uses seven features:

| Feature | Description |
|---|---|
| Latitude | Geographical latitude of the groundwater station |
| Longitude | Geographical longitude of the groundwater station |
| rainfall_30d_mm | Accumulated rainfall over 30 days |
| rainfall_90d_mm | Accumulated rainfall over 90 days |
| previous_gwl | Previous groundwater-level observation for the same well |
| month_sin | Sine encoding of observation month |
| month_cos | Cosine encoding of observation month |

### Feature construction

A well identifier is constructed using:

- Station
- Latitude
- Longitude

Observations are sorted chronologically within each well.

`previous_gwl` is constructed using the preceding groundwater-level observation for the corresponding well.

Month is represented using sine and cosine transformations to preserve its cyclical nature.

**Prediction setting:** The use of previous groundwater-level observations assumes a rolling one-step-ahead prediction setting in which the previous measurement is available when the next prediction is made. This is not a fixed-origin forecast of all future observations.

### Chronological split

The same chronological split is used throughout the main modelling experiments:

| Dataset | Date condition |
|---|---|
| Training | Before 2018-01-01 |
| Testing | On or after 2018-01-01 |

Verified sample counts:

| Dataset | Observations |
|---|---:|
| Training | 87,426 |
| Testing | 14,863 |

Training period: 1996-05-15 to 2017-11-05.

Testing period: 2018-01-05 to 2020-11-30.

The test set is retained for evaluation rather than used to fit the original Random Forest or novelty thresholds.

---

## 3. Baseline Model

### Model configuration

The baseline model is a Random Forest Regressor with:

- `n_estimators = 300`
- `min_samples_leaf = 3`
- `random_state = 42`
- `n_jobs = -1`

A persistence baseline is also evaluated. It predicts the next groundwater level using the previous groundwater-level observation.

### Baseline results

Verified from `results/baseline_metrics.csv`:

| Metric | Persistence | Random Forest |
|---|---:|---:|
| Test observations | 14,863 | 14,863 |
| MAE (m) | 2.6013 | 1.9901 |
| RMSE (m) | 4.4389 | 3.4977 |
| R² | 0.6280 | 0.7690 |

The Random Forest reduces MAE by approximately 23.5% and RMSE by approximately 21.2% relative to the persistence baseline.

This establishes the reference model used in subsequent experiments.

**Main files:**
- `src/baseline_model.py`
- `results/baseline_metrics.csv`
- `results/baseline_predictions.csv`

---

## 4. Input Extrapolation Analysis

### Objective

Determine whether test observations that are poorly represented in the training feature space experience higher prediction errors.

### Method

Nearest-neighbour novelty is calculated using standardized feature space.

The scaler is fitted using training observations only. Test observations are transformed using the training-fitted scaler.

The mean distance to the five nearest training observations is used as the novelty score.

Training observations are also used to calculate reference novelty thresholds:

- At or below the 95th percentile: Familiar
- Above the 95th and at or below the 99th percentile: Poorly represented
- Above the 99th percentile: Extreme novelty

The trained Random Forest is evaluated across these groups.

Well-level cluster bootstrap resampling is used to estimate 95% confidence intervals for group MAE.

### Full-feature novelty results

Verified from `results/input_extrapolation_summary.csv`:

| Novelty group | Observations | Wells | MAE (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| Familiar | 13,082 | 1,768 | 1.8155 | 2.9295 |
| Poorly represented | 1,090 | 674 | 3.2271 | 6.0301 |
| Extreme novelty | 691 | 503 | 3.3383 | 6.5833 |

The full-feature novelty thresholds are:

- 95th percentile: 0.350078
- 99th percentile: 0.554389

The MAE increases substantially for observations with higher multivariate novelty.

### Rainfall-only novelty

A separate novelty analysis uses only:

- `rainfall_30d_mm`
- `rainfall_90d_mm`

Results:

| Novelty group | Observations | Wells | MAE (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| Familiar | 10,684 | 1,773 | 1.9311 | 3.4819 |
| Poorly represented | 2,962 | 1,420 | 2.2731 | 3.9050 |
| Extreme novelty | 1,217 | 509 | 1.8153 | 2.4250 |

The rainfall-only novelty thresholds are:

- 95th percentile: 0.007326
- 99th percentile: 0.049964

Unlike full-feature novelty, rainfall-only novelty does not show a monotonic increase in prediction error.

This suggests that the observed full-feature novelty relationship cannot be attributed to rainfall novelty alone.

**Interpretation:** Joint feature-space novelty is associated with higher prediction errors, while rainfall-only novelty does not produce the same pattern.

**Main files:**
- `src/input_extrapolation_analysis.py`
- `results/input_extrapolation_observations.csv`
- `results/input_extrapolation_summary.csv`

---

## 5. Comparison of Novelty Types

### Objective

Investigate whether prediction errors are more strongly associated with geographic novelty or environmental novelty.

Three novelty definitions are evaluated:

1. Geographic novelty
2. Environmental novelty
3. Combined novelty

Environmental novelty includes rainfall, previous groundwater level and seasonal features. It must not be interpreted as rainfall-only novelty.

### Results

Verified from `results/novelty_type_summary.csv`:

| Novelty type | Q1 MAE (m) | Q4 MAE (m) |
|---|---:|---:|
| Geographic | 1.9015 | 1.9054 |
| Environmental | 1.5406 | 2.7542 |
| Combined | 1.3864 | 2.8737 |

Each quartile contains approximately 3,715–3,716 observations.

Geographic novelty produces almost no difference between the lowest and highest quartiles.

Environmental and combined novelty show substantially higher errors in their highest novelty quartiles.

### Interpretation

The results suggest that environmental feature-space novelty is more strongly associated with prediction errors than geographic distance alone.

This supports investigating the role of unfamiliar environmental conditions rather than treating geographical separation as the sole explanation for poor model performance.

**Main files:**
- `src/compare_novelty_types.py`
- `results/novelty_type_observations.csv`
- `results/novelty_type_summary.csv`

---

## 6. Within-Well Novelty Analysis

### Objective

Investigate whether the relationship between novelty and prediction error remains when observations are compared within the same groundwater wells.

This reduces the possibility that the novelty-error relationship is explained entirely by differences between geographically distinct wells.

### Method

The analysis considers wells that have sufficient observations in both the training and testing periods.

Novelty is evaluated within each well, and test observations are divided into four novelty quartiles.

### Results

Verified from `results/within_well_novelty_summary.csv`:

| Quartile | Observations | Wells | MAE (m) | RMSE (m) |
|---|---:|---:|---:|---:|
| Q1 | 4,361 | 1,643 | 1.7771 | 3.4087 |
| Q2 | 3,279 | 1,643 | 1.9255 | 3.3940 |
| Q3 | 3,000 | 1,643 | 1.9259 | 3.1023 |
| Q4 | 3,966 | 1,643 | 2.1879 | 3.5506 |

Total eligible observations: 14,606.

### Interpretation

MAE increases from 1.7771 m in Q1 to 2.1879 m in Q4, an increase of approximately 23.1%.

The RMSE pattern is not monotonic.

This provides additional evidence that higher novelty is associated with increased prediction errors even within the eligible wells, although the pooled quartile comparison is not a fully controlled causal comparison.

**Main files:**
- `src/within_well_novelty_analysis.py`
- `src/within_well_paired_test.py`
- `src/within_well_bootstrap.py`
- `results/within_well_novelty_observations.csv`
- `results/within_well_novelty_summary.csv`
- `results/within_well_novelty_per_well.csv`
- `results/within_well_paired_test_summary.csv`
- `results/within_well_paired_test_per_well.csv`

---

## 7. Controlled Training Removal Experiment

### Objective

Investigate whether removing unusual training observations affects model performance on naturally occurring unfamiliar test observations.

### Model variants

Three Random Forest models are compared:

1. **Full model:** Trained on the complete eligible training set.
2. **Extreme-removed model:** Trained after removing the top 1% most locally novel training observations.
3. **Matched-control model:** Trained after removing the same number of non-extreme observations, matched by year.

All three models use identical Random Forest hyperparameters and the same untouched chronological test set.

### Training novelty and support

Training novelty is calculated using five-nearest-neighbour distances in standardized seven-feature space.

The extreme training observations are selected using the highest local novelty scores.

The control observations are randomly sampled from non-extreme training observations within the same years as the removed extreme observations.

Test support distances are recalculated relative to each remaining training set.

### Overall results

Verified from `results/controlled_removal_metrics.csv`:

| Model | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|
| Full model | 1.9898 | 3.4979 | 0.7690 |
| Extreme-removed model | 2.0061 | 3.6110 | 0.7538 |
| Matched-control model | 1.9919 | 3.5000 | 0.7687 |

The overall MAE increase after extreme removal is approximately 0.0163 m.

The well-clustered bootstrap confidence interval for the difference is:

- Difference: +0.0163 m
- 95% CI: [-0.0038, 0.0384] m

Because this interval includes zero, the overall deterioration is not statistically conclusive.

### Results by test novelty group

| Test group | Full model MAE | Extreme-removed MAE | Difference |
|---|---:|---:|---:|
| Familiar | 1.8155 | 1.8134 | -0.0021 |
| Poorly represented | 3.2271 | 3.3093 | +0.0822 |
| Extreme novelty | 3.3383 | 3.5985 | +0.2602 |

Bootstrap confidence intervals for the extreme-removed model minus the full model:

| Test group | MAE difference (m) | 95% CI (m) |
|---|---:|---:|
| All observations | +0.0163 | [-0.0038, 0.0384] |
| Familiar | -0.0021 | [-0.0057, 0.0014] |
| Poorly represented | +0.0822 | [0.0305, 0.1410] |
| Extreme novelty | +0.2602 | [-0.1505, 0.7079] |

The clearest evidence of deterioration occurs in the poorly represented group.

The extreme-novelty group has a larger estimated deterioration, but its confidence interval is wide and includes zero.

### Training support comparison

Mean test support distances:

| Test group | Full training set | Extreme removed |
|---|---:|---:|
| All observations | 0.2254 | 0.2332 |
| Familiar | 0.1757 | 0.1757 |
| Poorly represented | 0.4236 | 0.4311 |
| Extreme novelty | 0.8542 | 1.0110 |

Removing extreme training observations increases support distances, particularly for extreme-novelty test observations.

For the extreme-novelty group, mean support distance increases by approximately 18.4%.

### Interpretation

The experiment provides evidence that removing rare training observations can reduce support for unfamiliar test conditions and may worsen predictions, especially for poorly represented inputs.

However, the overall performance difference is not statistically conclusive, and the extreme-novelty subgroup has substantial uncertainty.

The control observations are matched by year only, not by well or environmental characteristics. This limitation must be acknowledged.

### Bootstrap methodology

`src/controlled_removal_bootstrap.py` uses paired well-level cluster bootstrap resampling.

All observations belonging to a sampled well are retained together. The bootstrap estimates observation-weighted MAE differences between models.

The script uses 5,000 bootstrap repetitions.

**Main files:**
- `src/controlled_training_removal.py`
- `src/controlled_removal_bootstrap.py`
- `results/controlled_removal_predictions.csv`
- `results/controlled_removal_metrics.csv`
- `results/controlled_removal_support.csv`
- `results/controlled_removal_bootstrap.csv`

---

## 8. Temporal Prediction Error Analysis

### Objective

Investigate whether prediction errors increase over time and whether this pattern remains when comparing the same groundwater wells.

### Paired temporal comparison

The paired temporal analysis compares 2018 and 2020 prediction errors for wells with observations in both periods.

Verified from `results/paired_temporal_summary.csv`:

| Metric | Result |
|---|---:|
| Paired wells | 1,516 |
| Mean MAE, 2018 | 1.7123 m |
| Mean MAE, 2020 | 2.1279 m |
| Mean paired difference | +0.4156 m |
| Median paired difference | +0.1481 m |
| Fraction of wells with increased error | 54.95% |
| Bootstrap 95% CI | [0.2976, 0.5315] m |
| Wilcoxon p-value | 9.64 × 10⁻¹² |

This provides evidence of increased prediction errors between 2018 and 2020 across the paired wells.

### Matched-input temporal comparison

A separate experiment matches observations within the same well using:

- 30-day rainfall
- 90-day rainfall
- Previous groundwater level
- Seasonal sine feature
- Seasonal cosine feature

One-to-one matching is performed between 2018 and 2020 observations.

Verified from `results/within_well_matched_input_summary.csv`:

| Metric | Result |
|---|---:|
| Matched pairs | 2,077 |
| Matched wells | 1,311 |
| Mean input distance | 0.4932 |
| Mean MAE, 2018 | 1.2619 m |
| Mean MAE, 2020 | 1.7682 m |
| Mean error increase | +0.5063 m |
| Median error increase | +0.3233 m |
| Fraction of pairs with higher 2020 error | 59.94% |
| Fraction of wells with higher 2020 error | 62.09% |

The clustered bootstrap result is:

- Mean error increase: +0.5063 m
- 95% CI: [0.4100, 0.6049] m
- Bootstrap repetitions: 5,000

This is an important result because the comparison is performed within wells and between observations with similar environmental inputs.

It suggests that temporal deterioration cannot be explained solely by differences in the input values represented in the matched pairs.

However, matching does not establish causal equivalence, and other temporal factors may still differ.

**Main files:**
- `src/paired_temporal_stability.py`
- `src/paired_temporal_bin_test.py`
- `src/analyze_paired_temporal_distribution.py`
- `src/within_well_matched_input_analysis.py`
- `src/within_well_clustered_bootstrap.py`
- `results/paired_temporal_summary.csv`
- `results/paired_temporal_bin_summary.csv`
- `results/paired_temporal_distribution_summary.csv`
- `results/within_well_matched_input_pairs.csv`
- `results/within_well_matched_input_summary.csv`
- `results/within_well_clustered_bootstrap.csv`

---

## 9. Temporal Distribution Comparison

The input and target distributions were compared across 2018, 2019 and 2020.

Selected annual means:

| Variable | 2018 | 2019 | 2020 |
|---|---:|---:|---:|
| Groundwater level (m) | 5.81 | 7.09 | 6.62 |
| 30-day rainfall (mm) | 67.10 | 47.91 | 23.79 |
| 90-day rainfall (mm) | 272.71 | 240.10 | 349.00 |
| Previous groundwater level (m) | 6.43 | 6.69 | 6.43 |

The number of observations differs by year.

These distribution changes are important when interpreting temporal prediction deterioration.

They indicate that the test-period data are not identically distributed across years. Consequently, higher 2020 errors could be associated with changes in input distributions, predictive relationships, measurement conditions or other unobserved factors.

The distribution comparison does not independently establish concept drift.

**Main files:**
- `src/temporal_distribution_comparison.py`
- `results/temporal_distribution_comparison.csv`
- `results/temporal_distribution_matched_observations.csv`

---

## 10. Temporal Residual Relationship Analysis

### Objective

Investigate whether the relationship between input features and Random Forest residuals changes over time.

The analysis uses residual regression models with well fixed effects, implemented by demeaning observations within each well.

### Model specifications

Three models are compared:

1. Inputs only.
2. Inputs plus year effects.
3. Inputs plus year-feature interactions.

The analysis uses 12,142 observations from 1,479 wells.

### Results

Verified from `results/within_well_temporal_summary.csv`:

| Model | Observations | Wells | R² | Adjusted R² |
|---|---:|---:|---:|---:|
| Inputs only | 12,142 | 1,479 | 0.2727 | 0.2724 |
| Inputs + year | 12,142 | 1,479 | 0.2742 | 0.2738 |
| Inputs + year + interactions | 12,142 | 1,479 | 0.3044 | 0.3034 |

The interaction model improves R² compared with the inputs-only model.

Several temporal interactions involving rainfall and seasonal variables are statistically significant.

Examples from the interaction model include:

| Interaction | Coefficient | p-value |
|---|---:|---:|
| 2019 × 90-day rainfall | +0.8920 | 8.19 × 10⁻¹³ |
| 2019 × month sine | -0.2203 | 0.00279 |
| 2019 × month cosine | +0.3751 | 7.58 × 10⁻¹³ |
| 2020 × month sine | -1.0942 | 1.45 × 10⁻⁴⁵ |
| 2020 × month cosine | +0.8083 | 3.10 × 10⁻⁸ |

These are residual regression coefficients, not direct estimates of the original Random Forest's feature effects.

### Sensitivity analysis

The temporal residual sensitivity analysis evaluates the interaction model after trimming observations with the largest absolute residuals.

| Scenario | Observations | Interaction R² | Wald statistic | Joint p-value |
|---|---:|---:|---:|---:|
| Original | 13,082 | 0.03775 | 470.22 | 1.88 × 10⁻⁹¹ |
| Trim largest 1% | 12,951 | 0.05708 | 619.99 | 2.95 × 10⁻¹²³ |
| Trim largest 5% | 12,427 | 0.07020 | 710.14 | 1.77 × 10⁻¹⁴² |

The interaction effects remain statistically significant under these sensitivity scenarios.

This supports the observation that residual relationships differ over time.

However, trimming is outcome-based and should be interpreted as a sensitivity analysis, not as an independent validation dataset.

### Interpretation

The temporal residual analysis suggests that the relationship between model errors and input features changes across years.

Combined with the matched-input results, this provides evidence consistent with temporal predictive instability.

It does not prove that the underlying physical groundwater process has changed. Measurement practices, data quality, unobserved variables and other temporal factors remain possible explanations.

**Main files:**
- `src/within_well_temporal_stability.py`
- `src/multivariate_temporal_residual_test.py`
- `src/temporal_residual_sensitivity.py`
- `results/within_well_temporal_summary.csv`
- `results/within_well_temporal_coefficients.csv`
- `results/within_well_temporal_covariance.csv`
- `results/within_well_temporal_observations.csv`
- `results/multivariate_temporal_model_summary.csv`
- `results/multivariate_temporal_coefficients.csv`
- `results/temporal_residual_sensitivity.csv`

---

## 11. Research Figures

Five research figures have been generated in:

`results/figures/`

| Figure | Filename | Description |
|---|---|---|
| Figure 1 | `figure1_baseline_comparison.png` | Persistence and Random Forest baseline comparison |
| Figure 2 | `figure2_input_novelty.png` | Prediction error across input novelty groups |
| Figure 3 | `figure3_novelty_types.png` | Geographic, environmental and combined novelty |
| Figure 4 | `figure4_temporal_degradation.png` | Temporal prediction error comparison |
| Figure 5 | `figure5_feature_relationships.png` | Temporal feature-relationship coefficients |

Figure 5 uses the covariance matrix from the within-well temporal regression when calculating confidence intervals for combined base and interaction coefficients.

The figures are generated using:

`src/create_research_figures.py`

---

## 12. Main Findings So Far

The current experimental results support the following observations.

### Finding 1: Joint input novelty is associated with higher prediction errors

The full-feature novelty analysis shows a substantial increase in MAE between familiar and unfamiliar test observations.

However, rainfall-only novelty does not show the same monotonic pattern.

### Finding 2: Environmental novelty matters more than geographic distance alone

Geographic novelty produces almost flat MAE across quartiles, while environmental and combined novelty show considerably higher errors in their upper quartiles.

### Finding 3: The novelty-error relationship remains visible within wells

The within-well novelty analysis shows higher MAE in the highest novelty quartile.

### Finding 4: Prediction errors increase over time

Paired temporal and matched-input analyses both show higher prediction errors in 2020 than in 2018.

### Finding 5: Residual relationships differ across years

The temporal residual regression models show improved explanatory power when year-feature interactions are included.

Together, these findings suggest that both unfamiliar input conditions and temporal changes in predictive performance may contribute to groundwater prediction failures.

These are empirical findings from the current experiments, not proof of causal mechanisms.

---

## 13. Important Limitations

The following limitations should be considered during further development and final reporting.

1. **Rolling one-step-ahead setting:** Previous groundwater-level observations are assumed to be available when predictions are made.

2. **Novelty definition:** Nearest-neighbour distance measures multivariate feature-space novelty. It does not automatically identify a physical extreme hydroclimatic event.

3. **Environmental novelty:** The environmental novelty definition includes rainfall, previous groundwater level and seasonal features. It is not equivalent to rainfall-only novelty.

4. **Different sample counts:** Several experiments apply different well eligibility, matching and missing-data criteria. Their sample counts should not be treated as interchangeable.

5. **Controlled removal:** Control observations are matched by year only. They are not fully matched by well or environmental characteristics.

6. **Temporal distribution changes:** Input and groundwater-level distributions vary across the test years.

7. **Residual regression interpretation:** Temporal interaction coefficients describe associations in model residuals, not direct changes in the original Random Forest's learned feature effects.

8. **Potential measurement effects:** Changes in measurement procedures, observation quality and unobserved variables may contribute to temporal differences.

9. **Generalisation:** The experiments use a specific groundwater dataset and chronological split. Further validation is needed before claiming generalisation to other regions or datasets.

---

## 14. Repository Structure

The main project structure is:

```text
AFML-Groundwater-Project/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── groundwater_rainfall_features_qc.csv
│
├── notebooks/
│
├── papers/
│
├── report/
│   ├── baseline_handoff.md
│   └── research_handoff.md
│
├── results/
│   ├── figures/
│   │   ├── figure1_baseline_comparison.png
│   │   ├── figure2_input_novelty.png
│   │   ├── figure3_novelty_types.png
│   │   ├── figure4_temporal_degradation.png
│   │   └── figure5_feature_relationships.png
│   │
│   └── [experimental CSV outputs]
│
├── src/
│   └── [experiment and analysis scripts]
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 15. Reproducing the Main Experiments

The project uses Python and the dependencies pinned in `requirements.txt`.

From the repository root, activate the existing virtual environment:

```bash
source .venv/bin/activate
```

Run the baseline:

```bash
python src/baseline_model.py
```

Run full-feature and rainfall-only novelty analysis:

```bash
python src/input_extrapolation_analysis.py
```

Run novelty-type comparison:

```bash
python src/compare_novelty_types.py
```

Run within-well novelty analysis:

```bash
python src/within_well_novelty_analysis.py
```

Run controlled training removal:

```bash
python src/controlled_training_removal.py
```

Run its bootstrap evaluation:

```bash
python src/controlled_removal_bootstrap.py
```

Run the within-well matched-input analysis:

```bash
python src/within_well_matched_input_analysis.py
```

Run its clustered bootstrap:

```bash
python src/within_well_clustered_bootstrap.py
```

Run within-well temporal stability analysis:

```bash
python src/within_well_temporal_stability.py
```

Regenerate the research figures:

```bash
python src/create_research_figures.py
```

The scripts save their outputs in `results/`.

For a complete reproduction, scripts should be executed in the appropriate dependency order. Some analyses depend on outputs created by earlier scripts.

---

## 16. Current Handoff Status

### Completed

- Processed groundwater and rainfall dataset available.
- Chronological training/testing split established.
- Persistence and Random Forest baseline evaluated.
- Full-feature novelty analysis implemented.
- Rainfall-only novelty analysis implemented.
- Geographic, environmental and combined novelty comparisons completed.
- Within-well novelty analysis completed.
- Controlled training-removal experiment implemented and audited.
- Well-clustered bootstrap evaluation implemented.
- Paired temporal comparison completed.
- Within-well matched-input comparison completed.
- Within-well temporal residual regression completed.
- Temporal residual sensitivity analysis completed.
- Five research figures generated.

### Remaining work for the team

- Review the methodology and confirm all experiment definitions.
- Verify rainfall data provenance and preprocessing documentation.
- Review measurement consistency across the observation years.
- Consolidate the strongest findings into the final research report.
- Complete the repository reproducibility documentation.
- Review scripts and results for consistency before final submission.
- Conduct any additional validation required by the project supervisor.
- Prepare the final presentation and project handoff.

---

## 17. Overall Research Direction

The project has progressed beyond ordinary groundwater-level prediction evaluation.

The current work investigates two complementary aspects of model reliability:

- Whether test observations are sufficiently represented by the training feature space.
- Whether predictive residual relationships remain stable over time.

The preliminary evidence indicates that both input novelty and temporal changes in predictive performance deserve attention.

The next stage is to consolidate these findings, validate the remaining assumptions and develop the final research narrative.

**This document represents the current implementation and experimental handoff. It is not the final research paper or a claim that all possible validation has been completed.**
