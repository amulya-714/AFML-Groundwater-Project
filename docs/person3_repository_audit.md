# Person 3 — Repository Audit

**Date:** 2026-11-07
**Git commit at audit:** `f0b35df24aa5ac07ad8a7732ad6770ef728cd79f` (thrupthi912)
**Working source:** `amulya-714/AFML-Groundwater-Project` (commit `35298c2`)

---

## 1. Repository State

The `thrupthi912` repository was initially nearly empty (README, requirements, .gitignore only).
The full research implementation lives in `amulya-714/AFML-Groundwater-Project`.
All scripts and data have been copied into the `thrupthi912` working copy and verified to execute.

**Current working directory structure:**

```
AFML-Groundwater-Project/
├── data/
│   ├── processed/groundwater_rainfall_features_qc.csv  (104,781 rows × 30 cols)
│   └── raw/gwl_manual_quarterly_cgwb_ka_1991_2020.csv
├── src/                  (34 Python scripts)
├── results/              (40 CSV files + figures/)
├── docs/                 (Person 3 documentation)
├── tests/                (Person 3 automated tests)
├── report/
│   ├── baseline_handoff.md
│   └── research_handoff.md
├── REPRODUCIBILITY.md
└── requirements.txt
```

---

## 2. Existing Scripts Inventory

| Script | Purpose | Executed? |
|---|---|---|
| `baseline_model.py` | Random Forest + persistence baseline | ✓ |
| `input_extrapolation_analysis.py` | Full-feature + rainfall-only NN novelty | ✓ |
| `compare_novelty_types.py` | Geographic / environmental / combined novelty | ✓ |
| `within_well_novelty_analysis.py` | Within-well environmental novelty quartiles | ✓ |
| `within_well_paired_test.py` | Within-well paired novelty Wilcoxon test | ✓ |
| `within_well_bootstrap.py` | Bootstrap CI for within-well novelty | ✓ |
| `controlled_training_removal.py` | 3-model removal experiment (seed 42) | ✓ |
| `controlled_removal_bootstrap.py` | Well-cluster bootstrap for removal | ✓ |
| `within_well_matched_input_analysis.py` | Matched 2018-2020 input comparison | ✓ |
| `within_well_clustered_bootstrap.py` | Bootstrap CI for matched temporal diff | ✓ |
| `within_well_temporal_stability.py` | Within-well temporal residual regression | ✓ |
| `multivariate_temporal_residual_test.py` | Multivariate residual interaction test | ✓ |
| `temporal_residual_sensitivity.py` | Sensitivity to residual trimming | ✓ |
| `paired_temporal_stability.py` | 2018 vs 2020 paired comparison | ✓ |
| `create_research_figures.py` | Generates 5 research figures | ✓ |
| `analyze_baseline_errors.py` | Baseline error analysis | ✓ |
| `investigate_extreme_errors.py` | Extreme error investigation | ✓ |
| `check_problem_well.py` | Problem well check | ✓ |
| `support_analysis.py` | Rainfall support analysis | ✓ |
| `full_support_analysis.py` | Per-feature support (date fixed) | ✓ |
| `same_well_analysis.py` | Same-well rainfall comparison | ✓ |
| `paired_gwl_support.py` | Previous GWL support (date fixed) | ✓ |
| `paired_bootstrap.py` | Rainfall bootstrap | ✓ |
| `bootstrap_gwl_support.py` | GWL support bootstrap CI | ✓ |
| `conditional_residual_analysis.py` | Conditional residual analysis | ✓ |
| `paired_temporal_bin_test.py` | Paired temporal bin test | ✓ |
| `analyze_paired_temporal_distribution.py` | Temporal distribution | ✓ |
| `temporal_distribution_comparison.py` | Year distribution comparison | ✓ |
| `temporal_error_by_novelty.py` | Temporal error by novelty | ✓ |
| `matched_input_temporal_analysis.py` | Matched input temporal | ✓ |
| `person3_robustness_validation.py` | Person 3 Task A (metric verification) | ✓ |
| `person3_grouped_evaluation.py` | Person 3 Task B (grouped eval) | ✓ |
| `person3_bootstrap_validation.py` | Person 3 Task C (bootstrap, v2 corrected) | ✓ |
| `person3_rainfall_sensitivity.py` | Person 3 Task D (QC sensitivity) | ✓ |
| `person3_audit.py` | Person 3 second-stage audit (77 checks) | ✓ |

---

## 3. Existing Results Inventory

### Summary CSVs (verified)

| File | Rows | Key content |
|---|---|---|
| `baseline_metrics.csv` | 2 | Persistence & RF MAE/RMSE/R² |
| `baseline_predictions.csv` | 14,863 | Row-level predictions |
| `input_extrapolation_summary.csv` | 6 | Full-feature & rainfall-only novelty groups |
| `input_extrapolation_observations.csv` | 14,863 | Row-level novelty scores |
| `novelty_type_summary.csv` | 12 | Geographic/environmental/combined quartiles |
| `within_well_novelty_summary.csv` | 5 | Within-well novelty quartiles |
| `within_well_paired_test_summary.csv` | 1 | Wilcoxon test: +0.438 m, p<0.001 |
| `within_well_bootstrap_summary.csv` | 1 | Bootstrap CI: [0.326, 0.547] |
| `controlled_removal_metrics.csv` | 12 | 3 models × 4 test groups |
| `controlled_removal_bootstrap.csv` | 8 | Bootstrap CIs for removal differences |
| `controlled_removal_predictions.csv` | 14,863 | Row-level predictions for all 3 models |
| `paired_temporal_summary.csv` | 1 | 2018→2020: +0.416 m, CI [0.298, 0.531] |
| `within_well_matched_input_summary.csv` | 1 | Matched input: +0.506 m, CI [0.410, 0.605] |
| `within_well_clustered_bootstrap.csv` | 1 | 5,000 reps clustered bootstrap |
| `within_well_temporal_summary.csv` | 3 | 3 regression models R² |
| `within_well_temporal_coefficients.csv` | 41 | Regression coefficients + p-values |
| `within_well_temporal_observations.csv` | 12,142 | Row-level residuals |
| `multivariate_temporal_model_summary.csv` | 3 | R² for 3 interaction models |
| `temporal_residual_sensitivity.csv` | 3 | Trim sensitivity |
| `temporal_error_summary.csv` | 4 | Error by novelty group |

### Person 3 verified outputs

| File | Rows | Content |
|---|---|---|
| `verified_metrics.csv` | 2 | Independent MAE/RMSE/R² |
| `validation_checks.csv` | 12 | All PASS (no leakage) |
| `verified_group_metrics.csv` | 6 | Familiar/unfamiliar + high/low rain |
| `same_well_summary.csv` | 1 | 386 wells, paired diff -0.041 m |
| `same_well_per_well_differences.csv` | 386 | Per-well differences |
| `verified_bootstrap_ci.csv` | 6 | 6 bootstrap CIs (incl. corrected) |
| `rainfall_sensitivity.csv` | 5 | QC sensitivity scenarios |
| `audit_findings.csv` | 77 | Full audit: 75 PASS, 1 WARN, 0 FAIL |

---

## 4. Existing vs Missing Deliverables

### Existing and verified

- [x] Baseline model + persistence comparison
- [x] Full-feature novelty analysis (NN, k=5, training-scaled)
- [x] Rainfall-only novelty analysis
- [x] Geographic / environmental / combined novelty comparison
- [x] Within-well novelty quartile analysis
- [x] Within-well paired novelty test (Wilcoxon)
- [x] Within-well novelty bootstrap CI
- [x] Controlled training removal (single seed)
- [x] Controlled removal well-clustered bootstrap
- [x] Paired temporal (2018 vs 2020) comparison
- [x] Within-well matched-input temporal comparison
- [x] Within-well clustered bootstrap for temporal diff
- [x] Within-well temporal residual regression (3 models)
- [x] Multivariate temporal residual interaction test
- [x] Temporal residual sensitivity (trimming)
- [x] 5 research figures

### Existing but needs Person 3 strengthening

- [ ] Novelty NN self-match handling: verified, correct (k+1 with index exclusion)
- [ ] Training-only scaling: verified correct
- [ ] Novelty thresholds: verified training-derived (95th/99th pctile)
- [ ] Date-column consistency: fixed (full_support_analysis.py, paired_gwl_support.py)
- [ ] Familiar/unfamiliar bootstrap: corrected (paired well-cluster, 824 wells)

### Not yet implemented (Person 3 tasks)

- [ ] Data provenance documentation (rainfall source)
- [ ] Feature construction documentation
- [ ] Data quality summary CSV
- [ ] Yearly station coverage CSV
- [ ] Validated novelty analysis with separate feature spaces
- [ ] Novelty sensitivity (k, scaling, distance metric)
- [ ] Novelty concentration by year/well/district
- [ ] Repeated controlled removal (multiple seeds)
- [ ] Removal manifests and support-change measurement
- [ ] Familiar-input temporal analysis
- [ ] Temporal relationship comparison
- [ ] Automated test suite
- [ ] Updated REPRODUCIBILITY.md
- [ ] Technical handover document
- [ ] Results summary document
- [ ] Output inventory

### Missing inputs (blocked)

- [ ] **Rainfall source data** — the original rainfall grid files and download/processing scripts are NOT in the repository. Only the merged `groundwater_rainfall_features_qc.csv` is available. Provenance cannot be independently verified without the source.
- [ ] **Person 2 reliability model** — NOT required under the updated research direction. The `reliability_*.csv` files are explicitly excluded.

---

## 5. Key Findings from Audit

### Bugs found and fixed

1. **Date-column inconsistency** in `full_support_analysis.py` and `paired_gwl_support.py`: used `Data Acquisition Time` instead of `observation_date`. Fixed. Impact: GWL support paired wells changed 125→161, mean diff +0.780→+0.414 m. Conclusion unchanged (CI includes zero).

2. **Familiar/unfamiliar bootstrap** in `person3_bootstrap_validation.py`: original independent group resampling treated the same well as independent across groups. Corrected to paired within-well cluster bootstrap using 824 eligible wells. Impact: point estimate changed −0.152→+0.186 m, CI changed [−0.350,+0.065]→[+0.068,+0.307]. The corrected CI **excludes zero** — this is a substantive finding change.

### Verified correct

- Nearest-neighbour self-match handling: k+1 neighbours, index exclusion
- Training-only StandardScaler fitting
- Training-derived novelty thresholds (95th/99th percentile)
- Chronological split (before 2018-01-01 / from 2018-01-01)
- No training-test overlap (0 duplicate keys)
- No data leakage in baseline model
- Well-cluster bootstrap resampling (replace=True)

### Unverified assumptions

- Rainfall temporal aggregation window construction (source not available)
- Rainfall grid matching procedure (source not available)
- Whether `rainfall_30d_mm` / `rainfall_90d_mm` include or exclude the observation date
- Whether missing rainfall days were treated as zero or missing
