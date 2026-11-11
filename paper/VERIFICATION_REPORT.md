# Verification Report — AFML Groundwater Paper (Person 4)

**Generated:** Person 4 audit and paper writing phase  
**Repository:** amulya-714/AFML-Groundwater-Project  
**Paper file:** `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/main.tex`

---

## 1. Sections Completed

| Section | Title | Status |
|---------|-------|--------|
| Abstract | ~250 words, real numbers only | Complete |
| Index Terms | 10 IEEE-style keywords | Complete |
| I. Introduction | Motivation, objectives, 7 contributions listed | Complete |
| II. Related Work | 5 subsections, verifiable citations only | Complete |
| III. Dataset and Study Design | CGWB Karnataka, actual counts, QC table | Complete |
| IV. Methodology | 5 subsections (A–E), Person 2 gap documented | Complete |
| V. Experimental Results | 11 tables, 9 figures, all real numbers | Complete |
| VI. Discussion | 3 subsections, between-well confound addressed | Complete |
| VII. Limitations | 13 documented limitations | Complete |
| VIII. Conclusion and Future Work | Key findings, future directions | Complete |
| References | IEEE numbered format, all verified entries | Complete |

---

## 2. Source Files Inspected

### Documentation

| File | Status |
|------|--------|
| `README.md` | Read — project overview, team, research question |
| `REPRODUCIBILITY.md` | Read — full pipeline, script ordering, notes |
| `requirements.txt` | Read — Python dependencies confirmed |
| `report/baseline_handoff.md` | Read — baseline metrics, next steps |
| `report/research_handoff.md` | Read — comprehensive methodology and findings |
| `docs/person3_robustness_report.md` | Read — full audit report |
| `docs/person3_results_summary.md` | Read — results summary with numbers |
| `docs/data_provenance.md` | Present (not needed beyond dataset description) |
| `docs/feature_construction.md` | Present (confirmed feature descriptions) |
| `docs/person3_repository_audit.md` | Present |
| `docs/person3_output_inventory.md` | Present |
| `docs/person3_reproduction_guide.md` | Present |
| `docs/person3_technical_handover.md` | Present |

### Source Scripts (methodology verification)

| File | Purpose |
|------|---------|
| `src/baseline_model.py` | RF configuration, chronological split |
| `src/input_extrapolation_analysis.py` | k-NN novelty, StandardScaler, thresholds |
| `src/person3_robustness_validation.py` | Person 3 independent evaluation |
| `src/person3_grouped_evaluation.py` | Familiar/unfamiliar grouping logic |
| `src/person3_bootstrap_validation.py` | Bootstrap CI methodology |
| `src/person3_rainfall_sensitivity.py` | QC sensitivity analysis |
| `src/conditional_residual_analysis.py` | Conditional residuals by bin/year |
| `src/controlled_training_removal.py` | Three-model removal experiment |
| `src/support_analysis.py` | Support/novelty utilities |

---

## 3. Result CSVs Used

| CSV File | Key Values Extracted | Used In |
|----------|---------------------|---------|
| `results/baseline_metrics.csv` | Persistence: MAE=2.6013, RMSE=4.4389, R²=0.6280; RF: MAE=1.9901, RMSE=3.4977, R²=0.7690 | Table 1, Abstract |
| `results/input_extrapolation_summary.csv` | Full-feature: Familiar MAE=1.8155, Poorly rep. MAE=3.2271, Extreme MAE=3.3383; Rainfall-only: Familiar MAE=1.9311, Poorly rep. MAE=2.2731, Extreme MAE=1.8153 | Tables 2, 3 |
| `results/novelty_type_summary.csv` | Geographic Q1=1.9015/Q4=1.9054; Environmental Q1=1.5406/Q4=2.7542; Combined Q1=1.3864/Q4=2.8737 | Table 4 |
| `results/novelty_by_year.csv` | 2018: n=6239, MAE=1.804; 2019: n=5629, MAE=2.035; 2020: n=2995, MAE=2.291 | Table 11, Discussion |
| `results/novelty_sensitivity.csv` | k/scaler sensitivity confirmation (k=5, StandardScaler is primary) | Methods text |
| `results/conditional_residual_analysis.csv` | MAE by rainfall/GWL bin and year (2018–2020) | Methods, Discussion |
| `results/controlled_removal_metrics.csv` | Full/Extreme-removed/Matched-control MAE per novelty group | Table 10 |
| `results/temporal_error_summary.csv` | All: MAE=1.9898; Familiar: 1.8155; Poorly rep.: 3.2271; Extreme: 3.3383 | Table 10 context |
| `results/within_well_temporal_summary.csv` | Inputs only R²=0.2727; +year R²=0.2742; +interactions R²=0.3044; n=12142, wells=1479 | Table 11 |
| `results/data_quality_summary.csv` | Missing rainfall=746 (0.712%); negative GWL=57; GWL>50m=256; QC flags: 8+16 | Table 2 (data quality) |
| `results/person3/verified_metrics.csv` | Persistence MAE=2.6013; RF MAE=1.9901 (independent recalc) | Table 5 |
| `results/person3/verified_group_metrics.csv` | Familiar 5-95: n=13133, MAE=2.0011; Unfamiliar 5-95: n=1730, MAE=1.9062; High rain: n=1105, MAE=1.6625 | Table 6 |
| `results/person3/same_well_summary.csv` | n=386, mean_diff=-0.0412, median=-0.0328, frac=48.96%, CI=[-0.168,+0.080] | Table 8 |
| `results/person3/verified_bootstrap_ci.csv` | RF-Pers: -0.6361 CI=[-0.728,-0.562]; Unfam-Fam corrected: +0.1864 CI=[+0.068,+0.307]; High-rain: -0.5491 CI=[-0.715,-0.384]; Same-well: -0.041 CI=[-0.169,+0.080] | Table 7 |
| `results/person3/validation_checks.csv` | 12/12 PASS; all data integrity checks confirmed | Methodology, Abstract |
| `results/person3/rainfall_sensitivity.csv` | Full: MAE=1.9901; Excl 30d flag: MAE=1.9901 (identical); Excl 90d flag: MAE=1.9901 (identical); 0 test rows flagged | Table 9 |
| `results/person3/audit_findings.csv` | 70 audit checks: 65 PASS, 4 WARN (sensitivity type), 0 FAIL; date-column fix confirmed | Methodology §IV.E |
| `results/person3/same_well_per_well_differences.csv` | Per-well high-rain vs normal differences (386 wells) | Discussion |

### Confirmed Missing (Person 2)

| File | Status |
|------|--------|
| `results/reliability_metrics.csv` | **ABSENT** — confirmed by filesystem check |
| `results/reliability_predictions.csv` | **ABSENT** — confirmed by filesystem check |
| `results/reliability_by_condition.csv` | **ABSENT** — confirmed by filesystem check |
| `results/reliability_sensitivity.csv` | **ABSENT** — confirmed by filesystem check |

These files are absent and no Person 2 results appear anywhere in the paper.
Section IV.D explicitly documents this gap.

---

## 4. Figures Included in Paper

| Figure label | Paper/figures filename | Source path | Caption in paper |
|-------------|----------------------|-------------|-----------------|
| `fig:baseline` | `figure1_baseline_comparison.png` | `results/figures/figure1_baseline_comparison.png` | Persistence vs RF error comparison |
| `fig:novelty` | `figure2_input_novelty.png` | `results/figures/figure2_input_novelty.png` | MAE across full-feature novelty groups |
| `fig:novelty_types` | `figure3_novelty_types.png` | `results/figures/figure3_novelty_types.png` | MAE by novelty quartile for three novelty types |
| `fig:temporal` | `figure4_temporal_degradation.png` | `results/figures/figure4_temporal_degradation.png` | Temporal evolution of prediction error 2018–2020 |
| `fig:residuals` | `figure5_feature_relationships.png` | `results/figures/figure5_feature_relationships.png` | Year-dependent residual regression coefficients |
| `fig:bootstrap` | `bootstrap_results.png` | `results/figures/person3/bootstrap_results.png` | Bootstrap CIs for key comparisons |
| `fig:same_well` | `same_well_comparison.png` | `results/figures/person3/same_well_comparison.png` | Per-well MAE normal vs high-rainfall (386 wells) |

Additional figures copied to `paper/figures/` but not explicitly referenced in body text
(available for supplementary use):
- `figure6_validated_novelty.png`
- `figure7_repeated_removal.png`
- `figure8_familiar_input_temporal_comparison.png`
- `figure9_temporal_residuals.png`
- `baseline_comparison.png` (Person 3)
- `familiar_unfamiliar_comparison.png` (Person 3)
- `rainfall_comparison.png` (Person 3)
- `coverage_comparison.png` (Person 3)

---

## 5. Tables Included

| Table | Caption | Data source CSV |
|-------|---------|----------------|
| Table 1 (tab:features) | Input Features Used in the Random Forest Model | `report/research_handoff.md` |
| Table 2 (tab:data_quality) | Data Quality Summary | `results/data_quality_summary.csv` |
| Table 3 (tab:baseline) | Baseline Model Performance | `results/baseline_metrics.csv` |
| Table 4 (tab:p3metrics) | Person 3 Independent Verification Metrics | `results/person3/verified_metrics.csv` |
| Table 5 (tab:novelty_full) | Full-Feature Novelty: Errors by Group | `results/input_extrapolation_summary.csv` |
| Table 6 (tab:novelty_rain) | Rainfall-Only Novelty: Errors by Group | `results/input_extrapolation_summary.csv` |
| Table 7 (tab:novelty_type) | Novelty Type Q1 vs Q4 Comparison | `results/novelty_type_summary.csv` |
| Table 8 (tab:p3_grouped) | Person 3 Grouped Evaluation | `results/person3/verified_group_metrics.csv` |
| Table 9 (tab:bootstrap) | Bootstrap 95% CIs | `results/person3/verified_bootstrap_ci.csv` |
| Table 10 (tab:same_well) | Same-Well Paired Evaluation | `results/person3/same_well_summary.csv` |
| Table 11 (tab:qc_sensitivity) | Rainfall QC Sensitivity | `results/person3/rainfall_sensitivity.csv` |
| Table 12 (tab:removal) | Controlled Training Removal | `results/controlled_removal_metrics.csv` |
| Table 13 (tab:temporal_novelty) | Mean Prediction Error by Test Year | `results/novelty_by_year.csv` |
| Table 14 (tab:temporal_reg) | Temporal Residual Regression OLS | `results/within_well_temporal_summary.csv` |

---

## 6. References

| Cite key | Authors | Title (abbreviated) | Status |
|----------|---------|---------------------|--------|
| `famiglietti2014groundwater` | Famiglietti (2014) | The global groundwater crisis | **Verified** — Nature Climate Change 4(11):945–948, DOI 10.1038/nclimate2425 |
| `rodell2009satellite` | Rodell et al. (2009) | Satellite-based estimates of groundwater depletion in India | **Verified** — Nature 460:999–1002, DOI 10.1038/nature08238 |
| `rajaee2019review` | Rajaee et al. (2019) | A review of AI methods in groundwater level modeling | **Verified** — J. Hydrology 572:336–351, DOI 10.1016/j.jhydrol.2018.12.037 |
| `wunsch2021groundwater` | Wunsch et al. (2021) | Groundwater level forecasting with LSTM, CNN, NARX | **Verified** — HESS 25(3):1671–1687, DOI 10.5194/hess-25-1671-2021 |
| `sahoo2017machine` | Sahoo et al. (2017) | ML algorithms for groundwater level changes in US | **Verified** — Water Resources Research 53(5):3878–3895, DOI 10.1002/2016WR019933 |
| `breiman2001random` | Breiman (2001) | Random Forests | **Verified** — Machine Learning 45(1):5–32, DOI 10.1023/A:1010933404324 |
| `pedregosa2011scikit` | Pedregosa et al. (2011) | Scikit-learn: Machine Learning in Python | **Verified** — JMLR 12:2825–2830 |
| `quionero2009dataset` | Quiñonero-Candela et al. (2009) | Dataset Shift in Machine Learning | **Verified** — MIT Press book, ISBN 978-0-262-17005-5 |
| `gama2014survey` | Gama et al. (2014) | A survey on concept drift adaptation | **Verified** — ACM Computing Surveys 46(4):44:1–44:37, DOI 10.1145/2523813 |
| `vovk2005algorithmic` | Vovk et al. (2005) | Algorithmic Learning in a Random World | **Verified** — Springer book, DOI 10.1007/b106715 |
| `efron1993bootstrap` | Efron & Tibshirani (1993) | An Introduction to the Bootstrap | **Verified** — Chapman and Hall, DOI 10.1007/978-1-4899-4541-9 |

All 11 `\cite{}` keys have matching entries in `references.bib`.  
All 11 `\bibitem` entries in `references.bib` are cited in the paper body.  
No DOIs or author details have been fabricated.

---

## 7. Missing Dependencies

### Person 2 (reliability-aware modelling)

The following result files are **absent** from the repository:

- `results/reliability_metrics.csv`
- `results/reliability_predictions.csv`
- `results/reliability_by_condition.csv`
- `results/reliability_sensitivity.csv`

Planned methodology (prediction intervals, conformal coverage, interval
width analysis under familiar/unfamiliar conditions) is documented in
Section IV.D as a dependency gap.
No Person 2 results appear in any table, figure, or text claim.

### Other gaps

- Rainfall data provenance and preprocessing documentation is unresolved
  (noted in research handoff; not a blocker for paper writing).
- `Govt L V Polytechnic` extreme GWL values (up to 111.51 m) are likely
  data entry errors; they are present in the training set and noted in
  the data quality summary.

---

## 8. Limitations Documented (Section VII)

All 13 limitations are documented in Section VII of the paper:

1. Rolling one-step-ahead prediction setting
2. Novelty as feature-space distance (not physical extreme event)
3. Environmental ≠ rainfall-only novelty
4. Different eligibility criteria across experiments
5. Controlled removal: control matched by year only
6. Temporal distribution changes confound temporal comparisons
7. Residual regression coefficients ≠ direct RF feature effects
8. Potential measurement effects (procedures, quality, unobserved variables)
9. Rainfall QC sensitivity: evaluation-filter only vs retrain distinction
10. Absent Person 2 reliability-aware modelling outputs
11. Bootstrap cross-group independence underestimates CI width
12. Small high-rainfall sample per well (median = 2 observations)
13. Generalisability: single dataset, single split, Karnataka only

---

## 9. Compilation Status

**LaTeX not available on this machine.**

A systematic search found no `pdflatex`, `latexmk`, or `xelatex`
executables in `/usr`, `/opt`, `/Library`, or the user's home directory.

**To compile on a machine with LaTeX installed:**

```bash
cd /Users/Mikey/GitHub/AFML-Groundwater-Project/paper
latexmk -pdf main.tex
```

Or manually:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Known potential compilation issues to verify manually:**

1. Packages `siunitx`, `booktabs`, `multirow`, and `hyperref` must be
   installed (all are in standard TeX Live / MiKTeX distributions).
2. All `\includegraphics` targets have been verified to exist in
   `paper/figures/` before writing the paper.
3. All `\cite{}` keys match `references.bib` entries (verified in
   Section 6 above).
4. All `\label{}`/`\ref{}` and `\label{}`/`\eqref{}` pairs are consistent
   (verified by review of the LaTeX source).
5. The `IEEEtran` document class must be installed; it is part of
   standard TeX Live.

**`main.pdf` does not exist** — compilation must be performed by the user.

---

## 10. Final Output File Locations

| File | Path | Status |
|------|------|--------|
| Main paper | `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/main.tex` | ✅ Written |
| Bibliography | `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/references.bib` | ✅ Written |
| Figures dir | `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/figures/` | ✅ 15 PNGs copied |
| README | `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/README.md` | ✅ Written |
| Verification report | `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/VERIFICATION_REPORT.md` | ✅ Written |
| PDF output | `/Users/Mikey/GitHub/AFML-Groundwater-Project/paper/main.pdf` | ❌ LaTeX not installed; compile manually |

---

## 11. Data Integrity Summary

All numerical values in the paper were extracted from the CSV files
listed above.
Key values cross-checked:

| Value | Source CSV | Paper location |
|-------|-----------|----------------|
| RF MAE = 1.9901 m | `baseline_metrics.csv`, `person3/verified_metrics.csv` | Table 1, Abstract |
| Persistence MAE = 2.6013 m | `baseline_metrics.csv` | Table 1 |
| Familiar novelty MAE = 1.8155 m | `input_extrapolation_summary.csv` | Table 5 |
| Extreme novelty MAE = 3.3383 m | `input_extrapolation_summary.csv` | Table 5 |
| Environmental Q4 MAE = 2.7542 m | `novelty_type_summary.csv` | Table 7 |
| Combined Q4 MAE = 2.8737 m | `novelty_type_summary.csv` | Table 7 |
| Same-well mean diff = -0.0412 m | `person3/same_well_summary.csv` | Table 10 |
| Corrected Unfam-Fam CI = [+0.068, +0.307] m | `person3/verified_bootstrap_ci.csv` | Table 9 |
| 12/12 data integrity checks PASS | `person3/validation_checks.csv` | Methodology §IV.E, Abstract |
| Interactions model R² = 0.3044 | `within_well_temporal_summary.csv` | Table 14 |
| Removal: poorly-rep CI = [+0.031, +0.141] m | `research_handoff.md`, `controlled_removal_metrics.csv` | §V.H, Discussion |
