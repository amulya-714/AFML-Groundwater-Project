# Person 3 — Results Summary

All numbers below are from actual executed analyses. No values are invented.

---

## Baseline Performance

| Model | N | MAE (m) | RMSE (m) | R² |
|---|---:|---:|---:|---:|
| Persistence | 14,863 | 2.6013 | 4.4389 | 0.6280 |
| Random Forest | 14,863 | 1.9901 | 3.4977 | 0.7690 |

Verified independently from row-level predictions. 12/12 data integrity checks pass.

---

## Training-Referenced Novelty Results (Full 7-Feature Space)

| Group | N obs | N wells | MAE (m) | RMSE (m) | Bias (m) |
|---|---:|---:|---:|---:|---:|
| Well-represented | 13,082 | 1,768 | 1.8153 | 2.9292 | +0.045 |
| Sparse-support | 1,090 | 674 | 3.2270 | 6.0116 | +0.928 |
| Very-low-support | 691 | 503 | 3.3485 | 6.6099 | +0.794 |

Difference (Very-low − Well-represented): +1.533 m MAE.

---

## Novelty by Feature Space

| Feature space | Well-rep MAE | VLS MAE | Difference |
|---|---:|---:|---:|
| Full (7 feats) | 1.82 | 3.35 | +1.53 |
| Environmental (5 feats) | 1.89 | 2.99 | +1.10 |
| Geographic (2 feats) | 1.96 | 3.88 | +1.91 |
| Rainfall-only (2 feats) | 1.93 | 1.82 | −0.11 |

Rainfall-only novelty does NOT show higher errors for novel observations.

---

## Within-Well Novelty

| Quartile | N obs | N wells | MAE (m) |
|---|---:|---:|---:|
| Q1 (lowest) | 4,361 | 1,643 | 1.7771 |
| Q4 (highest) | 3,966 | 1,643 | 2.1879 |

Within-well paired test: mean diff +0.438 m, Wilcoxon p < 0.001, bootstrap CI [0.326, 0.547].

---

## Controlled Removal (Single Seed, Person 2)

| Comparison | Diff (m) | 95% CI (m) |
|---|---:|---|
| Extreme-removed − Full (overall) | +0.016 | [−0.004, +0.038] |
| Extreme-removed − Full (poorly rep.) | +0.082 | [+0.031, +0.141] |
| Extreme-removed − Full (extreme novelty) | +0.260 | [−0.150, +0.708] |

Single-seed overall CI includes zero → inconclusive.

---

## Repeated Controlled Removal (10 Seeds, Person 3)

| Model | Mean MAE (m) | Std MAE (m) |
|---|---:|---:|
| Full model | 1.990644 | 0.001121 |
| Extreme-removed | 2.006251 | 0.001115 |
| Matched-control | 1.991324 | 0.001639 |

| Comparison | Mean diff (m) | 95% CI (m) | Excl. zero? |
|---|---:|---|---|
| B-A (extreme − full) | +0.0156 | [+0.0144, +0.0169] | **Yes** |
| B-C (extreme − control) | +0.0149 | — | — |

10 seeds strengthen the evidence: removing novel training observations consistently degrades predictions.

---

## Temporal Deterioration

### Paired temporal (2018 vs 2020)

| Metric | Value |
|---|---:|
| Paired wells | 1,516 |
| MAE 2018 | 1.712 m |
| MAE 2020 | 2.128 m |
| Mean diff | +0.416 m |
| Bootstrap CI | [+0.298, +0.531] |

### Matched-input temporal (within well, similar inputs)

| Metric | Value |
|---|---:|
| Matched pairs | 2,077 |
| Matched wells | 1,311 |
| Error increase | +0.506 m |
| Bootstrap CI | [+0.410, +0.605] |

### Familiar-input temporal (by novelty group, Person 3)

| Novelty group | Wells | MAE 2018 | MAE 2020 | Diff (m) | 95% CI | Excl. zero? |
|---|---:|---:|---:|---:|---|---|
| Well-represented | 1,516 | 1.709 | 2.131 | +0.422 | [+0.304, +0.536] | Yes |
| Sparse-support | 130 | 2.238 | 4.298 | +2.060 | [+1.107, +3.018] | Yes |
| Very-low-support | 29 | 3.588 | 4.098 | +0.510 | [−1.688, +2.388] | No |

Temporal deterioration persists among well-represented (familiar) observations.

---

## Bootstrap Confidence Intervals (Person 3 Verified)

| Quantity | Point (m) | 95% CI (m) | Excl. zero? |
|---|---:|---|---|
| RF − Persistence MAE | −0.636 | [−0.728, −0.562] | Yes |
| Unfam − Fam MAE (corrected paired) | +0.186 | [+0.068, +0.307] | **Yes** |
| High-rain − Normal MAE (unpaired) | −0.549 | [−0.715, −0.384] | Yes |
| Same-well paired high-rain diff | −0.041 | [−0.169, +0.080] | No |
| GWL support (prev GWL, corrected) | +0.414 | [−0.716, +1.452] | No |

---

## Rainfall QC Sensitivity

| Scenario | N obs | MAE (m) | R² |
|---|---:|---:|---:|
| Full test set | 14,863 | 1.9901 | 0.7690 |
| Excl. flagged 30d (0 test rows flagged) | 14,863 | 1.9901 | 0.7690 |
| Excl. flagged 90d (0 test rows flagged) | 14,863 | 1.9901 | 0.7690 |
| Retrain excl. 16 flagged training rows | 14,863 | 1.9900 | 0.7686 |

QC flags do not affect test-set evaluation. Retrain effect is negligible (ΔMAE = −0.0001 m).

---

## Remaining Uncertainties

1. Rainfall provenance unresolved (no source data or documentation in repository)
2. Rainfall window construction cannot be independently verified
3. Govt L V Polytechnic extreme values (101 m, 111 m) retained, likely data entry errors
4. Very-low-support temporal comparison based on only 29 paired wells
5. Residual-rainfall correlations change across years but cannot establish causation
6. Unmeasured confounders: pumping, irrigation, geology, land-use changes
